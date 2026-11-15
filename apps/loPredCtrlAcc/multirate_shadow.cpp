/** \file multirate_shadow.cpp
 * \brief Upstream-counter or strict-timestamp individual-sample lifting and native DDSPC management, without I/O.
 */
#include "multirate_shadow.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace DDSPC
{
namespace
{
constexpr std::int64_t period = 500000, slot = 125000;

/// Bound cubic/quadric work before allocating either learner; native records carry at most two calibrated modes.
ShadowConfig native_model( const OnlineShadowConfig &c )
{
    auto m                   = c.m_model;
    m.m_accelChannels        = 8;
    m.m_assistedCommandSlots = 4;
    m.m_externalOrigins      = true;
    const double features    = ( double( m.m_future ) + m.m_history - 1.0 ) * 4 * m.m_modes +
                               double( m.m_history ) * m.m_modes + 8.0 * m.m_accelHistory;
    if( m.m_modes < 1 || m.m_modes > 2 || m.m_history < 1 || m.m_history > 16 || m.m_future < 1 || m.m_future > 8 ||
        m.m_accelHistory < 1 || m.m_accelHistory > 16 || features > 128 || c.m_adcCapacity < 8 ||
        c.m_adcCapacity > 8192 || c.m_queueCapacity < 16 || c.m_queueCapacity > 8192 ||
        c.m_predictionCapacity > 65536 || c.m_pendingCapacity < 2 || c.m_pendingCapacity > 128 ||
        ( m.m_standardCommandSlots != 1 && m.m_standardCommandSlots != 4 ) || c.m_periodUs < 1000 ||
        c.m_periodUs > 100000 || c.m_budgetUs < 50 || c.m_budgetUs > c.m_periodUs / 2 )
        throw std::invalid_argument( "Native DDSPC dimensions or worker budget exceed bounds" );
    return m;
}
} // namespace

bool NativeBinding::matches( const NativeObservation &r ) const
{
    return m_device && m_inode && m_generation && m_device == r.m_device && m_inode == r.m_fileInode &&
           m_image == r.m_imageInode && m_generation == r.m_generation;
}

bool native_shadow_evidence( const MultirateContract &c )
{
    if( c.m_upstream.m_combinedRequestModel )
        return c.m_upstream.m_enabled && c.m_inputs.m_evidence == InputEvidence::Requested;
    return c.m_inputs.m_evidence == InputEvidence::VerifiedApplied;
}

struct MultirateShadow::Impl : public ShadowPredictionSink
{
    struct Origin
    {
        OnlineOrigin m_source;            ///< Immutable upstream/origin evidence; exposures only in strict mode.
        Matrix       m_acceleration;      ///< Four distinct ADC pairs frozen at the origin, newest sample first.
        bool         m_adcValid{ false }; ///< Complete, causally available four-sample group.
    };
    OnlineShadowConfig    m_config;      ///< Numerical and memory bounds.
    ShadowConfig          m_modelConfig; ///< Explicit native lift; single-rate defaults elsewhere remain unchanged.
    InputEvidence         m_allowed;  ///< Production physical evidence or an explicit offline-only synthetic instance.
    MultirateContract     m_contract; ///< Independently established source contract, never inferred from raw streams.
    ShadowController      m_models;   ///< Actual RLS/QR DDSPC pair.
    ShadowPredictionSink *m_sink{ nullptr }; ///< Borrowed downstream diagnostics consumer; numerical-thread-only.
    OnlineShadowStatus    m_status;          ///< Bounded readback and inhibition reasons.
    std::array<std::vector<NativeObservation>, 4> m_raw; ///< Per-source fixed retention, including every ADC sample.
    std::array<std::size_t, 4>                    m_next{}, m_count{}; ///< Ring cursors; histories are not work queues.
    std::deque<Origin>            m_origins; ///< Bounded frozen origins awaiting independently observed inputs.
    std::deque<OnlineAppliedSlot> m_inputs;  ///< Bounded independently confirmed non-overlapping slot intervals.
    std::array<Matrix, 2> m_plans; ///< Separate newest-first future command plans; never returned as applied inputs.
    std::array<bool, 2>   m_forecastValid{}; ///< Per-mode diagnostic forecasts, never output authority.
    std::array<bool, 2>   m_planValid{};     ///< Per-mode current causal/deadline eligibility.
    std::int64_t          m_now{ -1 }, m_lastOriginNs{ -1 }, m_lastInputEnd{ -1 }; ///< Monotonic/session high waters.
    std::uint64_t         m_lastOrigin{ 0 }, m_lastCertificate{ 0 }; ///< Independent WFS/certificate sequences.
    std::int64_t  m_latestModelReceipt{ -1 }; ///< High-water availability of all evidence used by histories/fit.
    std::uint64_t m_lastComplete{ 0 };        ///< Last complete frame in the admitted request/physical input domain.
    unsigned      m_goodSync{ 0 }; ///< Consecutive complete upstream-locked input groups, not a phase-lock algorithm.

    /// Allocate all fixed storage outside the acquisition worker.
    Impl( const OnlineShadowConfig &c /**< [in] Fixed resource bounds. */,
          InputEvidence             evidence /**< [in] Required physical or explicitly synthetic provenance. */ );

    /// Inspect a retained counter without inferring any parent/source relationship.
    const NativeObservation *find( std::size_t   source /**< [in] Raw source index. */,
                                   std::uint64_t counter /**< [in] Original source publication. */ ) const;

    /// Break model and pending histories while preserving coefficient state and source high waters.
    void invalidate( const char *reason /**< [in] Stable inhibition description. */ );

    /// Validate common availability/identity against the reviewed contract.
    bool stamp( const InputStamp &s /**< [in] Independent evidence metadata. */,
                std::uint64_t     generation /**< [in] Expected source incarnation. */,
                std::int64_t      now /**< [in] Actual processing time. */ ) const;

    /// Form four counter- or strict-timestamp-selected raw pairs, without averaging or future receipt.
    bool acceleration( const OnlineOrigin &origin /**< [in] Frozen WFS origin. */,
                       Matrix             &features /**< [out] Four distinct newest-first ADC pairs. */ ) const;

    /// Consume exactly covered closed intervals and update both models with the same actual input lift.
    void complete();

    /// Attach source/contract identity without changing a forecast or doing I/O.
    void prediction( const ShadowPrediction &sample /**< [in] Original conditional score. */ ) noexcept override;
};

void MultirateShadow::Impl::prediction( const ShadowPrediction &sample ) noexcept
{
    if( !m_sink )
        return;
    auto tagged         = sample;
    tagged.m_generation = m_contract.m_inputs.m_wfsGeneration;
    tagged.m_epoch = m_contract.m_upstream.m_enabled ? m_contract.m_upstream.m_epoch : m_contract.m_inputs.m_clockEpoch;
    tagged.m_layout = m_contract.m_inputs.m_layout;
    m_sink->prediction( tagged );
}

MultirateShadow::Impl::Impl( const OnlineShadowConfig &c, InputEvidence evidence )
    : m_config( c ), m_modelConfig( native_model( c ) ), m_allowed( evidence ), m_models( m_modelConfig )
{
    if( evidence != InputEvidence::VerifiedApplied && evidence != InputEvidence::SyntheticApplied )
        throw std::invalid_argument( "Publication and SDK evidence cannot admit native DDSPC" );
    for( std::size_t i = 0; i < 4; ++i )
        m_raw[i].resize( i ? 256 : c.m_adcCapacity );
    m_plans[0] =
        Matrix::Zero( m_modelConfig.m_future * m_modelConfig.m_standardCommandSlots * m_modelConfig.m_modes, 1 );
    m_plans[1] = Matrix::Zero( m_modelConfig.m_future * 4 * m_modelConfig.m_modes, 1 );
}

const NativeObservation *MultirateShadow::Impl::find( std::size_t source, std::uint64_t counter ) const
{
    for( std::size_t n = 0; n < m_count[source]; ++n )
    {
        const auto &r = m_raw[source][( m_next[source] + m_raw[source].size() - n - 1 ) % m_raw[source].size()];
        if( r.m_counter == counter )
            return &r;
    }
    return nullptr;
}

void MultirateShadow::Impl::invalidate( const char *reason )
{
    m_models.invalidate( {} );
    m_origins.clear();
    m_inputs.clear();
    m_planValid.fill( false );
    m_forecastValid.fill( false );
    m_status.m_predictionValid = false;
    m_lastComplete             = 0;
    m_status.m_candidateValid  = false;
    m_status.m_reason          = reason;
    m_goodSync                 = 0;
    m_status.m_syncValid       = false;
}

bool MultirateShadow::Impl::stamp( const InputStamp &s, std::uint64_t generation, std::int64_t now ) const
{
    const auto &c = m_contract.m_inputs;
    return s.m_snapshotValid && s.m_counter > 0 && s.m_counter < std::numeric_limits<std::uint64_t>::max() &&
           s.m_generation == generation && s.m_clockEpoch == c.m_clockEpoch && s.m_layout == c.m_layout &&
           s.m_availableNs >= 0 && s.m_receivedNs >= s.m_availableNs && now >= s.m_receivedNs &&
           s.m_uncertaintyNs >= 0 && s.m_uncertaintyNs <= c.m_maxUncertaintyNs;
}

bool MultirateShadow::Impl::acceleration( const OnlineOrigin &o, Matrix &features ) const
{
    features.resize( 8, 1 );
    if( m_contract.m_upstream.m_enabled )
    {
        const auto &u         = m_contract.m_upstream;
        const auto  reference = u.m_referenceIsModal ? o.m_stamp.m_counter : o.m_cameraCounter;
        const auto  anchor    = u.m_referenceIsModal ? u.m_modalAnchor : u.m_referenceAnchor;
        if( !u.m_adcAnchor || !o.m_synchronized || !o.m_locked || o.m_syncEpoch != u.m_epoch ||
            o.m_syncStatusReceivedNs < 0 || o.m_syncStatusReceivedNs > o.m_decisionNs ||
            o.m_decisionNs - o.m_syncStatusReceivedNs > u.m_statusMaxAgeNs || reference < anchor ||
            reference - anchor > ( std::numeric_limits<std::uint64_t>::max() - u.m_adcAnchor - 3 ) / 4 ||
            o.m_firstAdcCounter != u.m_adcAnchor + 4 * ( reference - anchor ) )
            return false;
        std::uint64_t segment = 0;
        for( unsigned s = 0; s < 4; ++s )
        {
            const auto *r = find( 0, o.m_firstAdcCounter + s );
            if( !r || !m_contract.m_sources[0].matches( *r ) || r->m_receivedNs < 0 || r->m_availableNs < 0 ||
                r->m_receivedNs > o.m_decisionNs || r->m_availableNs > o.m_decisionNs ||
                o.m_decisionNs - r->m_receivedNs > u.m_featureMaxAgeNs || ( s && r->m_segment != segment ) ||
                !std::isfinite( r->m_values[0] ) || !std::isfinite( r->m_values[1] ) )
                return false;
            segment                       = r->m_segment;
            features( 2 * ( 3 - s ) )     = r->m_values[0];
            features( 2 * ( 3 - s ) + 1 ) = r->m_values[1];
        }
        return true;
    }
    const auto   &ring    = m_raw[0];
    const auto    count   = m_count[0];
    const auto    first   = ( m_next[0] + ring.size() - count ) % ring.size();
    const auto   &c       = m_contract.m_inputs;
    std::uint64_t counter = 0, segment = 0;
    for( int s = 0; s < 4; ++s )
    {
        const auto  target = o.m_gridNs - period + m_contract.m_adcPhaseNs + s * slot;
        std::size_t lo = 0, hi = count;
        while( lo < hi )
        {
            const auto mid = ( lo + hi ) / 2;
            if( ring[( first + mid ) % ring.size()].m_acquiredMonoNs < target - c.m_jitterNs )
                lo = mid + 1;
            else
                hi = mid;
        }
        if( lo == count )
            return false;
        const auto &r = ring[( first + lo ) % ring.size()];
        if( !m_contract.m_sources[0].matches( r ) || r.m_acquiredMonoNs < 0 ||
            std::abs( r.m_acquiredMonoNs - target ) + c.m_maxUncertaintyNs > c.m_jitterNs || r.m_receivedNs < 0 ||
            r.m_availableNs < 0 || r.m_acquiredMonoNs > r.m_receivedNs || r.m_receivedNs > o.m_decisionNs ||
            r.m_availableNs > o.m_decisionNs ||
            ( counter && ( r.m_counter != counter + 1 || r.m_segment != segment ) ) ||
            !std::isfinite( r.m_values[0] ) || !std::isfinite( r.m_values[1] ) )
            return false;
        // More than one sample in a phase bin is ambiguous, not an opportunity to select a favorable value.
        if( lo + 1 < count && ring[( first + lo + 1 ) % ring.size()].m_acquiredMonoNs <= target + c.m_jitterNs )
            return false;
        counter                       = r.m_counter;
        segment                       = r.m_segment;
        features( 2 * ( 3 - s ) )     = r.m_values[0];
        features( 2 * ( 3 - s ) + 1 ) = r.m_values[1];
    }
    return true;
}

void MultirateShadow::Impl::complete()
{
    const int modes = m_modelConfig.m_modes;
    while( !m_origins.empty() )
    {
        const auto &origin = m_origins.front();
        const auto  start  = origin.m_source.m_gridNs;
        while( !m_inputs.empty() && m_inputs.front().m_endNs <= start )
            m_inputs.pop_front();
        if( m_inputs.size() < 4 )
            return;
        ShadowRecord record;
        record.m_index = origin.m_source.m_stamp.m_counter;
        record.m_residual.resize( modes, 1 );
        record.m_appliedCommand.resize( 4 * modes, 1 );
        for( int s = 0; s < 4; ++s )
        {
            const auto &u = m_inputs[s];
            if( u.m_startNs != start + s * slot || u.m_endNs != start + ( s + 1 ) * slot )
            {
                invalidate( "missing_or_ambiguous_applied_slot_coverage" );
                return;
            }
            for( int m = 0; m < modes; ++m )
                record.m_appliedCommand( ( 3 - s ) * modes + m ) = u.m_values[m];
        }
        for( int m = 0; m < modes; ++m )
            record.m_residual( m ) = origin.m_source.m_residual[m];
        record.m_assistedAppliedCommand = record.m_appliedCommand;
        record.m_acceleration           = origin.m_acceleration;
        record.m_wfsValid = record.m_appliedValid = record.m_assistedAppliedValid = record.m_timingValid = true;
        if( m_modelConfig.m_standardCommandSlots == 1 )
        {
            for( int s = 1; s < 4; ++s )
                for( int m = 0; m < modes; ++m )
                    record.m_appliedValid = record.m_appliedValid && m_inputs[s].m_values[m] == m_inputs[0].m_values[m];
            record.m_appliedCommand = record.m_assistedAppliedCommand.bottomRows( modes ).eval();
        }
        record.m_accelValid = origin.m_adcValid;
        m_models.process( record );
        m_latestModelReceipt = std::max( m_latestModelReceipt, origin.m_source.m_decisionNs );
        for( int s = 0; s < 4; ++s )
            m_latestModelReceipt = std::max( m_latestModelReceipt, m_inputs[s].m_stamp.m_receivedNs );
        m_lastComplete = record.m_index;
        ++m_status.m_records;
        m_origins.pop_front();
        for( int s = 0; s < 4; ++s )
            m_inputs.pop_front();
    }
}

MultirateShadow::MultirateShadow( const OnlineShadowConfig &c, InputEvidence evidence )
    : m_impl( std::make_unique<Impl>( c, evidence ) )
{
}
MultirateShadow::~MultirateShadow() = default;

void MultirateShadow::prediction_sink( ShadowPredictionSink *sink )
{
    m_impl->m_sink = sink;
    m_impl->m_models.prediction_sink( sink ? m_impl.get() : nullptr );
}

bool MultirateShadow::qualify( const MultirateContract &c )
{
    auto &p = *m_impl;
    p.invalidate( "source_host_mapping_unqualified" );
    p.m_status.m_qualified   = false;
    p.m_status.m_clock       = c.m_clockVerified;
    p.m_status.m_exposure    = c.m_exposureVerified;
    p.m_status.m_association = c.m_associationVerified;
    p.m_status.m_schedule    = c.m_scheduleVerified;
    p.m_status.m_upstream    = c.m_upstream.m_enabled;
    if( c.m_provenance.size() > 4096 )
        return false;
    if( !c.m_upstream.m_enabled )
    {
        if( !c.m_clockVerified )
            return false;
        p.m_status.m_reason = "exposure_boundaries_unavailable";
        if( !c.m_exposureVerified )
            return false;
        p.m_status.m_reason = "modal_camera_association_missing";
        if( !c.m_associationVerified )
            return false;
    }
    else
    {
        const auto &u       = c.m_upstream;
        p.m_status.m_reason = "upstream_sync_counter_contract_unqualified";
        if( !u.m_epoch || !u.m_modalAnchor || ( !u.m_referenceIsModal && !u.m_referenceAnchor ) ||
            u.m_adcAnchor > std::numeric_limits<std::uint64_t>::max() - 3 || !u.m_recoveryFrames ||
            u.m_recoveryFrames > 128 || u.m_statusMaxAgeNs <= 0 || u.m_statusMaxAgeNs > 2000000000 ||
            u.m_featureMaxAgeNs < period || u.m_featureMaxAgeNs > 2 * period )
            return false;
    }
    const auto &i        = c.m_inputs;
    p.m_status.m_reason  = "applied_input_provenance_or_modal_projection_unqualified";
    const bool requested = c.m_upstream.m_enabled && c.m_upstream.m_combinedRequestModel &&
                           p.m_allowed == InputEvidence::VerifiedApplied && i.m_evidence == InputEvidence::Requested;
    if( c.m_upstream.m_combinedRequestModel && !requested &&
        !( p.m_allowed == InputEvidence::SyntheticApplied && i.m_evidence == InputEvidence::SyntheticApplied ) )
        return false;
    if( ( i.m_evidence != p.m_allowed && !requested ) || !i.m_timingVerified || !i.m_mappingVerified ||
        c.m_provenance.empty() )
        return false;
    if( i.m_modes != p.m_modelConfig.m_modes || i.m_periodNs != period ||
        ( !c.m_upstream.m_enabled && ( i.m_exposureNs <= 0 || i.m_exposureNs > period ) ) || i.m_jitterNs < 0 ||
        i.m_jitterNs >= slot / 2 || i.m_maxAgeNs < period || i.m_maxAgeNs > 1000000000 || i.m_maxUncertaintyNs < 0 ||
        i.m_maxUncertaintyNs > i.m_jitterNs || !i.m_clockEpoch || !i.m_layout || !i.m_wfsGeneration ||
        !i.m_commandGeneration || i.m_responseLagNs != 0 || i.m_minCommandDelayNs != 0 ||
        c.m_wfsEndOffsetNs < -2 * period || c.m_wfsEndOffsetNs > 2 * period || c.m_adcPhaseNs < 0 ||
        c.m_adcPhaseNs >= slot )
        throw std::invalid_argument( "Unsupported native multirate timing/model contract" );
    for( std::size_t n = 0; n < 4; ++n )
    {
        if( n == 0 && c.m_upstream.m_enabled && !c.m_upstream.m_adcAnchor )
            continue; // Standard model needs no ADC binding.
        if( n == 2 && c.m_upstream.m_enabled && c.m_upstream.m_referenceIsModal )
            continue;
        if( !c.m_sources[n].m_device || !c.m_sources[n].m_inode || !c.m_sources[n].m_generation ||
            ( n && !c.m_sources[n].m_image ) )
            return false;
    }
    if( c.m_sources[1].m_generation != i.m_wfsGeneration )
        return false;
    p.m_contract = c;
    // Old calibrated/mapped sensor samples cannot be relabelled with a new clock/layout contract.
    // Retained DM requests bind identity only; actual values/times still require new independent certificates.
    for( std::size_t n = 0; n < 3; ++n )
        p.m_count[n] = p.m_next[n] = 0;
    p.m_now = p.m_lastOriginNs = p.m_lastInputEnd = -1;
    p.m_latestModelReceipt                        = -1;
    p.m_lastOrigin = p.m_lastCertificate = 0;
    // A new calibration/clock/source contract must never reuse old weights merely because dimensions match.
    p.m_models.reset( ShadowMode::Standard );
    p.m_models.reset( ShadowMode::AccelAssisted );
    p.m_models.training( ShadowMode::Standard, true );
    p.m_models.training( ShadowMode::AccelAssisted, true );
    p.m_status.m_qualified     = true;
    p.m_status.m_inputEvidence = i.m_evidence;
    p.m_status.m_reason        = c.m_upstream.m_enabled ? "awaiting_upstream_frames_and_command_history"
                                                        : "awaiting_qualified_exposures_and_applied_slots";
    return true;
}

void MultirateShadow::observe( const NativeObservation &r )
{
    auto &p = *m_impl;
    ++p.m_status.m_raw;
    const auto i = std::size_t( r.m_source );
    if( i >= 4 )
        return; // Software completion is diagnostic; it is never converted into an applied certificate.
    if( i == 2 && p.m_contract.m_upstream.m_enabled && p.m_contract.m_upstream.m_referenceIsModal )
        return; // Camera lineage is diagnostic-only for the synchronized modal-sample convention.
    const auto *last =
        p.m_count[i] ? &p.m_raw[i][( p.m_next[i] + p.m_raw[i].size() - 1 ) % p.m_raw[i].size()] : nullptr;
    const bool changed = last && ( r.m_generation != last->m_generation || r.m_fileInode != last->m_fileInode ||
                                   r.m_device != last->m_device || r.m_imageInode != last->m_imageInode );
    if( last && !changed && r.m_event == ObservationEvent::Sample && r.m_counter == last->m_counter )
        return;
    const bool gap = last && !changed && r.m_event == ObservationEvent::Sample && r.m_counter != last->m_counter + 1;
    const bool timeBackwards =
        last && r.m_acquiredMonoNs >= 0 && last->m_acquiredMonoNs >= 0 && r.m_acquiredMonoNs < last->m_acquiredMonoNs;
    if( r.m_event != ObservationEvent::Sample || changed || gap || timeBackwards ||
        ( r.m_flags & ( ObservationInvalidPayload | ObservationInvalidTime | ObservationTimestampDiscontinuity |
                        ObservationClockMismatch ) ) )
    {
        p.m_count[i] = p.m_next[i] = 0;
        if( i == 0 )
        {
            p.m_models.invalidate_acceleration();
            p.m_goodSync                 = 0;
            p.m_status.m_syncValid       = false;
            p.m_planValid[1]             = false;
            p.m_forecastValid[1]         = false;
            p.m_status.m_predictionValid = p.m_forecastValid[0];
            p.m_status.m_candidateValid  = p.m_planValid[0];
            p.m_status.m_effective       = ShadowMode::Standard;
            if( p.m_status.m_qualified )
                p.m_status.m_reason = "adc_discontinuity_standard_only_if_current_plan_valid";
        }
        else
            p.invalidate( p.m_status.m_qualified ? "native_source_discontinuity" : p.m_status.m_reason );
    }
    if( i != 0 && p.m_status.m_qualified && !p.m_contract.m_sources[i].matches( r ) )
    {
        p.invalidate( "source_identity_changed_new_contract_required" );
        p.m_status.m_qualified = false;
        p.m_status.m_clock = p.m_status.m_exposure = p.m_status.m_association = p.m_status.m_schedule = false;
    }
    if( r.m_event != ObservationEvent::Sample ||
        ( r.m_flags & ( ObservationInvalidPayload | ObservationInvalidTime | ObservationClockMismatch ) ) )
        return;
    p.m_raw[i][p.m_next[i]] = r;
    p.m_next[i]             = ( p.m_next[i] + 1 ) % p.m_raw[i].size();
    p.m_count[i]            = std::min( p.m_count[i] + 1, p.m_raw[i].size() );
}

bool MultirateShadow::origin( const OnlineOrigin &o, std::int64_t now )
{
    auto &p = *m_impl;
    advance( now );
    if( !p.m_status.m_qualified )
        return false;
    const auto &c              = p.m_contract.m_inputs;
    const auto *modal          = p.find( 1, o.m_stamp.m_counter );
    const auto *camera         = p.find( 2, o.m_cameraCounter );
    const bool  upstream       = p.m_contract.m_upstream.m_enabled;
    const auto &u              = p.m_contract.m_upstream;
    const auto  originTime     = upstream ? o.m_decisionNs : o.m_gridNs;
    const bool  modalReference = upstream && u.m_referenceIsModal;
    const bool  association =
        !upstream ||
        ( o.m_stamp.m_counter >= u.m_modalAnchor &&
          ( modalReference ? o.m_cameraCounter == 0
                           : ( o.m_cameraCounter >= u.m_referenceAnchor &&
                               o.m_stamp.m_counter - u.m_modalAnchor == o.m_cameraCounter - u.m_referenceAnchor ) ) );
    const bool cameraValid =
        modalReference || ( camera && p.m_contract.m_sources[2].matches( *camera ) && camera->m_availableNs >= 0 &&
                            camera->m_receivedNs >= 0 && camera->m_availableNs <= o.m_decisionNs &&
                            camera->m_receivedNs <= o.m_decisionNs );
    const bool exposure =
        upstream || ( o.m_gridNs >= period && o.m_gridNs <= std::numeric_limits<std::int64_t>::max() - 2 * period &&
                      o.m_exposureStartNs >= 0 && o.m_exposureEndNs > o.m_exposureStartNs &&
                      o.m_exposureEndNs - o.m_exposureStartNs == c.m_exposureNs &&
                      o.m_exposureEndNs - o.m_gridNs == p.m_contract.m_wfsEndOffsetNs &&
                      o.m_exposureEndNs <= o.m_stamp.m_availableNs );
    if( !association || !exposure || !p.stamp( o.m_stamp, c.m_wfsGeneration, now ) || !modal || !cameraValid ||
        !p.m_contract.m_sources[1].matches( *modal ) || modal->m_availableNs < 0 || modal->m_receivedNs < 0 ||
        modal->m_availableNs > o.m_decisionNs || modal->m_receivedNs > o.m_decisionNs ||
        o.m_decisionNs < o.m_stamp.m_receivedNs || now < o.m_decisionNs ||
        o.m_decisionNs - o.m_stamp.m_receivedNs > c.m_maxAgeNs ||
        ( p.m_lastOrigin && ( o.m_stamp.m_counter <= p.m_lastOrigin || originTime <= p.m_lastOriginNs ) ) )
    {
        ++p.m_status.m_rejections;
        p.invalidate( upstream ? "invalid_upstream_reference_order_or_availability"
                               : "invalid_exposure_association_clock_or_counter" );
        return false;
    }
    if( upstream && !modalReference )
        p.m_status.m_association = true;
    if( upstream && ( o.m_stamp.m_availableNs < modal->m_availableNs || o.m_stamp.m_receivedNs < modal->m_receivedNs ) )
    {
        p.invalidate( "upstream_measurement_availability_backdated" );
        return false;
    }
    for( int m = 0; m < p.m_modelConfig.m_modes; ++m )
        if( !std::isfinite( o.m_residual[m] ) || realT( modal->m_values[m] ) != o.m_residual[m] )
        {
            p.invalidate( "modal_payload_not_bound_to_observed_frame" );
            return false;
        }
    if( p.m_lastOrigin &&
        ( o.m_stamp.m_counter != p.m_lastOrigin + 1 || ( !upstream && o.m_gridNs - p.m_lastOriginNs != period ) ) )
        p.invalidate( "wfs_counter_gap_new_history_segment" );
    p.m_lastOrigin   = o.m_stamp.m_counter;
    p.m_lastOriginNs = originTime;
    if( !upstream )
        p.complete();
    Impl::Origin frozen;
    frozen.m_source   = o;
    frozen.m_adcValid = p.acceleration( o, frozen.m_acceleration );
    if( upstream )
    {
        p.m_goodSync           = frozen.m_adcValid ? std::min( p.m_goodSync + 1, u.m_recoveryFrames ) : 0;
        frozen.m_adcValid      = frozen.m_adcValid && p.m_goodSync >= u.m_recoveryFrames;
        p.m_status.m_syncValid = frozen.m_adcValid;
    }
    if( !frozen.m_adcValid )
        p.m_models.invalidate_acceleration();
    p.m_planValid.fill( false );
    p.m_status.m_candidateValid = false;
    p.m_status.m_originNs       = o.m_decisionNs;
    p.m_status.m_reason =
        !frozen.m_adcValid ? "adc_phase_history_incomplete_or_unavailable_at_origin" : "models_warming_or_unexcited";
    bool schedule = p.m_status.m_schedule && o.m_gridNs >= period &&
                    o.m_gridNs <= std::numeric_limits<std::int64_t>::max() - 2 * period && o.m_prefixSlots <= 4 &&
                    o.m_prefixSlots < unsigned( 4 * p.m_modelConfig.m_future ) &&
                    ( !o.m_prefixSlots ||
                      ( o.m_prefixVerified && o.m_prefixAvailableNs <= o.m_decisionNs && o.m_prefixAvailableNs >= 0 ) );
    for( unsigned i = 0; i < std::min<unsigned>( o.m_prefixSlots, 4 ) * p.m_modelConfig.m_modes; ++i )
        schedule = schedule && std::isfinite( o.m_prefix[i] );
    const unsigned advertised = o.m_prefixSlots == 4 ? 8 : 4;
    for( unsigned s = 0; s < advertised; ++s )
        schedule = schedule && o.m_cutoffsNs[s] > 0 && o.m_cutoffsNs[s] <= o.m_gridNs + s * slot &&
                   ( !s || o.m_cutoffsNs[s] > o.m_cutoffsNs[s - 1] );
    p.m_status.m_deadlineNs  = schedule ? o.m_cutoffsNs[o.m_prefixSlots] : -1;
    const bool pastAvailable = p.m_lastComplete + 1 == o.m_stamp.m_counter && p.m_latestModelReceipt <= o.m_decisionNs;
    bool       diagnostic    = false;
    p.m_forecastValid.fill( false );
    p.m_status.m_predictionValid = false;
    if( upstream && pastAvailable )
    {
        Matrix y( p.m_modelConfig.m_modes, 1 ), empty;
        for( int m = 0; m < y.rows(); ++m )
            y( m ) = o.m_residual[m];
        const bool late      = o.m_predictionDeadlineNs < 0 || now > o.m_predictionDeadlineNs;
        p.m_forecastValid[0] = p.m_models.freeze_next( ShadowMode::Standard, o.m_stamp.m_counter, y, empty, late );
        p.m_forecastValid[1] =
            frozen.m_adcValid &&
            p.m_models.freeze_next( ShadowMode::AccelAssisted, o.m_stamp.m_counter, y, frozen.m_acceleration, late );
        const bool assisted          = p.m_status.m_requested == ShadowMode::AccelAssisted && p.m_forecastValid[1];
        p.m_status.m_effective       = assisted ? ShadowMode::AccelAssisted : ShadowMode::Standard;
        p.m_status.m_predictionValid = assisted || p.m_forecastValid[0];
        p.m_status.m_predictionLate  = late;
        diagnostic                   = p.m_forecastValid[0] || p.m_forecastValid[1];
        p.m_status.m_forecasts += diagnostic;
        p.m_status.m_lateForecasts += diagnostic && late;
    }
    if( schedule && now <= p.m_status.m_deadlineNs && pastAvailable )
    {
        Matrix y( p.m_modelConfig.m_modes, 1 ), prefix( o.m_prefixSlots * p.m_modelConfig.m_modes, 1 ), empty;
        for( int m = 0; m < y.rows(); ++m )
            y( m ) = o.m_residual[m];
        for( unsigned s = 0; s < o.m_prefixSlots; ++s )
            for( int m = 0; m < y.rows(); ++m )
                prefix( ( o.m_prefixSlots - s - 1 ) * y.rows() + m ) = o.m_prefix[s * y.rows() + m];
        if( p.m_modelConfig.m_standardCommandSlots == 4 )
            p.m_planValid[0] =
                p.m_models.predict_next( ShadowMode::Standard, o.m_stamp.m_counter, y, empty, p.m_plans[0], &prefix );
        // A single-rate proposal cannot independently change a partially committed WFS-frame input.
        else if( !o.m_prefixSlots )
            p.m_planValid[0] =
                p.m_models.predict_next( ShadowMode::Standard, o.m_stamp.m_counter, y, empty, p.m_plans[0] );
        else if( o.m_prefixSlots == 4 )
        {
            bool held = true;
            for( int s = 1; s < 4; ++s )
                held = held && prefix.middleRows( s * y.rows(), y.rows() ) == prefix.topRows( y.rows() );
            if( held )
            {
                Matrix fixed     = prefix.bottomRows( y.rows() );
                p.m_planValid[0] = p.m_models.predict_next(
                    ShadowMode::Standard, o.m_stamp.m_counter, y, empty, p.m_plans[0], &fixed );
            }
        }
        p.m_planValid[1] =
            frozen.m_adcValid &&
            p.m_models.predict_next(
                ShadowMode::AccelAssisted, o.m_stamp.m_counter, y, frozen.m_acceleration, p.m_plans[1], &prefix );
        const bool assisted         = p.m_status.m_requested == ShadowMode::AccelAssisted && p.m_planValid[1];
        p.m_status.m_effective      = assisted ? ShadowMode::AccelAssisted : ShadowMode::Standard;
        p.m_status.m_candidateValid = assisted || p.m_planValid[0];
        if( p.m_status.m_candidateValid )
        {
            ++p.m_status.m_candidates;
            p.m_status.m_reason = assisted ? "assisted_shadow_candidate" : "standard_shadow_candidate_or_fallback";
        }
        else if( p.m_modelConfig.m_standardCommandSlots == 1 && o.m_prefixSlots > 0 && o.m_prefixSlots < 4 &&
                 p.m_models.status( ShadowMode::Standard ).m_ready )
            p.m_status.m_reason = "single_rate_standard_cannot_change_partially_committed_frame";
    }
    else
        p.m_status.m_reason = !pastAvailable ? "past_applied_frame_not_confirmed_at_frozen_origin"
                              : schedule     ? "input_ready_cutoff_already_expired"
                                             : "synco_slot_or_committed_prefix_unqualified";
    if( upstream && diagnostic && !p.m_status.m_candidateValid )
        p.m_status.m_reason = "shadow_forecast_only_no_eligible_synco_plan";
    if( p.m_origins.size() == p.m_config.m_pendingCapacity )
    {
        p.invalidate( "qualified_origin_queue_overflow" );
        return false;
    }
    p.m_origins.push_back( std::move( frozen ) );
    return true;
}

bool MultirateShadow::commands( const OnlineCommandFrame &u, std::int64_t now )
{
    auto &p = *m_impl;
    advance( now );
    if( !p.m_status.m_qualified || !p.m_contract.m_upstream.m_enabled ||
        !p.m_contract.m_upstream.m_combinedRequestModel )
        return false;
    const auto &c = p.m_contract.m_inputs;
    bool valid    = u.m_complete && p.stamp( u.m_stamp, c.m_commandGeneration, now ) && u.m_evidence == c.m_evidence &&
                    !p.m_origins.empty() && p.m_origins.front().m_source.m_stamp.m_counter == u.m_wfsCounter &&
                    ( !p.m_lastCertificate || u.m_stamp.m_counter > p.m_lastCertificate );
    std::int64_t closingAvailable = -1;
    if( valid )
    {
        const auto reference =
            p.m_contract.m_upstream.m_referenceIsModal ? u.m_wfsCounter : p.m_origins.front().m_source.m_cameraCounter;
        const auto  source = p.m_contract.m_upstream.m_referenceIsModal ? 1 : 2;
        const auto *closed =
            reference < std::numeric_limits<std::uint64_t>::max() ? p.find( source, reference + 1 ) : nullptr;
        valid = closed && p.m_contract.m_sources[source].matches( *closed ) && closed->m_receivedNs >= 0 &&
                closed->m_receivedNs <= u.m_stamp.m_receivedNs;
        if( valid )
            closingAvailable = closed->m_availableNs;
    }
    for( unsigned s = 0; s < 4; ++s )
    {
        const auto *request = p.find( 3, u.m_counters[s] );
        valid = valid && request && p.m_contract.m_sources[3].matches( *request ) && u.m_hashes[s] &&
                request->m_payloadHash == u.m_hashes[s] && request->m_receivedNs >= 0 && u.m_sampledNs[s] >= 0 &&
                request->m_availableNs >= 0 && request->m_receivedNs <= u.m_sampledNs[s] &&
                request->m_availableNs <= u.m_sampledNs[s] && u.m_sampledNs[s] <= closingAvailable &&
                ( !s || ( u.m_counters[s] >= u.m_counters[s - 1] && u.m_sampledNs[s] > u.m_sampledNs[s - 1] ) );
        for( std::size_t n = 0; valid && n < p.m_count[3]; ++n )
        {
            const auto &other = p.m_raw[3][( p.m_next[3] + p.m_raw[3].size() - n - 1 ) % p.m_raw[3].size()];
            if( other.m_counter > u.m_counters[s] && other.m_availableNs <= u.m_sampledNs[s] )
                valid = false;
        }
        for( int m = 0; m < p.m_modelConfig.m_modes; ++m )
            valid = valid && std::isfinite( u.m_values[s * p.m_modelConfig.m_modes + m] );
    }
    if( !valid )
    {
        ++p.m_status.m_rejections;
        p.invalidate( "combined_request_history_identity_coverage_or_sequence_invalid" );
        return false;
    }
    const auto &frozen = p.m_origins.front();
    if( p.m_lastCertificate && u.m_stamp.m_counter != p.m_lastCertificate + 1 )
    {
        p.m_models.invalidate( {} );
        p.m_planValid.fill( false );
        p.m_status.m_candidateValid = false;
    }
    const int    modes = p.m_modelConfig.m_modes;
    ShadowRecord record;
    record.m_index = u.m_wfsCounter;
    record.m_residual.resize( modes, 1 );
    record.m_assistedAppliedCommand.resize( 4 * modes, 1 );
    for( int m = 0; m < modes; ++m )
        record.m_residual( m ) = frozen.m_source.m_residual[m];
    for( unsigned s = 0; s < 4; ++s )
        for( int m = 0; m < modes; ++m )
            record.m_assistedAppliedCommand( ( 3 - s ) * modes + m ) = u.m_values[s * modes + m];
    record.m_appliedCommand = record.m_assistedAppliedCommand;
    record.m_wfsValid = record.m_timingValid = record.m_appliedValid = record.m_assistedAppliedValid = true;
    if( p.m_modelConfig.m_standardCommandSlots == 1 )
    {
        for( unsigned s = 1; s < 4; ++s )
            for( int m = 0; m < modes; ++m )
                record.m_appliedValid = record.m_appliedValid && u.m_values[s * modes + m] == u.m_values[m];
        record.m_appliedCommand = record.m_assistedAppliedCommand.bottomRows( modes ).eval();
    }
    record.m_acceleration = frozen.m_acceleration;
    record.m_accelValid   = frozen.m_adcValid;
    p.m_models.process( record );
    p.m_latestModelReceipt =
        std::max( { p.m_latestModelReceipt, frozen.m_source.m_decisionNs, u.m_stamp.m_receivedNs } );
    p.m_lastComplete    = record.m_index;
    p.m_lastCertificate = u.m_stamp.m_counter;
    ++p.m_status.m_records;
    p.m_origins.pop_front();
    return true;
}

bool MultirateShadow::applied( const OnlineAppliedSlot &u, std::int64_t now )
{
    auto &p = *m_impl;
    advance( now );
    if( !p.m_status.m_qualified )
        return false;
    if( p.m_contract.m_upstream.m_enabled && p.m_contract.m_upstream.m_combinedRequestModel )
        return false; // Request-domain identification never consumes or manufactures physical-input certificates.
    const auto &c  = p.m_contract.m_inputs;
    const auto *dm = p.find( 3, u.m_dmCounter );
    bool        valid =
        p.stamp( u.m_stamp, c.m_commandGeneration, now ) && u.m_evidence == p.m_allowed && dm &&
        p.m_contract.m_sources[3].matches( *dm ) && dm->m_availableNs >= 0 && dm->m_receivedNs >= 0 &&
        dm->m_availableNs <= u.m_stamp.m_receivedNs && dm->m_receivedNs <= u.m_stamp.m_receivedNs && u.m_startNs >= 0 &&
        u.m_endNs > u.m_startNs && u.m_endNs - u.m_startNs == slot && u.m_stamp.m_availableNs >= u.m_endNs &&
        ( !p.m_lastCertificate || ( u.m_stamp.m_counter > p.m_lastCertificate && u.m_startNs >= p.m_lastInputEnd ) );
    for( int m = 0; m < p.m_modelConfig.m_modes; ++m )
        valid = valid && std::isfinite( u.m_values[m] );
    if( !valid || p.m_inputs.size() == 4 * p.m_config.m_pendingCapacity )
    {
        ++p.m_status.m_rejections;
        p.invalidate( "applied_input_proof_sequence_or_coverage_invalid" );
        return false;
    }
    if( p.m_lastCertificate && ( u.m_stamp.m_counter != p.m_lastCertificate + 1 || u.m_startNs != p.m_lastInputEnd ) )
        p.invalidate( "applied_slot_gap_new_history_segment" );
    p.m_lastCertificate = u.m_stamp.m_counter;
    p.m_lastInputEnd    = u.m_endNs;
    p.m_inputs.push_back( u );
    p.complete();
    return true;
}

void MultirateShadow::advance( std::int64_t now )
{
    auto &p = *m_impl;
    if( now < 0 || ( p.m_now >= 0 && now < p.m_now ) )
    {
        p.invalidate( "clock_reversal_new_contract_required" );
        p.m_status.m_qualified = false;
        p.m_status.m_clock     = false;
        return;
    }
    p.m_now = now;
    if( !p.m_status.m_qualified )
        return;
    if( p.m_status.m_candidateValid && now > p.m_status.m_deadlineNs )
    {
        p.m_planValid.fill( false );
        p.m_status.m_candidateValid = false;
        p.m_status.m_reason         = "input_ready_cutoff_expired";
    }
    if( p.m_lastOriginNs >= 0 && now - p.m_lastOriginNs > p.m_contract.m_inputs.m_maxAgeNs )
        p.invalidate( "common_input_silence" );
    if( ( !p.m_origins.empty() && now - ( p.m_contract.m_upstream.m_enabled ? p.m_origins.front().m_source.m_decisionNs
                                                                            : p.m_origins.front().m_source.m_gridNs ) >
                                      p.m_contract.m_inputs.m_maxAgeNs ) ||
        ( !p.m_inputs.empty() && now - p.m_inputs.front().m_startNs > p.m_contract.m_inputs.m_maxAgeNs ) )
        p.invalidate( "pending_input_evidence_stale" );
}

void MultirateShadow::finish_origin( std::int64_t now )
{
    auto &p = *m_impl;
    if( p.m_contract.m_upstream.m_enabled )
    {
        if( p.m_origins.empty() || p.m_origins.back().m_source.m_predictionDeadlineNs < 0 ||
            now > p.m_origins.back().m_source.m_predictionDeadlineNs )
        {
            p.m_models.mark_next_forecasts_late();
            if( p.m_status.m_predictionValid && !p.m_status.m_predictionLate )
                ++p.m_status.m_lateForecasts;
            p.m_status.m_predictionLate = true;
        }
    }
    else if( now > p.m_status.m_deadlineNs && p.m_status.m_candidateValid )
        p.m_models.discard_next_forecasts();
    advance( now );
}

void MultirateShadow::invalidate( const char *reason )
{
    m_impl->invalidate( reason );
}
void MultirateShadow::request( ShadowMode mode )
{
    auto &p = *m_impl;
    if( ( mode != ShadowMode::Standard && mode != ShadowMode::AccelAssisted ) || mode == p.m_status.m_requested )
        return;
    p.m_status.m_requested      = mode;
    const bool assisted         = mode == ShadowMode::AccelAssisted && p.m_planValid[1];
    p.m_status.m_effective      = assisted ? ShadowMode::AccelAssisted : ShadowMode::Standard;
    p.m_status.m_candidateValid = assisted || p.m_planValid[0];
    if( p.m_contract.m_upstream.m_enabled && !p.m_status.m_candidateValid )
    {
        const bool selectedAssisted  = mode == ShadowMode::AccelAssisted && p.m_forecastValid[1];
        p.m_status.m_effective       = selectedAssisted ? ShadowMode::AccelAssisted : ShadowMode::Standard;
        p.m_status.m_predictionValid = selectedAssisted || p.m_forecastValid[0];
    }
}
OnlineShadowStatus MultirateShadow::status() const
{
    auto s       = m_impl->m_status;
    s.m_standard = m_impl->m_models.status( ShadowMode::Standard );
    s.m_assisted = m_impl->m_models.status( ShadowMode::AccelAssisted );
    return s;
}
bool MultirateShadow::candidate( ShadowMode mode, Matrix &plan ) const
{
    const auto i = mode == ShadowMode::Standard ? 0 : 1;
    if( !m_impl->m_status.m_candidateValid || !m_impl->m_planValid[i] )
        return false;
    plan = m_impl->m_plans[i];
    return true;
}
Matrix MultirateShadow::coefficients( ShadowMode mode )
{
    return m_impl->m_models.prediction_matrix( mode );
}
bool MultirateShadow::prepare_plan( ShadowMode mode, Matrix &plan )
{
    auto &p = *m_impl;
    if( !p.m_contract.m_upstream.m_enabled || !p.m_status.m_qualified || p.m_origins.empty() )
        return false;
    const auto &o = p.m_origins.back();
    if( p.m_lastComplete + 1 != o.m_source.m_stamp.m_counter || p.m_latestModelReceipt > o.m_source.m_decisionNs ||
        ( mode == ShadowMode::AccelAssisted && !o.m_adcValid ) )
        return false;
    Matrix y( p.m_modelConfig.m_modes, 1 ), empty;
    for( int m = 0; m < y.rows(); ++m )
        y( m ) = o.m_source.m_residual[m];
    plan.resize( p.m_modelConfig.m_future * p.m_modelConfig.m_modes *
                     ( mode == ShadowMode::Standard ? p.m_modelConfig.m_standardCommandSlots : 4 ),
                 1 );
    return p.m_models.predict_next(
        mode, o.m_source.m_stamp.m_counter, y, mode == ShadowMode::AccelAssisted ? o.m_acceleration : empty, plan );
}
} // namespace DDSPC
