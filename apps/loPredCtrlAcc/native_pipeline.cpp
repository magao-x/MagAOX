/** \file native_pipeline.cpp
 * \brief Live observation-to-native-DDSPC adapter with explicit local request sampling and no output operations.
 */
#include "native_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace DDSPC
{
namespace synco = MagAOX::app::synco;
NativeCalibration load_native_calibration( const std::string &path )
{
    NativeCalibration c;
    if( path.empty() )
        return c;
    std::ifstream in( path );
    std::string   version;
    unsigned      modes = 0, pixels = 0, shapes = 0;
    if( !( in >> version >> c.m_layout >> modes >> pixels >> c.m_indices[0] >> c.m_indices[1] >> shapes >>
           c.m_modalUnits >> c.m_commandUnits ) ||
        version != "DDSPC_MODAL_V1" || !c.m_layout || modes < 1 || modes > 2 || pixels < 1 || pixels > 4096 ||
        shapes > 1 || c.m_modalUnits.size() > 64 || c.m_commandUnits.size() > 64 )
        throw std::invalid_argument( "Invalid native DDSPC modal calibration header" );
    c.m_projection.resize( modes, pixels );
    for( unsigned row = 0; row < modes; ++row )
        for( unsigned col = 0; col < pixels; ++col )
            if( !( in >> c.m_projection( row, col ) ) || !std::isfinite( c.m_projection( row, col ) ) )
                throw std::invalid_argument( "Invalid DM-to-modal projection" );
    if( shapes )
    {
        c.m_shapes.resize( pixels, modes );
        for( unsigned row = 0; row < pixels; ++row )
            for( unsigned col = 0; col < modes; ++col )
                if( !( in >> c.m_shapes( row, col ) ) || !std::isfinite( c.m_shapes( row, col ) ) )
                    throw std::invalid_argument( "Invalid modal-to-auxiliary shapes" );
        if( ( c.m_projection * c.m_shapes - Matrix::Identity( modes, modes ) ).cwiseAbs().maxCoeff() > .001f )
            throw std::invalid_argument( "Auxiliary shapes are inconsistent with modal calibration" );
    }
    std::string extra;
    if( in >> extra )
        throw std::invalid_argument( "Unexpected native calibration data" );
    return c;
}

NativeInputPipeline::NativeInputPipeline( MultirateShadow            &engine,
                                          const NativePipelineConfig &config,
                                          const ShadowConfig         &model )
    : m_engine( engine ), m_config( config ), m_model( model )
{
    if( !config.m_enabled || config.m_statusMaxAgeNs < 1000000 || config.m_statusMaxAgeNs > 2000000000 ||
        config.m_sampleToleranceNs < 0 || config.m_sampleToleranceNs >= 125000 || config.m_predictionLeadNs < 0 ||
        config.m_predictionLeadNs > 100000000 || model.m_modes < 1 || model.m_modes > 2 ||
        config.m_syncoCycleOffset < -1000000000 || config.m_syncoCycleOffset > 1000000000 ||
        ( config.m_adcAnchor && ( !config.m_modalAnchor || !config.m_syncEpoch || config.m_statusDevice.empty() ||
                                  config.m_reference.empty() ) ) )
        throw std::invalid_argument( "Invalid native receiver/sampler contract" );
    const auto &cal = config.m_calibration;
    if( cal.m_shapes.size() )
    {
        if( cal.m_shapes.cols() != model.m_modes || cal.m_shapes.rows() != cal.m_projection.cols() )
            throw std::invalid_argument( "Auxiliary calibration extent" );
        m_producer = std::make_shared<SyncoShadowProducer>( cal.m_shapes.rows(), model.m_future );
    }
}

bool NativeInputPipeline::bind( const NativeObservation &modal )
{
    const auto &cal = m_config.m_calibration;
    const auto &dm  = m_latest[3];
    if( !cal.m_layout || cal.m_projection.rows() != m_model.m_modes || !cal.m_projection.allFinite() ||
        !dm.m_generation || dm.m_elements != std::uint64_t( cal.m_projection.cols() ) )
    {
        m_reason = "native_modal_calibration_or_complete_dm_snapshot_missing";
        return false;
    }
    MultirateContract c;
    auto             &i = c.m_inputs;
    i.m_modes           = m_model.m_modes;
    i.m_periodNs        = 500000;
    i.m_clockEpoch =
        std::max<std::uint64_t>( 1, m_config.m_syncEpoch ); // Observer session only; not remote clock mapping.
    i.m_layout            = cal.m_layout;
    i.m_wfsGeneration     = modal.m_generation;
    i.m_commandGeneration = dm.m_generation;
    i.m_maxAgeNs          = 100000000;
    i.m_evidence          = InputEvidence::Requested;
    i.m_timingVerified = i.m_mappingVerified = true; // Explicit local sampler and the reviewed supplied calibration.
    c.m_upstream.m_enabled = c.m_upstream.m_combinedRequestModel = true;
    c.m_upstream.m_epoch                                         = i.m_clockEpoch;
    c.m_upstream.m_modalAnchor    = m_config.m_modalAnchor ? m_config.m_modalAnchor : modal.m_counter;
    c.m_upstream.m_adcAnchor      = m_config.m_adcAnchor;
    c.m_upstream.m_statusMaxAgeNs = m_config.m_statusMaxAgeNs;
    for( unsigned n = 0; n < 4; ++n )
    {
        const auto &r  = n == 1 ? modal : m_latest[n];
        c.m_sources[n] = { r.m_device, r.m_fileInode, r.m_imageInode, r.m_generation };
    }
    if( !c.m_sources[0].m_generation )
        c.m_upstream.m_adcAnchor = 0; // Standard acquisition needs no ADC; later assistance requires a fresh binding.
    c.m_provenance = "Native complete observed requests, local observer sampling; calibrated " + cal.m_modalUnits +
                     "/" + cal.m_commandUnits + "; upstream reference=" + m_config.m_reference +
                     "; no physical-application or exposure assertion";
    if( !m_engine.qualify( c ) )
    {
        m_reason = "native_source_binding_incomplete";
        return false;
    }
    m_contract = std::move( c );
    m_bound    = true;
    m_engine.observe( modal ); // Qualification starts a new history; bind the actual current snapshot again.
    return true;
}

void NativeInputPipeline::command( const NativeObservation &r, std::span<const float> values )
{
    // Projection is independent of whether a WFS frame has arrived yet.
    MultirateContract c    = m_contract;
    c.m_upstream.m_enabled = c.m_upstream.m_combinedRequestModel = true;
    c.m_inputs.m_mappingVerified                                 = true;
    c.m_inputs.m_modes                                           = m_model.m_modes;
    c.m_sources[3] = { r.m_device, r.m_fileInode, r.m_imageInode, r.m_generation };
    ProjectedRequest p;
    if( !project_request( c, r, values, m_config.m_calibration.m_projection, p ) )
    {
        m_reason = "complete_request_projection_failed";
        invalidate();
        return;
    }
    if( !m_requests.empty() )
    {
        const auto &old = m_requests.back().m_source;
        if( old.m_counter == r.m_counter && old.m_generation == r.m_generation )
            return;
        if( !c.m_sources[3].matches( old ) || r.m_counter != old.m_counter + 1 )
        {
            invalidate();
            m_requests.clear();
        }
    }
    if( m_requests.size() == 256 )
    {
        invalidate();
        m_requests.clear();
        m_reason = "native_projected_request_history_overflow";
    }
    m_requests.push_back( p );
    m_latest[3] = r;
}

void NativeInputPipeline::tick( std::int64_t now )
{
    if( m_pendingWfs && now - m_openedNs > 1000000 )
    {
        invalidate();
        m_reason = "native_wfs_observation_cadence_lost";
        return;
    }
    if( !m_pendingWfs || m_samples == 4 || m_samplerLost )
        return;
    const auto target = m_openedNs + 125000 * m_samples;
    if( now < target )
        return;
    if( now - target > m_config.m_sampleToleranceNs )
    {
        m_samplerLost = true;
        m_reason      = "observer_request_sampling_opportunity_missed";
        return;
    }
    m_sampled[m_samples++] = now; // Actual observation time, not the nominal target or a physical DM timestamp.
}

void NativeInputPipeline::modal( const NativeObservation &r, std::int64_t now )
{
    const auto processingStarted = std::chrono::steady_clock::now();
    if( !m_bound && !bind( r ) )
        return;
    if( m_pendingWfs )
    {
        std::array<ProjectedRequest, 256> rows;
        std::copy( m_requests.begin(), m_requests.end(), rows.begin() );
        OnlineCommandFrame u;
        if( m_samples == 4 && !m_samplerLost &&
            synchronized_commands( m_contract,
                                   m_pendingWfs,
                                   ++m_sequence,
                                   std::span<const ProjectedRequest>( rows.data(), m_requests.size() ),
                                   m_sampled,
                                   r,
                                   u ) )
            m_engine.commands( u, now );
        else
            m_engine.invalidate( "native_request_sampling_or_history_coverage_incomplete" );
        // Keep the actual last state plus later updates; bounded retention eviction is not source loss.
        while( m_requests.size() > 1 && m_requests[1].m_source.m_availableNs <= m_sampled[3] )
            m_requests.pop_front();
    }
    UpstreamStatus status;
    for( const auto &s : m_syncHistory )
        if( s.m_status.m_receivedNs <= r.m_availableNs )
            status = s.m_status;
    status.m_locked = status.m_locked && !m_syncLost;
    OnlineOrigin o;
    if( !synchronized_origin(
            m_contract, status, r, r.m_availableNs, r.m_availableNs + m_config.m_predictionLeadNs, o ) ||
        !m_engine.origin( o, now ) )
    {
        m_reason     = "native_wfs_origin_or_availability_invalid";
        m_pendingWfs = 0;
        return;
    }
    m_reason = !m_contract.m_upstream.m_adcAnchor                ? "standard_only_adc_counter_contract_missing"
               : !m_contract.m_sources[0].matches( m_latest[0] ) ? "adc_source_changed_new_counter_binding_required"
               : m_syncLost                                      ? "upstream_restart_or_gap_new_anchors_required"
               : !status.m_synchronized || !status.m_locked ? "upstream_status_unlocked_or_unavailable_standard_only"
                                                            : "observed_request_domain_not_physical_application";
    m_pendingWfs  = r.m_counter;
    m_openedNs    = r.m_availableNs;
    m_samples     = 0;
    m_samplerLost = false;
    if( m_producer && m_config.m_syncoAssociation && m_scheduleReceipt >= 0 && m_scheduleReceipt <= now &&
        synco::validProductionRecord( m_schedule, now ) && m_schedule[synco::steps] == 4 &&
        m_schedule[synco::period] == 500000 && m_schedule[synco::cycle] >= 0 &&
        m_schedule[synco::cycle] < std::numeric_limits<std::int64_t>::max() &&
        o.m_stamp.m_counter <= std::uint64_t( std::numeric_limits<std::int64_t>::max() ) &&
        o.m_stamp.m_counter < std::uint64_t( std::numeric_limits<std::int64_t>::max() - 1000000001 ) &&
        std::int64_t( o.m_stamp.m_counter ) + m_config.m_syncoCycleOffset == m_schedule[synco::cycle] + 1 )
    {
        for( unsigned mode = 0; mode < 2; ++mode )
        {
            Matrix plan;
            if( !m_engine.prepare_plan( mode ? ShadowMode::AccelAssisted : ShadowMode::Standard, plan ) )
                continue;
            if( !mode && m_model.m_standardCommandSlots == 1 )
            {
                Matrix held( 4 * plan.rows(), 1 );
                for( int f = 0; f < m_model.m_future; ++f )
                    for( int phase = 0; phase < 4; ++phase )
                        held.middleRows( ( 4 * f + phase ) * m_model.m_modes, m_model.m_modes ) =
                            plan.middleRows( f * m_model.m_modes, m_model.m_modes );
                plan.swap( held ); // Preserve the single-rate baseline as a held four-phase proposal, never average.
            }
            const auto completed = now + std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - processingStarted )
                                             .count();
            m_producer->prepare( mode ? ShadowMode::AccelAssisted : ShadowMode::Standard,
                                 plan,
                                 m_config.m_calibration.m_shapes,
                                 m_schedule,
                                 m_schedule[synco::targets],
                                 completed,
                                 o.m_decisionNs );
        }
    }
    // The owner reports actual completion through finish_origin(); offline callers supply their virtual clock.
}

void NativeInputPipeline::observe( const NativeObservation &r, std::int64_t now )
{
    const auto n = std::size_t( r.m_source );
    if( n >= 4 )
        return;
    if( m_bound && n != 0 && n != 2 && !m_contract.m_sources[n].matches( r ) )
    {
        invalidate();
        m_bound      = false;
        m_pendingWfs = 0;
    }
    m_engine.observe( r );
    if( r.m_event != ObservationEvent::Sample || ( r.m_flags & ObservationInvalidPayload ) )
    {
        if( n == 0 )
        {
            if( m_producer )
                m_producer->inhibit( true );
        }
        else if( n != 2 )
        {
            invalidate();
            m_bound     = false;
            m_latest[n] = {};
            if( n == 3 )
                m_requests.clear(); // An invalid command source cannot masquerade as a valid held request.
        }
        return;
    }
    m_latest[n] = r;
    if( n == 1 )
        modal( r, now );
}

void NativeInputPipeline::sync( const NativeSyncStatus &s )
{
    if( s.m_status.m_receivedNs < m_sync.m_status.m_receivedNs )
        return;
    if( m_sync.m_status.m_receivedNs >= 0 &&
        ( s.m_sequence < m_sync.m_sequence || s.m_referenceCounter < m_sync.m_referenceCounter ||
          s.m_gaps != m_sync.m_gaps ) )
        m_syncLost = true; // New counter anchors must be reviewed after producer restarts/gaps.
    m_sync = s;
    if( m_syncHistory.size() == 8 )
        m_syncHistory.pop_front();
    m_syncHistory.push_back( m_sync );
    if( !s.m_status.m_locked || !s.m_status.m_synchronized || m_syncLost )
        if( m_producer )
            m_producer->inhibit( true );
}

void NativeInputPipeline::schedule( const synco::Record &s, std::int64_t received, std::int64_t now )
{
    if( s[synco::epoch] != m_schedule[synco::epoch] || s[synco::generation] != m_schedule[synco::generation] ||
        !synco::validProductionRecord( s, now ) )
        if( m_producer )
            m_producer->inhibit( false );
    m_schedule        = s;
    m_scheduleReceipt = received;
}

void NativeInputPipeline::invalidate()
{
    m_engine.invalidate( "native_input_bridge_discontinuity" );
    m_pendingWfs = 0;
    if( m_producer )
        m_producer->inhibit( false );
}
void NativeInputPipeline::status( OnlineShadowStatus &s ) const
{
    s.m_adapterReason = m_reason;
    if( m_producer )
    {
        m_producer->models( s.m_requested, s.m_standard.m_ready, s.m_assisted.m_ready && s.m_syncValid );
        m_producer->status( s );
    }
    if( !s.m_qualified )
        s.m_reason = m_reason;
}
std::shared_ptr<SyncoShadowProducer> NativeInputPipeline::producer() const
{
    return m_producer;
}
} // namespace DDSPC
