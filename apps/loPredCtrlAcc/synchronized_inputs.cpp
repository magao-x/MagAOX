/** \file synchronized_inputs.cpp
 * \brief Read-only synchronized measurement/request adapters; no producer, clock, actuator or scheduling API.
 */
#include "synchronized_inputs.hpp"
#include <cmath>
#include <bit>
#include <limits>

namespace DDSPC
{
std::uint64_t command_payload_hash( std::span<const float> values )
{
    std::uint64_t hash = 14695981039346656037ULL;
    for( float value : values )
        hash = ( hash ^ std::bit_cast<std::uint32_t>( value ) ) * 1099511628211ULL;
    return hash;
}
bool synchronized_origin( const MultirateContract &c,
                          const UpstreamStatus    &s,
                          const NativeObservation &w,
                          std::int64_t             decision,
                          std::int64_t             deadline,
                          OnlineOrigin            &record )
{
    const auto &u = c.m_upstream;
    if( !u.m_enabled || !u.m_referenceIsModal || !u.m_epoch || c.m_inputs.m_modes < 1 || c.m_inputs.m_modes > 2 ||
        u.m_adcAnchor > std::numeric_limits<std::uint64_t>::max() - 3 || w.m_source != ObservationSource::Modal ||
        w.m_event != ObservationEvent::Sample || !c.m_sources[1].matches( w ) || w.m_counter < u.m_modalAnchor ||
        w.m_counter - u.m_modalAnchor > ( std::numeric_limits<std::uint64_t>::max() - u.m_adcAnchor - 3 ) / 4 ||
        w.m_receivedNs < 0 || w.m_availableNs < 0 || decision < w.m_receivedNs || decision < w.m_availableNs ||
        ( w.m_flags & ObservationInvalidPayload ) )
        return false;
    OnlineOrigin out;
    out.m_stamp = { w.m_generation,
                    w.m_counter,
                    c.m_inputs.m_clockEpoch,
                    c.m_inputs.m_layout,
                    w.m_availableNs,
                    w.m_receivedNs,
                    0,
                    true };
    for( int m = 0; m < c.m_inputs.m_modes; ++m )
    {
        if( m >= 2 || !std::isfinite( w.m_values[m] ) )
            return false;
        out.m_residual[m] = float( w.m_values[m] );
    }
    out.m_decisionNs           = decision;
    out.m_predictionDeadlineNs = deadline;
    out.m_exposureStartNs = out.m_exposureEndNs = -1;
    out.m_syncEpoch                             = s.m_epoch;
    out.m_firstAdcCounter      = u.m_adcAnchor ? u.m_adcAnchor + 4 * ( w.m_counter - u.m_modalAnchor ) : 0;
    out.m_synchronized         = s.m_synchronized;
    out.m_locked               = s.m_locked;
    out.m_syncStatusReceivedNs = s.m_receivedNs;
    record                     = out;
    return true; // The engine independently checks lock freshness, all four ADCs, histories and command evidence.
}

bool project_request( const MultirateContract &c,
                      const NativeObservation &s,
                      std::span<const float>   values,
                      const Matrix            &projection,
                      ProjectedRequest        &record )
{
    if( !c.m_upstream.m_enabled || !c.m_upstream.m_combinedRequestModel || !c.m_inputs.m_mappingVerified ||
        c.m_inputs.m_modes < 1 || c.m_inputs.m_modes > 2 || s.m_source != ObservationSource::Dm ||
        s.m_event != ObservationEvent::Sample || !c.m_sources[3].matches( s ) || !s.m_payloadHash || !s.m_counter ||
        ( s.m_flags & ObservationInvalidPayload ) || values.empty() || values.size() > 4096 ||
        s.m_elements != values.size() || projection.rows() != c.m_inputs.m_modes ||
        projection.cols() != int( values.size() ) || !projection.allFinite() ||
        command_payload_hash( values ) != s.m_payloadHash )
        return false;
    ProjectedRequest out;
    out.m_source = s;
    for( int m = 0; m < projection.rows(); ++m )
    {
        double value = 0;
        for( int k = 0; k < projection.cols(); ++k )
        {
            if( !std::isfinite( values[k] ) )
                return false;
            value += double( projection( m, k ) ) * values[k];
        }
        if( !std::isfinite( value ) || !std::isfinite( float( value ) ) )
            return false;
        out.m_values[m] = float( value );
    }
    record = out;
    return true;
}

bool synchronized_commands( const MultirateContract           &c,
                            std::uint64_t                      wfs,
                            std::uint64_t                      sequence,
                            std::span<const ProjectedRequest>  history,
                            const std::array<std::int64_t, 4> &cutoffs,
                            const NativeObservation           &closed,
                            OnlineCommandFrame                &record )
{
    if( !c.m_upstream.m_enabled || !c.m_upstream.m_referenceIsModal || !c.m_upstream.m_combinedRequestModel ||
        !c.m_inputs.m_mappingVerified || c.m_inputs.m_modes < 1 || c.m_inputs.m_modes > 2 || !sequence || !wfs ||
        wfs == std::numeric_limits<std::uint64_t>::max() || closed.m_source != ObservationSource::Modal ||
        closed.m_counter != wfs + 1 || !c.m_sources[1].matches( closed ) || closed.m_availableNs < 0 ||
        closed.m_receivedNs < closed.m_availableNs || history.empty() || history.size() > 256 )
        return false;
    for( std::size_t n = 0; n < history.size(); ++n )
    {
        const auto &r = history[n].m_source;
        if( !c.m_sources[3].matches( r ) || r.m_source != ObservationSource::Dm || !r.m_payloadHash || !r.m_counter ||
            r.m_counter == std::numeric_limits<std::uint64_t>::max() || r.m_availableNs < 0 ||
            r.m_receivedNs < r.m_availableNs ||
            ( n && ( r.m_counter != history[n - 1].m_source.m_counter + 1 ||
                     r.m_availableNs < history[n - 1].m_source.m_availableNs ) ) )
            return false;
    }
    OnlineCommandFrame out;
    out.m_stamp      = { c.m_inputs.m_commandGeneration,
                         sequence,
                         c.m_inputs.m_clockEpoch,
                         c.m_inputs.m_layout,
                         closed.m_availableNs,
                         closed.m_receivedNs,
                         0,
                         true };
    out.m_wfsCounter = wfs;
    out.m_evidence   = c.m_inputs.m_evidence;
    out.m_sampledNs  = cutoffs;
    for( unsigned s = 0; s < 4; ++s )
    {
        if( cutoffs[s] < 0 || cutoffs[s] > closed.m_availableNs || ( s && cutoffs[s] <= cutoffs[s - 1] ) )
            return false;
        const ProjectedRequest *latest = nullptr;
        for( const auto &r : history )
            if( r.m_source.m_availableNs <= cutoffs[s] && r.m_source.m_receivedNs <= cutoffs[s] )
                latest = &r;
        if( !latest )
            return false;
        out.m_counters[s] = latest->m_source.m_counter;
        out.m_hashes[s]   = latest->m_source.m_payloadHash;
        for( int m = 0; m < c.m_inputs.m_modes; ++m )
        {
            if( m >= 2 || !std::isfinite( latest->m_values[m] ) )
                return false;
            out.m_values[s * c.m_inputs.m_modes + m] = latest->m_values[m];
        }
    }
    out.m_complete = true;
    record         = out;
    return true;
}
} // namespace DDSPC
