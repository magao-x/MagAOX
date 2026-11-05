/** \file causal_input.cpp
 * \brief Conservative exposure/held-input pairing without guessed application or exposure times.
 */
#include "causal_input.hpp"
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <utility>

namespace DDSPC
{

const char *input_evidence( InputEvidence evidence )
{
    switch( evidence )
    {
    case InputEvidence::Unverified:
        return "unverified";
    case InputEvidence::Requested:
        return "requested";
    case InputEvidence::DriverPrepared:
        return "driver_prepared";
    case InputEvidence::DriverAccepted:
        return "driver_accepted";
    case InputEvidence::VerifiedApplied:
        return "externally_verified_applied";
    case InputEvidence::SyntheticApplied:
        return "synthetic_applied";
    }
    return "unknown_input_provenance";
}

const char *input_reason( InputReason reason )
{
    switch( reason )
    {
    case InputReason::UnverifiedContract:
        return "unverified_input_contract";
    case InputReason::AwaitingRecords:
        return "awaiting_exposure_and_applied_interval";
    case InputReason::Ready:
        return "paired_input_records";
    case InputReason::Reset:
        return "input_history_reset";
    case InputReason::InvalidPayload:
        return "invalid_payload_or_snapshot";
    case InputReason::InvalidTime:
        return "invalid_timestamp_or_uncertainty";
    case InputReason::WrongIdentity:
        return "generation_clock_or_layout_mismatch";
    case InputReason::UnverifiedCommand:
        return "command_application_unverified";
    case InputReason::OutOfOrder:
        return "duplicate_or_out_of_order_record";
    case InputReason::CounterGap:
        return "source_counter_gap";
    case InputReason::TimingGap:
        return "source_timing_gap";
    case InputReason::MissingCoverage:
        return "missing_applied_interval_coverage";
    case InputReason::AmbiguousExposure:
        return "command_transition_or_timing_ambiguity";
    case InputReason::UnsupportedLatency:
        return "latency_requires_committed_input_prefix";
    case InputReason::Stale:
        return "stale_input_evidence";
    case InputReason::Overflow:
        return "pending_input_overflow";
    case InputReason::ClockReversal:
        return "clock_reversal_new_session_required";
    }
    return "unknown_input_reason";
}

CausalInputBuilder::CausalInputBuilder( const InputConfig &config ) : m_config( config )
{
    if( config.m_modes <= 0 || config.m_accelChannels < 0 || config.m_periodNs <= 0 || config.m_exposureNs <= 0 ||
        config.m_exposureNs > config.m_periodNs || config.m_jitterNs < 0 ||
        config.m_jitterNs > ( config.m_periodNs - 1 ) / 2 || config.m_maxAgeNs <= 0 || config.m_maxUncertaintyNs < 0 ||
        config.m_maxUncertaintyNs > std::numeric_limits<std::int64_t>::max() / 2 || config.m_responseLagNs < 0 ||
        config.m_minCommandDelayNs < 0 || config.m_capacity < 2 )
        throw std::invalid_argument( "Invalid causal input dimensions, timing, or queue capacity" );
    if( config.m_timingVerified && config.m_mappingVerified && config.m_wfsGeneration && config.m_commandGeneration &&
        config.m_clockEpoch && config.m_layout &&
        ( config.m_evidence == InputEvidence::VerifiedApplied ||
          config.m_evidence == InputEvidence::SyntheticApplied ) )
        m_reason = InputReason::AwaitingRecords;
}

void CausalInputBuilder::reject( InputBuildResult &result, InputReason reason )
{
    m_wfs.clear();
    m_commands.clear();
    result.m_records.clear();
    result.m_accepted      = false;
    result.m_discontinuity = true;
    result.m_reason        = reason;
}

bool CausalInputBuilder::prepare( InputBuildResult &result, std::int64_t nowNs )
{
    if( !m_config.m_timingVerified || !m_config.m_mappingVerified || !m_config.m_wfsGeneration ||
        !m_config.m_commandGeneration || !m_config.m_clockEpoch || !m_config.m_layout ||
        ( m_config.m_evidence != InputEvidence::VerifiedApplied &&
          m_config.m_evidence != InputEvidence::SyntheticApplied ) )
    {
        reject( result, InputReason::UnverifiedContract );
        return false;
    }
    if( nowNs < 0 || m_clockFault || ( m_haveNow && nowNs < m_lastNowNs ) )
    {
        m_clockFault = true;
        reject( result, InputReason::ClockReversal );
        return false;
    }
    m_lastNowNs = nowNs;
    m_haveNow   = true;
    if( ( m_lastWfs.m_have && nowNs - m_lastWfs.m_endNs > m_config.m_maxAgeNs ) ||
        ( m_lastCommand.m_have && nowNs - m_lastCommand.m_endNs > m_config.m_maxAgeNs ) ||
        ( !m_wfs.empty() && nowNs - m_wfs.front().m_exposureEndNs > m_config.m_maxAgeNs ) )
        reject( result, InputReason::Stale ); // A fresh event can establish a new segment after this break.
    return true;
}

bool CausalInputBuilder::valid_stamp( const InputStamp &stamp, std::uint64_t generation, std::int64_t nowNs ) const
{
    return stamp.m_generation == generation && stamp.m_clockEpoch == m_config.m_clockEpoch &&
           stamp.m_layout == m_config.m_layout && stamp.m_availableNs >= 0 &&
           stamp.m_receivedNs >= stamp.m_availableNs && nowNs >= stamp.m_receivedNs && stamp.m_uncertaintyNs >= 0 &&
           stamp.m_uncertaintyNs <= m_config.m_maxUncertaintyNs;
}

InputBuildResult CausalInputBuilder::finish( InputBuildResult result )
{
    m_reason = result.m_reason;
    return result;
}

InputBuildResult CausalInputBuilder::push_wfs( const TimedWfsRecord &record, std::int64_t nowNs )
{
    InputBuildResult result;
    if( !prepare( result, nowNs ) )
        return finish( std::move( result ) );
    const auto &stamp = record.m_stamp;
    if( stamp.m_generation != m_config.m_wfsGeneration || stamp.m_clockEpoch != m_config.m_clockEpoch ||
        stamp.m_layout != m_config.m_layout )
        reject( result, InputReason::WrongIdentity );
    else if( !stamp.m_snapshotValid || record.m_residual.rows() != m_config.m_modes || record.m_residual.cols() != 1 ||
             !record.m_residual.allFinite() || stamp.m_counter == std::numeric_limits<std::uint64_t>::max() )
        reject( result, InputReason::InvalidPayload );
    else if( !valid_stamp( stamp, m_config.m_wfsGeneration, nowNs ) ||
             record.m_exposureStartNs < stamp.m_uncertaintyNs || record.m_exposureEndNs <= record.m_exposureStartNs ||
             record.m_exposureEndNs > std::numeric_limits<std::int64_t>::max() - stamp.m_uncertaintyNs ||
             stamp.m_availableNs < record.m_exposureEndNs + stamp.m_uncertaintyNs ||
             record.m_decisionNs < stamp.m_receivedNs || record.m_decisionNs > nowNs ||
             std::abs( ( record.m_exposureEndNs - record.m_exposureStartNs ) - m_config.m_exposureNs ) >
                 m_config.m_jitterNs )
        reject( result, InputReason::InvalidTime );
    else if( nowNs - ( record.m_exposureEndNs - stamp.m_uncertaintyNs ) > m_config.m_maxAgeNs )
        reject( result, InputReason::Stale );
    else if( m_lastWfs.m_have &&
             ( stamp.m_counter <= m_lastWfs.m_stamp.m_counter || record.m_exposureStartNs <= m_lastWfs.m_startNs ||
               record.m_exposureStartNs < m_lastWfs.m_endNs || record.m_decisionNs <= m_lastWfs.m_decisionNs ||
               stamp.m_availableNs < m_lastWfs.m_stamp.m_availableNs ||
               stamp.m_receivedNs < m_lastWfs.m_stamp.m_receivedNs ) )
        reject( result, InputReason::OutOfOrder );
    else
    {
        if( m_lastWfs.m_have && stamp.m_counter != m_lastWfs.m_stamp.m_counter + 1 )
            reject( result, InputReason::CounterGap );
        else if( m_lastWfs.m_have && std::abs( ( record.m_exposureStartNs - m_lastWfs.m_startNs ) -
                                               m_config.m_periodNs ) > m_config.m_jitterNs )
            reject( result, InputReason::TimingGap );
        if( m_wfs.size() >= m_config.m_capacity )
            reject( result, InputReason::Overflow );
        else
        {
            m_wfs.push_back( record );
            m_lastWfs         = { true, stamp, record.m_exposureStartNs, record.m_exposureEndNs, record.m_decisionNs };
            result.m_accepted = true;
            drain( result, nowNs );
        }
    }
    return finish( std::move( result ) );
}

InputBuildResult CausalInputBuilder::push_command( const AppliedInputRecord &record, std::int64_t nowNs )
{
    InputBuildResult result;
    if( !prepare( result, nowNs ) )
        return finish( std::move( result ) );
    const auto &stamp = record.m_stamp;
    if( record.m_evidence != m_config.m_evidence )
        reject( result, InputReason::UnverifiedCommand );
    else if( stamp.m_generation != m_config.m_commandGeneration || stamp.m_clockEpoch != m_config.m_clockEpoch ||
             stamp.m_layout != m_config.m_layout )
        reject( result, InputReason::WrongIdentity );
    else if( !stamp.m_snapshotValid || record.m_applied.rows() != m_config.m_modes || record.m_applied.cols() != 1 ||
             !record.m_applied.allFinite() || stamp.m_counter == std::numeric_limits<std::uint64_t>::max() )
        reject( result, InputReason::InvalidPayload );
    else if( !valid_stamp( stamp, m_config.m_commandGeneration, nowNs ) || record.m_requestedNs < 0 ||
             record.m_effectiveStartNs < record.m_requestedNs || record.m_effectiveEndNs <= record.m_effectiveStartNs ||
             record.m_effectiveEndNs > std::numeric_limits<std::int64_t>::max() - stamp.m_uncertaintyNs ||
             record.m_effectiveEndNs - record.m_effectiveStartNs <= 2 * stamp.m_uncertaintyNs ||
             stamp.m_availableNs < record.m_effectiveEndNs + stamp.m_uncertaintyNs )
        reject( result, InputReason::InvalidTime );
    else if( nowNs - record.m_effectiveEndNs > m_config.m_maxAgeNs )
        reject( result, InputReason::Stale );
    else if( m_lastCommand.m_have && ( stamp.m_counter <= m_lastCommand.m_stamp.m_counter ||
                                       record.m_effectiveStartNs < m_lastCommand.m_endNs ||
                                       stamp.m_availableNs < m_lastCommand.m_stamp.m_availableNs ||
                                       stamp.m_receivedNs < m_lastCommand.m_stamp.m_receivedNs ) )
        reject( result, InputReason::OutOfOrder );
    else
    {
        if( m_lastCommand.m_have && stamp.m_counter != m_lastCommand.m_stamp.m_counter + 1 )
            reject( result, InputReason::CounterGap );
        else if( m_lastCommand.m_have && record.m_effectiveStartNs > m_lastCommand.m_endNs )
            reject( result, InputReason::TimingGap );
        if( m_commands.size() >= m_config.m_capacity )
            reject( result, InputReason::Overflow );
        else
        {
            m_commands.push_back( record );
            m_lastCommand     = { true, stamp, record.m_effectiveStartNs, record.m_effectiveEndNs, 0 };
            result.m_accepted = true;
            drain( result, nowNs );
        }
    }
    return finish( std::move( result ) );
}

void CausalInputBuilder::drain( InputBuildResult &result, std::int64_t nowNs )
{
    while( m_wfs.size() >= 2 )
    {
        const auto &origin      = m_wfs[0];
        const auto &target      = m_wfs[1];
        const auto  uncertainty = target.m_stamp.m_uncertaintyNs;
        if( target.m_exposureStartNs - uncertainty < m_config.m_responseLagNs )
        {
            reject( result, InputReason::InvalidTime );
            return;
        }
        const auto start = target.m_exposureStartNs - uncertainty - m_config.m_responseLagNs;
        const auto end   = target.m_exposureEndNs + uncertainty - m_config.m_responseLagNs;
        if( start < origin.m_decisionNs || start - origin.m_decisionNs < m_config.m_minCommandDelayNs )
        {
            reject( result, InputReason::UnsupportedLatency );
            return;
        }
        while( !m_commands.empty() && m_commands.front().m_effectiveEndNs <= start )
            m_commands.pop_front();
        if( m_commands.empty() )
            return;
        const auto &command = m_commands.front();
        if( command.m_effectiveStartNs + command.m_stamp.m_uncertaintyNs > start )
        {
            reject( result,
                    command.m_effectiveStartNs > start ? InputReason::MissingCoverage
                                                       : InputReason::AmbiguousExposure );
            return;
        }
        if( command.m_effectiveEndNs - command.m_stamp.m_uncertaintyNs < end )
        {
            reject( result, InputReason::AmbiguousExposure );
            return;
        }
        if( nowNs - ( origin.m_exposureEndNs - origin.m_stamp.m_uncertaintyNs ) > m_config.m_maxAgeNs )
        {
            reject( result, InputReason::Stale );
            return;
        }
        PairedInputRecord paired;
        paired.m_record.m_index          = origin.m_stamp.m_counter;
        paired.m_record.m_residual       = origin.m_residual;
        paired.m_record.m_appliedCommand = command.m_applied;
        paired.m_record.m_wfsValid = paired.m_record.m_appliedValid = paired.m_record.m_timingValid = true;
        paired.m_record.m_accelValid =
            origin.m_accelValid && m_config.m_accelChannels > 0 &&
            origin.m_acceleration.rows() == m_config.m_accelChannels && origin.m_acceleration.cols() == 1 &&
            origin.m_acceleration.allFinite() && origin.m_accelAcquiredNs >= 0 &&
            origin.m_accelAvailableNs >= origin.m_accelAcquiredNs && origin.m_accelAvailableNs <= origin.m_decisionNs &&
            origin.m_decisionNs - origin.m_accelAcquiredNs <= m_config.m_maxAgeNs;
        if( paired.m_record.m_accelValid )
            paired.m_record.m_acceleration = origin.m_acceleration;
        paired.m_origin       = origin.m_stamp;
        paired.m_target       = target.m_stamp;
        paired.m_command      = command.m_stamp;
        paired.m_evidence     = command.m_evidence;
        paired.m_decisionNs   = origin.m_decisionNs;
        paired.m_inputStartNs = start;
        paired.m_inputEndNs   = end;
        result.m_records.push_back( std::move( paired ) );
        result.m_reason = InputReason::Ready;
        m_wfs.pop_front();
    }
}

InputBuildResult CausalInputBuilder::advance( std::int64_t nowNs )
{
    InputBuildResult result;
    if( prepare( result, nowNs ) )
    {
        result.m_accepted = !result.m_discontinuity;
        drain( result, nowNs );
    }
    return finish( std::move( result ) );
}

InputBuildResult CausalInputBuilder::invalidate()
{
    InputBuildResult result;
    reject( result, InputReason::Reset );
    return finish( std::move( result ) );
}

InputReason CausalInputBuilder::reason() const
{
    return m_reason;
}

InputEvidence CausalInputBuilder::evidence() const
{
    return m_config.m_evidence;
}

} // namespace DDSPC
