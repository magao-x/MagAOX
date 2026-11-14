/** \file shadow_pipeline.cpp
 * \brief Confirmed-input application replay orchestration; shadow recommendations never enter learning records.
 */
#include "shadow_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace DDSPC
{
void ShadowPipeline::request( PolicyMode mode )
{
    m_policy.request( mode );
    if( mode != m_status.m_requested )
    {
        m_status.m_candidateValid = m_decision.m_candidateValid = false;
        m_status.m_validStreak                                  = 0;
        m_status.m_reason                                       = "mode_change_pending";
    }
    m_status.m_requested = mode;
}

ShadowPipeline::ShadowPipeline( ShadowController    &controller,
                                const ShadowConfig  &models,
                                const InputConfig   &inputs,
                                const FeatureConfig &features,
                                const PolicyConfig  &policy,
                                PolicyMode           requested,
                                std::uint64_t        calibration,
                                std::size_t          capacity )
    : m_controller( controller ), m_models( models ), m_inputConfig( inputs ), m_featureConfig( features ),
      m_inputs( inputs ), m_policy( requested, policy ), m_calibration( calibration ),
      m_depth( std::max( models.m_history, models.m_accelHistory ) + 1 ), m_evidence( models.m_modes ),
      m_decision( models.m_modes )
{
    if( models.m_modes != inputs.m_modes || models.m_modes != policy.m_modes || models.m_accelChannels != 2 ||
        inputs.m_accelChannels != 2 || models.m_history < 1 || models.m_accelHistory < 1 || !calibration ||
        capacity < 2 || capacity > 65536 || m_inputs.reason() == InputReason::UnverifiedContract )
        throw std::invalid_argument( "Invalid or unverified shadow pipeline contract" );
    acceleration_window( {}, 2000000000LL, features, m_feature );
    m_adc.resize( capacity );
    m_scratch.resize( capacity );
    m_y.setZero( m_depth, models.m_modes );
    m_u.setZero( m_depth, models.m_modes );
    m_a.setZero( m_depth, 2 );
    m_aValid.resize( m_depth );
    const int standard = ( 2 * models.m_history - 1 ) * models.m_modes;
    m_standardState.setZero( standard, 1 );
    m_assistedState.setZero( standard + 2 * models.m_accelHistory, 1 );
    m_previous.setZero( models.m_modes, 1 );
    m_zero              = m_previous;
    m_origin.m_residual = m_previous;
    m_origin.m_acceleration.setZero( 2, 1 );
    m_status.m_requested = requested;
}

bool ShadowPipeline::admit_time( std::int64_t nowNs )
{
    if( m_clockFault || nowNs < 0 || ( m_lastNow >= 0 && nowNs < m_lastNow ) )
    {
        m_clockFault = true;
        invalidate();
        m_status.m_reason = "clock_reversal";
        return false;
    }
    m_lastNow = nowNs;
    return true;
}

void ShadowPipeline::break_adc()
{
    m_adcCount = m_adcNext = 0;
    std::fill( m_aValid.begin(), m_aValid.end(), 0 );
    m_policy.reset();
    m_status.m_candidateValid   = false;
    m_status.m_featureReason    = FeatureReason::Incomplete;
    m_status.m_validStreak      = 0;
    m_status.m_reason           = "adc_discontinuity";
    m_decision.m_candidateValid = false;
}

void ShadowPipeline::break_common()
{
    m_controller.invalidate( ShadowRecord{} );
    m_havePair = false;
    m_count = m_head = 0;
    std::fill( m_aValid.begin(), m_aValid.end(), 0 );
    m_policy.reset();
    m_status.m_fallbacks += m_status.m_effective == PolicyMode::Assisted;
    m_status.m_validStreak    = 0;
    m_status.m_candidateValid = m_status.m_commonValid = false;
    m_status.m_effective                               = PolicyMode::Safe;
    m_status.m_reason                                  = "common_input_invalid";
    m_decision.m_candidateValid                        = false;
}

bool ShadowPipeline::push_adc( const ShadowAdcRecord &record, std::int64_t nowNs )
{
    if( !admit_time( nowNs ) )
        return false;
    const auto &s = record.m_sample;
    if( !record.m_snapshotValid || record.m_clockEpoch != m_inputConfig.m_clockEpoch ||
        record.m_calibration != m_calibration || record.m_uncertaintyNs < 0 ||
        record.m_uncertaintyNs > m_inputConfig.m_maxUncertaintyNs || !s.m_generation || !s.m_segment ||
        !s.m_frame.m_counter || s.m_frame.m_counter == std::numeric_limits<std::uint64_t>::max() ||
        s.m_acquiredNs < 0 || s.m_publishedNs < s.m_acquiredNs || s.m_receivedNs < s.m_publishedNs ||
        nowNs < s.m_receivedNs || !std::isfinite( s.m_acceleration[0] ) || !std::isfinite( s.m_acceleration[1] ) )
    {
        break_adc();
        return false;
    }
    if( m_haveAdc && ( s.m_generation < m_lastAdc.m_generation ||
                       ( s.m_generation == m_lastAdc.m_generation && s.m_segment < m_lastAdc.m_segment ) ||
                       s.m_acquiredNs <= m_lastAdc.m_acquiredNs ) )
    {
        const bool duplicate = s.m_generation == m_lastAdc.m_generation && s.m_segment == m_lastAdc.m_segment &&
                               s.m_frame.m_counter == m_lastAdc.m_frame.m_counter &&
                               s.m_acquiredNs == m_lastAdc.m_acquiredNs;
        if( !duplicate )
            break_adc();
        return false;
    }
    const bool same = m_haveAdc && s.m_generation == m_lastAdc.m_generation && s.m_segment == m_lastAdc.m_segment;
    if( same && s.m_frame.m_counter == m_lastAdc.m_frame.m_counter && s.m_acquiredNs == m_lastAdc.m_acquiredNs )
        return false; // Duplicate observation does not become another sample or break a healthy history.
    if( m_haveAdc &&
        ( !same || s.m_frame.m_counter != m_lastAdc.m_frame.m_counter + 1 || s.m_acquiredNs <= m_lastAdc.m_acquiredNs ||
          s.m_acquiredNs - m_lastAdc.m_acquiredNs > m_featureConfig.m_maxGapNs ) )
    {
        break_adc();
        if( same && ( s.m_frame.m_counter <= m_lastAdc.m_frame.m_counter || s.m_acquiredNs <= m_lastAdc.m_acquiredNs ) )
            return false;
    }
    m_adc[m_adcNext]              = s;
    m_adc[m_adcNext].m_receivedNs = nowNs; // Actual frontend availability, not a retrospective worker receipt.
    m_adcNext                     = ( m_adcNext + 1 ) % m_adc.size();
    m_adcCount                    = std::min( m_adcCount + 1, m_adc.size() );
    m_lastAdc                     = s;
    m_haveAdc                     = true;
    m_status.m_adcGeneration      = s.m_generation;
    m_status.m_adcSegment         = s.m_segment;
    return true;
}

bool ShadowPipeline::consume( InputBuildResult result )
{
    m_status.m_inputReason = result.m_reason;
    if( result.m_discontinuity )
        break_common();
    bool accepted = result.m_accepted;
    for( auto &paired : result.m_records )
    {
        auto &r = paired.m_record;
        // Delayed confirmation cannot revive features from an earlier ADC source incarnation.
        r.m_accelValid = r.m_accelValid && m_haveAdc && paired.m_accelGeneration == m_lastAdc.m_generation &&
                         paired.m_accelSegment == m_lastAdc.m_segment;
        if( m_havePair && r.m_index != m_lastPair + 1 )
            break_common();
        accepted          = m_controller.process( r ) && accepted;
        m_y.row( m_head ) = r.m_residual.transpose();
        m_u.row( m_head ) = r.m_appliedCommand.transpose();
        if( r.m_accelValid )
            m_a.row( m_head ) = r.m_acceleration.transpose();
        else
            m_a.row( m_head ).setZero();
        m_aValid[m_head] = r.m_accelValid;
        m_head           = ( m_head + 1 ) % m_depth;
        m_count          = std::min( m_count + 1, m_depth );
        m_lastPair       = r.m_index;
        m_havePair       = true;
    }
    return accepted;
}

void ShadowPipeline::state( Matrix &output, const Matrix &current, bool assisted )
{
    int row                                    = 0;
    output.middleRows( row, m_models.m_modes ) = current;
    row += m_models.m_modes;
    for( int i = 0; i < m_models.m_history - 1; ++i, row += m_models.m_modes )
        output.middleRows( row, m_models.m_modes ) = m_y.row( ( m_head + m_depth - 1 - i ) % m_depth ).transpose();
    for( int i = 0; i < m_models.m_history - 1; ++i, row += m_models.m_modes )
        output.middleRows( row, m_models.m_modes ) = m_u.row( ( m_head + m_depth - 1 - i ) % m_depth ).transpose();
    if( assisted )
    {
        output.middleRows( row, 2 ) = m_feature.m_value;
        row += 2;
        for( int i = 0; i < m_models.m_accelHistory - 1; ++i, row += 2 )
            output.middleRows( row, 2 ) = m_a.row( ( m_head + m_depth - 1 - i ) % m_depth ).transpose();
    }
}

bool ShadowPipeline::push_wfs( const TimedWfsRecord &record, std::int64_t nowNs )
{
    if( !admit_time( nowNs ) )
        return false;
    if( nowNs < m_featureConfig.m_windowNs + m_featureConfig.m_lagNs )
    {
        invalidate();
        m_status.m_reason = "invalid_origin_time";
        return false;
    }
    if( nowNs != record.m_decisionNs )
    {
        invalidate();
        m_status.m_reason = "missed_origin_deadline";
        return false;
    }
    for( std::size_t i = 0; i < m_adcCount; ++i )
        m_scratch[i] = m_adc[( m_adcNext + m_adc.size() - m_adcCount + i ) % m_adc.size()];
    acceleration_window(
        std::span<const AdcSample>( m_scratch.data(), m_adcCount ), nowNs, m_featureConfig, m_feature );
    // Copy only measured WFS evidence; prebuilt ADC matrices/validity are neither copied nor trusted.
    m_origin.m_stamp            = record.m_stamp;
    m_origin.m_residual         = record.m_residual;
    m_origin.m_exposureStartNs  = record.m_exposureStartNs;
    m_origin.m_exposureEndNs    = record.m_exposureEndNs;
    m_origin.m_decisionNs       = record.m_decisionNs;
    m_origin.m_acceleration     = m_feature.m_value; // Ignore all caller-supplied accelerometer values/validity.
    m_origin.m_accelAcquiredNs  = m_feature.m_effectiveNs;
    m_origin.m_accelAvailableNs = m_feature.m_availableNs;
    m_origin.m_accelGeneration  = m_feature.m_generation;
    m_origin.m_accelSegment     = m_feature.m_segment;
    m_origin.m_accelValid       = m_feature.m_reason == FeatureReason::Ready;
    const bool accepted         = consume( m_inputs.push_wfs( m_origin, nowNs ) );
    m_evidence.m_commonValid    = accepted && m_havePair && m_lastPair + 1 == record.m_stamp.m_counter &&
                                  m_count >= std::size_t( m_models.m_history - 1 );
    bool features               = m_origin.m_accelValid && m_count >= std::size_t( m_models.m_accelHistory - 1 );
    for( int i = 0; features && i < m_models.m_accelHistory - 1; ++i )
        features = m_aValid[( m_head + m_depth - 1 - i ) % m_depth] != 0;
    m_evidence.m_featuresValid = features;
    m_evidence.m_featureReason = m_feature.m_reason;
    m_evidence.m_standardReady = m_evidence.m_assistedReady = false;
    if( m_evidence.m_commonValid )
    {
        state( m_standardState, record.m_residual, false );
        m_evidence.m_standardReady = m_controller.predict_cached(
            ShadowMode::Standard, record.m_stamp.m_counter, 1, m_standardState, m_evidence.m_standard );
        if( features )
        {
            state( m_assistedState, record.m_residual, true );
            m_evidence.m_assistedReady = m_controller.predict_cached(
                ShadowMode::AccelAssisted, record.m_stamp.m_counter, 1, m_assistedState, m_evidence.m_assisted );
        }
    }
    const auto previousMode = m_status.m_effective;
    m_policy.decide( m_evidence, m_previous, m_zero, m_decision );
    m_previous           = m_decision.m_command; // Diagnostic continuity only; never used by consume()/push_command().
    m_status.m_requested = m_decision.m_requested;
    m_status.m_effective = m_decision.m_effective;
    m_status.m_candidateValid = m_decision.m_candidateValid;
    m_status.m_commonValid    = m_evidence.m_commonValid;
    m_status.m_reason         = m_decision.m_reason;
    m_status.m_validStreak    = m_decision.m_validStreak;
    m_status.m_step           = m_decision.m_step;
    m_status.m_featureReason  = m_feature.m_reason;
    m_status.m_index          = record.m_stamp.m_counter;
    m_status.m_originNs       = nowNs;
    m_status.m_featureNs      = m_feature.m_effectiveNs;
    m_status.m_availableNs    = m_feature.m_availableNs;
    ++m_status.m_decisions;
    m_status.m_fallbacks += previousMode == PolicyMode::Assisted && m_status.m_effective != PolicyMode::Assisted;
    m_status.m_recoveries += previousMode != PolicyMode::Assisted && m_status.m_effective == PolicyMode::Assisted;
    return accepted;
}

bool ShadowPipeline::push_command( const AppliedInputRecord &record, std::int64_t nowNs )
{
    return admit_time( nowNs ) && consume( m_inputs.push_command( record, nowNs ) );
}

bool ShadowPipeline::advance( std::int64_t nowNs )
{
    return admit_time( nowNs ) && consume( m_inputs.advance( nowNs ) );
}

void ShadowPipeline::invalidate()
{
    consume( m_inputs.invalidate() );
    break_adc();
    m_haveAdc = false;
}

ShadowPipelineStatus ShadowPipeline::status() const
{
    return m_status;
}

InputEvidence ShadowPipeline::evidence() const
{
    return m_inputs.evidence();
}

bool ShadowPipeline::candidate( Matrix &output ) const
{
    if( output.rows() != m_models.m_modes || output.cols() != 1 )
        return false;
    output.setZero();
    if( !m_status.m_candidateValid )
        return false;
    output = m_decision.m_command;
    return true;
}
} // namespace DDSPC
