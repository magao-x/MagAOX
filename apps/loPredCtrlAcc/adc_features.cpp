/** \file adc_features.cpp
 * \brief Allocation-free causal zero-order-held ADC windows with explicit provenance.
 */
#include "adc_features.hpp"
#include <cmath>
#include <stdexcept>
namespace DDSPC
{
const char *feature_reason( FeatureReason reason )
{
    switch( reason )
    {
    case FeatureReason::Ready:
        return "ready";
    case FeatureReason::Incomplete:
        return "adc_incomplete";
    case FeatureReason::Gap:
        return "adc_gap";
    case FeatureReason::Stale:
        return "adc_stale";
    default:
        return "adc_invalid";
    }
}

AccelFeature
acceleration_window( std::span<const AdcSample> samples, std::int64_t decisionNs, const FeatureConfig &config )
{
    AccelFeature result;
    acceleration_window( samples, decisionNs, config, result );
    return result;
}

void acceleration_window( std::span<const AdcSample> samples,
                          std::int64_t               decisionNs,
                          const FeatureConfig       &c,
                          AccelFeature              &result )
{
    if( c.m_windowNs <= 0 || c.m_windowNs > 1000000000 || c.m_lagNs < 0 || c.m_lagNs > 1000000000 ||
        c.m_maxGapNs <= 0 || c.m_maxGapNs > c.m_windowNs || c.m_maxAgeNs <= 0 ||
        decisionNs < c.m_lagNs + c.m_windowNs || result.m_value.rows() != 2 || result.m_value.cols() != 1 )
        throw std::invalid_argument( "Invalid ADC feature-window contract or workspace" );
    result.m_value.setZero();
    result.m_reason      = FeatureReason::Incomplete;
    result.m_effectiveNs = result.m_latestNs = result.m_availableNs = -1;
    result.m_samples = result.m_generation = result.m_segment = 0;
    const auto end = decisionNs - c.m_lagNs, start = end - c.m_windowNs;
    const auto eligible = [end, decisionNs]( const AdcSample &s )
    { return s.m_acquiredNs < end && s.m_receivedNs <= decisionNs && s.m_publishedNs <= decisionNs; };
    std::size_t first = samples.size(), last = samples.size();
    for( std::size_t i = 0; i < samples.size(); ++i )
    {
        const auto &s = samples[i];
        if( !eligible( s ) )
            continue;
        if( s.m_acquiredNs < 0 || s.m_publishedNs < s.m_acquiredNs || s.m_receivedNs < s.m_publishedNs ||
            !s.m_generation || !s.m_segment || !std::isfinite( s.m_acceleration[0] ) ||
            !std::isfinite( s.m_acceleration[1] ) ||
            ( last < samples.size() && s.m_acquiredNs <= samples[last].m_acquiredNs ) )
        {
            result.m_reason = FeatureReason::Invalid;
            return;
        }
        last = i;
        if( s.m_acquiredNs <= start )
            first = i;
    }
    if( last == samples.size() )
        return;
    if( decisionNs - samples[last].m_acquiredNs > c.m_maxAgeNs )
    {
        result.m_reason = FeatureReason::Stale;
        return;
    }
    if( first == samples.size() )
        return;
    auto                 at           = start;
    double               weightedTime = 0;
    std::array<realT, 2> value{};
    for( std::size_t i = first; i <= last; )
    {
        const auto &s    = samples[i];
        auto        next = i + 1;
        while( next <= last && !eligible( samples[next] ) )
            ++next;
        const auto until = next <= last ? samples[next].m_acquiredNs : end;
        if( until - s.m_acquiredNs > c.m_maxGapNs ||
            ( next <= last &&
              ( samples[next].m_generation != s.m_generation || samples[next].m_segment != s.m_segment ||
                samples[next].m_frame.m_counter != s.m_frame.m_counter + 1 ) ) )
        {
            result.m_reason = FeatureReason::Gap;
            return;
        }
        const double weight = double( until - at ) / c.m_windowNs;
        for( int axis = 0; axis < 2; ++axis )
            value[axis] += realT( weight * s.m_acceleration[axis] );
        weightedTime += weight * double( s.m_acquiredNs - start );
        at                   = until;
        result.m_latestNs    = s.m_acquiredNs;
        result.m_availableNs = std::max( result.m_availableNs, s.m_receivedNs );
        ++result.m_samples;
        i = next;
    }
    result.m_effectiveNs = start + std::llround( weightedTime );
    if( decisionNs - result.m_effectiveNs > c.m_maxAgeNs )
    {
        result.m_reason = FeatureReason::Stale;
        return;
    }
    if( !std::isfinite( value[0] ) || !std::isfinite( value[1] ) )
    {
        result.m_reason = FeatureReason::Invalid;
        return;
    }
    for( int axis = 0; axis < 2; ++axis )
        result.m_value( axis ) = value[axis];
    result.m_generation = samples[first].m_generation;
    result.m_segment    = samples[first].m_segment;
    result.m_reason     = FeatureReason::Ready;
}
} // namespace DDSPC
