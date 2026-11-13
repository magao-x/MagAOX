/** \file shadow_policy.cpp
 * \brief Platform-independent DDSPC shadow selection and bounded candidate generation.
 */
#include "shadow_policy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace DDSPC
{
ControlEvidence::ControlEvidence( int modes )
{
    if( modes < 1 || modes > 4096 )
        throw std::invalid_argument( "Invalid proposal dimensions" );
    m_standard.setZero( modes, 1 );
    m_assisted.setZero( modes, 1 );
}

ControlDecision::ControlDecision( int modes )
{
    if( modes < 1 || modes > 4096 )
        throw std::invalid_argument( "Invalid candidate dimensions" );
    m_command.setZero( modes, 1 );
}

void ShadowPolicy::reset()
{
    m_effective   = PolicyMode::Safe;
    m_validStreak = 0;
}

const char *policy_mode( PolicyMode mode )
{
    switch( mode )
    {
    case PolicyMode::Standard:
        return "standard";
    case PolicyMode::Assisted:
        return "assisted";
    default:
        return "safe";
    }
}

ShadowPolicy::ShadowPolicy( PolicyMode requested, const PolicyConfig &config )
    : m_config( config ), m_requested( requested )
{
    if( config.m_modes < 1 || config.m_modes > 4096 ||
        ( requested != PolicyMode::Standard && requested != PolicyMode::Assisted ) || !config.m_recoveryFrames ||
        config.m_recoveryFrames > 100000 || !std::isfinite( config.m_limit ) || config.m_limit <= 0 ||
        !std::isfinite( config.m_maxStep ) || config.m_maxStep <= 0 )
        throw std::invalid_argument( "Invalid shadow candidate policy" );
    m_target.setZero( config.m_modes, 1 );
}

Matrix bounded_command( const Matrix &request, const Matrix &previous, const PolicyConfig &config )
{
    if( config.m_modes < 1 || config.m_modes > 4096 )
        throw std::invalid_argument( "Invalid candidate dimensions" );
    Matrix result( config.m_modes, 1 );
    bounded_command( request, previous, config, result );
    return result;
}

void bounded_command( const DDSPC::Matrix &request,
                      const DDSPC::Matrix &previous,
                      const PolicyConfig  &c,
                      Matrix              &result )
{
    if( result.rows() != c.m_modes || result.cols() != 1 || result.data() == request.data() ||
        result.data() == previous.data() || request.rows() != c.m_modes || request.cols() != 1 ||
        previous.rows() != c.m_modes || previous.cols() != 1 || !request.allFinite() || !previous.allFinite() ||
        !std::isfinite( c.m_limit ) || c.m_limit <= 0 || !std::isfinite( c.m_maxStep ) || c.m_maxStep <= 0 ||
        previous.cwiseAbs().maxCoeff() > c.m_limit )
        throw std::invalid_argument( "Invalid offline command or bound" );
    for( int axis = 0; axis < c.m_modes; ++axis )
    {
        const double lo = std::max( -c.m_limit, double( previous( axis ) ) - c.m_maxStep );
        const double hi = std::min( c.m_limit, double( previous( axis ) ) + c.m_maxStep );
        result( axis )  = float( std::clamp( double( request( axis ) ), lo, hi ) );
        // Round inward so float conversion cannot exceed a double-valued safety bound.
        if( result( axis ) > hi )
            result( axis ) = std::nextafter( result( axis ), -std::numeric_limits<float>::infinity() );
        if( result( axis ) < lo )
            result( axis ) = std::nextafter( result( axis ), std::numeric_limits<float>::infinity() );
    }
}

ControlDecision ShadowPolicy::decide( const ControlEvidence &e, const Matrix &previous, const Matrix &exploration )
{
    ControlDecision output( m_config.m_modes );
    decide( e, previous, exploration, output );
    return output;
}

void ShadowPolicy::decide( const ControlEvidence &e,
                           const DDSPC::Matrix   &previous,
                           const DDSPC::Matrix   &exploration,
                           ControlDecision       &d )
{
    const bool standard = e.m_commonValid && e.m_standardReady && e.m_standard.rows() == m_config.m_modes &&
                          e.m_standard.cols() == 1 && e.m_standard.allFinite();
    const bool assisted = e.m_commonValid && e.m_featuresValid && e.m_featureReason == FeatureReason::Ready &&
                          e.m_assistedReady && e.m_assisted.rows() == m_config.m_modes && e.m_assisted.cols() == 1 &&
                          e.m_assisted.allFinite();
    m_validStreak       = assisted ? std::min( m_validStreak + 1, m_config.m_recoveryFrames ) : 0;
    if( d.m_command.rows() != m_config.m_modes || d.m_command.cols() != 1 )
        throw std::invalid_argument( "Incorrect shadow decision shape" );
    d.m_effective   = PolicyMode::Safe;
    d.m_requested   = m_requested;
    d.m_validStreak = m_validStreak;
    if( !e.m_commonValid )
        d.m_reason = "common_input_invalid";
    else if( m_requested == PolicyMode::Standard )
        d.m_reason = standard ? "requested_standard" : "standard_unready";
    else if( e.m_featureReason != FeatureReason::Ready )
        d.m_reason = feature_reason( e.m_featureReason );
    else if( !e.m_featuresValid )
        d.m_reason = "adc_history_incomplete";
    else if( !assisted )
        d.m_reason = "assisted_unready";
    else if( m_validStreak < m_config.m_recoveryFrames )
        d.m_reason = "recovery_hysteresis";
    else
        d.m_reason = "assisted_qualified";
    if( m_requested == PolicyMode::Assisted && assisted && m_validStreak >= m_config.m_recoveryFrames )
        d.m_effective = PolicyMode::Assisted;
    else if( standard )
        d.m_effective = PolicyMode::Standard;
    m_target.setZero();
    if( d.m_effective != PolicyMode::Safe )
    {
        if( exploration.rows() != m_config.m_modes || exploration.cols() != 1 || !exploration.allFinite() )
            throw std::invalid_argument( "Invalid offline exploration" );
        m_target = ( d.m_effective == PolicyMode::Assisted ? e.m_assisted : e.m_standard ) + exploration;
    }
    bounded_command( m_target, previous, m_config, d.m_command );
    d.m_candidateValid = d.m_effective != PolicyMode::Safe;
    d.m_step           = ( d.m_command.cast<double>() - previous.cast<double>() ).cwiseAbs().maxCoeff();
    d.m_switched       = d.m_effective != m_effective;
    m_effective        = d.m_effective;
}

} // namespace DDSPC
