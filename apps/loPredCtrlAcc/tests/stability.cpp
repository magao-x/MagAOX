/** \file stability.cpp
 * \brief Test-only symmetric covariance backend and bounded, explicitly sourced synthetic recovery actions.
 */
#include "stability.hpp"
// Reopen only application guards for the independently named pinned candidate; Eigen guards remain intact.
#undef loPredCtrlAcc_async_shadow_hpp
#undef PCSHADOW_HPP
#undef PCARC_HPP
#undef PCRLS_HPP
#undef PC_QRDRLS_HPP
#undef PCUTILS_HPP
#define DDSPC DDSPCSymmetric
#include "symmetric/tests/async_shadow.hpp"
#undef DDSPC
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
/// Convert an independent namespace's coherent metadata without changing values or combining publications.
template <class Source>
void metadata( DDSPC::OfflineModelInfo &out /**< [out] Destination. */, const Source &in /**< [in] One publication. */ )
{
    out.m_generation           = in.m_generation;
    out.m_reset                = in.m_reset;
    out.m_publication          = in.m_publication;
    out.m_version              = in.m_version;
    out.m_sequence             = in.m_sequence;
    out.m_streamEpoch          = in.m_streamEpoch;
    out.m_measurementNs        = in.m_measurementNs;
    out.m_availableNs          = in.m_availableNs;
    out.m_publishedNs          = in.m_publishedNs;
    out.m_segmentFirstSequence = in.m_segmentFirstSequence;
    auto       &a              = out.m_status;
    const auto &b              = in.m_status;
    a.m_ready                  = b.m_ready;
    a.m_trainingEnabled        = b.m_trainingEnabled;
    a.m_training               = b.m_training;
    a.m_dataValid              = b.m_dataValid;
    a.m_predictionValid        = b.m_predictionValid;
    a.m_samples                = b.m_samples;
    a.m_updates                = b.m_updates;
    a.m_predictions            = b.m_predictions;
    a.m_features               = b.m_features;
    a.m_rank                   = b.m_rank;
    a.m_rmse                   = b.m_rmse;
    a.m_reason                 = static_cast<DDSPC::ShadowReason>( b.m_reason );
}
/// Preserve immutable dimensions, statistical parameters and readiness thresholds for the candidate.
DDSPCSymmetric::ShadowConfig config( const DDSPC::ShadowConfig &in /**< [in] Reference configuration. */ )
{
    DDSPCSymmetric::ShadowConfig out;
    out.m_modes         = in.m_modes;
    out.m_history       = in.m_history;
    out.m_future        = in.m_future;
    out.m_accelChannels = in.m_accelChannels;
    out.m_accelHistory  = in.m_accelHistory;
    out.m_gamma         = in.m_gamma;
    out.m_prior         = in.m_prior;
    out.m_lambda        = in.m_lambda;
    out.m_rankTolerance = in.m_rankTolerance;
    out.m_minUpdates    = in.m_minUpdates;
    out.m_useQrd        = in.m_useQrd;
    return out;
}
/// Namespace types have identical declared lane identities; no model-selection coupling is introduced.
DDSPCSymmetric::ShadowMode mode( DDSPC::ShadowMode in /**< [in] Reference lane. */ )
{
    return static_cast<DDSPCSymmetric::ShadowMode>( in ); // Preserve invalid values for the backend's existing checks.
}
} //namespace

struct StabilityWorker::Impl
{
    std::unique_ptr<DDSPC::OfflineAsyncShadow>          m_original;  ///< Reference pair, if selected.
    std::unique_ptr<DDSPCSymmetric::OfflineAsyncShadow> m_symmetric; ///< Separate candidate pair, if selected.
    DDSPCSymmetric::OfflineRecord                       m_record;    ///< Fixed candidate ingress storage.
    std::array<DDSPCSymmetric::OfflineSnapshot, 2>      m_snapshots; ///< Fixed candidate diagnostic copies.
    /// Allocate the independent backend and immutable buffers.
    Impl( const DDSPC::ShadowConfig &c /**< [in] Dimensions. */,
          std::size_t                capacity /**< [in] Bound. */,
          bool                       symmetric /**< [in] Estimator candidate selection. */ );
};
StabilityWorker::Impl::Impl( const DDSPC::ShadowConfig &c, std::size_t capacity, bool symmetric )
    : m_record( config( c ) ),
      m_snapshots{ DDSPCSymmetric::OfflineSnapshot( config( c ), DDSPCSymmetric::ShadowMode::Standard ),
                   DDSPCSymmetric::OfflineSnapshot( config( c ), DDSPCSymmetric::ShadowMode::AccelAssisted ) }
{
    if( symmetric )
    {
        if( c.m_useQrd )
            throw std::invalid_argument( "Symmetric RLS is not a QR replacement" );
        m_symmetric = std::make_unique<DDSPCSymmetric::OfflineAsyncShadow>(
            config( c ), capacity, 0, 1500000, DDSPCSymmetric::OfflineOverflowPolicy::Independent );
    }
    else
        m_original = std::make_unique<DDSPC::OfflineAsyncShadow>(
            c, capacity, 0, 1500000, DDSPC::OfflineOverflowPolicy::Independent );
}
StabilityWorker::StabilityWorker( const DDSPC::ShadowConfig &c, std::size_t capacity, bool symmetric )
    : m_impl( std::make_unique<Impl>( c, capacity, symmetric ) )
{
}
StabilityWorker::~StabilityWorker() = default;
DDSPC::OfflineAdmission StabilityWorker::submit_record( const DDSPC::OfflineRecord &in, std::uint64_t now )
{
    if( m_impl->m_original )
        return m_impl->m_original->submit_record( in, now );
    auto &out                   = m_impl->m_record;
    out.m_streamEpoch           = in.m_streamEpoch;
    out.m_measurementNs         = in.m_measurementNs;
    out.m_exposureStartNs       = in.m_exposureStartNs;
    out.m_appliedNs             = in.m_appliedNs;
    out.m_appliedAvailableNs    = in.m_appliedAvailableNs;
    out.m_availableNs           = in.m_availableNs;
    out.m_accelNs               = in.m_accelNs;
    out.m_accelAvailableNs      = in.m_accelAvailableNs;
    out.m_data.m_index          = in.m_data.m_index;
    out.m_data.m_residual       = in.m_data.m_residual;
    out.m_data.m_appliedCommand = in.m_data.m_appliedCommand;
    out.m_data.m_acceleration   = in.m_data.m_acceleration;
    out.m_data.m_wfsValid       = in.m_data.m_wfsValid;
    out.m_data.m_appliedValid   = in.m_data.m_appliedValid;
    out.m_data.m_timingValid    = in.m_data.m_timingValid;
    out.m_data.m_accelValid     = in.m_data.m_accelValid;
    const auto a                = m_impl->m_symmetric->submit_record( out, now );
    return { a.m_standard, a.m_assisted };
}
bool StabilityWorker::manual_step( DDSPC::ShadowMode m, std::uint64_t now )
{
    return m_impl->m_original ? m_impl->m_original->manual_step( m, now )
                              : m_impl->m_symmetric->manual_step( mode( m ), now );
}
void StabilityWorker::reset( DDSPC::ShadowMode m )
{
    if( m_impl->m_original )
        m_impl->m_original->reset( m );
    else
        m_impl->m_symmetric->reset( mode( m ) );
}
bool StabilityWorker::predict( DDSPC::ShadowMode         m,
                               std::uint64_t             epoch,
                               std::uint64_t             sequence,
                               std::uint64_t             stateNs,
                               std::uint64_t             now,
                               std::uint64_t             maxAge,
                               const DDSPC::Matrix      &state,
                               DDSPC::Matrix            &output,
                               DDSPC::OfflinePrediction &result )
{
    if( m_impl->m_original )
        return m_impl->m_original->predict( m, epoch, sequence, stateNs, now, maxAge, state, output, result );
    DDSPCSymmetric::OfflinePrediction prediction;
    const bool                        valid =
        m_impl->m_symmetric->predict( mode( m ), epoch, sequence, stateNs, now, maxAge, state, output, prediction );
    metadata( result.m_model, prediction.m_model );
    result.m_ageNs  = prediction.m_ageNs;
    result.m_reason = static_cast<DDSPC::OfflinePredictionReason>( prediction.m_reason );
    return valid;
}
bool StabilityWorker::read_snapshot( DDSPC::ShadowMode m, DDSPC::OfflineSnapshot &out )
{
    if( m_impl->m_original )
        return m_impl->m_original->read_snapshot( m, out );
    auto &in = m_impl->m_snapshots[m == DDSPC::ShadowMode::Standard ? 0 : 1];
    if( !m_impl->m_symmetric->read_snapshot( mode( m ), in ) )
        return false;
    metadata( out.m_info, in.m_info );
    out.m_theta          = in.m_theta;
    out.m_gain           = in.m_gain;
    out.m_originState    = in.m_originState;
    out.m_originProposal = in.m_originProposal;
    return true;
}
DDSPC::OfflineWorkerStats StabilityWorker::stats( DDSPC::ShadowMode m ) const
{
    if( m_impl->m_original )
        return m_impl->m_original->stats( m );
    const auto s = m_impl->m_symmetric->stats( mode( m ) );
    return { s.m_processed,
             s.m_discarded,
             s.m_publishMisses,
             s.m_depth,
             s.m_highWater,
             s.m_traceDrops,
             s.m_accepted,
             s.m_overflows,
             s.m_threadCpuNs,
             s.m_threadWallNs };
}

RecoverySupervisor::RecoverySupervisor( const VibrationConfig &c, int features, const StabilityPolicy &policy )
    : m_policy( policy ), m_features( features ), m_delay( c.m_future + std::max( c.m_history, c.m_accelHistory ) ),
      m_budget( c.m_train ), m_step( DDSPC::Matrix::Zero( 2, 1 ) ), m_previous( m_step )
{
    if( features <= 0 )
        throw std::invalid_argument( "Recovery needs a positive validation window" );
    if( !( policy.m_quantile > 0 && policy.m_quantile < 1 ) )
        throw std::invalid_argument( "Invalid empirical confidence quantile" );
    m_calibration.reserve( 1000 );
    m_errors.reserve( features );
}
void RecoverySupervisor::calibrate( const DDSPC::Matrix &command )
{
    m_step     = m_step.cwiseMax( ( command - m_previous ).cwiseAbs() );
    m_previous = command;
}
void RecoverySupervisor::observe( double error, bool calibration )
{
    if( !std::isfinite( error ) || error < 0 )
        return;
    auto &values = calibration ? m_calibration : m_errors;
    if( calibration && m_boundReady )
        return;
    const std::size_t bound = calibration ? 1000 : std::size_t( m_features );
    if( values.size() == bound )
        values.erase( values.begin() );
    values.push_back( error );
}
DDSPC::Matrix RecoverySupervisor::slew( const DDSPC::Matrix &held, const DDSPC::Matrix &target ) const
{
    DDSPC::Matrix result( held.rows(), 1 );
    for( int i = 0; i < held.rows(); ++i )
    {
        const double delta = double( target( i ) ) - held( i ), step = m_step( i );
        if( std::abs( delta ) <= step )
            result( i ) = target( i );
        else
        {
            const double next = double( held( i ) ) + std::copysign( step, delta );
            result( i )       = float( next );
            if( std::abs( double( result( i ) ) - held( i ) ) > step )
                result( i ) = std::nextafter( result( i ), held( i ) );
        }
    }
    return result;
}
DDSPC::Matrix RecoverySupervisor::command( bool                 valid,
                                           bool                 healthy,
                                           const DDSPC::Matrix &proposal,
                                           const DDSPC::Matrix &held,
                                           const DDSPC::Matrix &probe,
                                           double               limit,
                                           SimAction           &action )
{
    if( !m_policy.m_neutralReturn )
    {
        action = valid ? SimAction::Model : SimAction::Hold;
        return valid ? proposal : held;
    }
    if( !m_boundReady && m_calibration.size() >= std::size_t( m_features ) )
    {
        auto values = m_calibration;
        std::sort( values.begin(), values.end() );
        m_threshold  = values[std::size_t( m_policy.m_quantile * ( values.size() - 1 ) )];
        m_boundReady = true;
    }
    const double mean =
        m_errors.empty() ? 0 : std::accumulate( m_errors.begin(), m_errors.end(), 0. ) / m_errors.size();
    const bool confidenceBad =
        m_policy.m_confidence && m_boundReady && m_errors.size() == std::size_t( m_features ) && mean > m_threshold;
    m_bad = valid ? 0 : m_bad + 1;
    if( m_state == SimRecovery::Tracking && ( confidenceBad || m_bad >= m_delay ) )
    {
        m_state = SimRecovery::Neutral;
        m_reset = true;
        m_errors.clear();
    }
    if( m_state == SimRecovery::Neutral || m_state == SimRecovery::Disabled )
    {
        action    = SimAction::Neutral;
        auto next = slew( held, DDSPC::Matrix::Zero( 2, 1 ) );
        if( next.isZero( 0 ) && m_state != SimRecovery::Disabled )
            m_state = SimRecovery::Waiting;
        return next;
    }
    if( m_state == SimRecovery::Waiting )
    {
        action = SimAction::Neutral;
        if( healthy && m_policy.m_retrain && m_probeFrames < m_budget )
            m_state = SimRecovery::Training;
        return slew( held, DDSPC::Matrix::Zero( 2, 1 ) );
    }
    if( m_state == SimRecovery::Training )
    {
        if( !healthy || m_probeFrames >= m_budget )
        {
            m_state = m_probeFrames >= m_budget ? SimRecovery::Disabled : SimRecovery::Neutral;
            action  = SimAction::Neutral;
            return slew( held, DDSPC::Matrix::Zero( 2, 1 ) );
        }
        if( valid && m_errors.size() == std::size_t( m_features ) && m_boundReady && mean <= m_threshold )
        {
            m_state      = SimRecovery::Ramping;
            m_rampFrames = 0;
        }
        else
        {
            ++m_probeFrames;
            action = SimAction::Probe;
            return slew( held, probe );
        }
    }
    if( m_state == SimRecovery::Ramping )
    {
        if( confidenceBad || m_bad >= m_delay )
        {
            m_state = SimRecovery::Neutral;
            m_reset = true;
            m_errors.clear();
            action = SimAction::Neutral;
            return slew( held, DDSPC::Matrix::Zero( 2, 1 ) );
        }
        if( valid )
        {
            ++m_rampFrames;
            const double gain      = std::min( 1., double( m_rampFrames ) / m_features );
            const double amplitude = proposal.cwiseAbs().maxCoeff();
            const double authority = amplitude > limit ? limit / amplitude : 1.;
            action                 = SimAction::Model;
            if( m_rampFrames >= m_features )
                m_state = SimRecovery::Tracking;
            return slew( held, ( gain * authority * proposal ).eval() );
        }
    }
    if( valid )
    {
        action                 = SimAction::Model;
        const double amplitude = proposal.cwiseAbs().maxCoeff();
        const double authority = amplitude > limit ? limit / amplitude : 1.;
        return slew( held, ( authority * proposal ).eval() );
    }
    action = SimAction::Hold;
    return held;
}
bool RecoverySupervisor::take_reset()
{
    const bool reset = m_reset;
    m_reset          = false;
    return reset;
}
SimRecovery RecoverySupervisor::state() const
{
    return m_state;
}
double RecoverySupervisor::threshold() const
{
    return m_threshold;
}
const DDSPC::Matrix &RecoverySupervisor::step_bound() const
{
    return m_step;
}
} //namespace libXWCTest::loPredCtrlAccTest
