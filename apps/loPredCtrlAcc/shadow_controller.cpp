/** \file shadow_controller.cpp
 * \brief Independent shadow DDSPC training and causal conditional prediction diagnostics.
 */

#include "shadow_controller.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace DDSPC
{

const char *shadow_reason( ShadowReason reason )
{
    switch( reason )
    {
    case ShadowReason::AwaitingRecord:
        return "awaiting_verified_record";
    case ShadowReason::InvalidData:
        return "invalid_or_missing_data";
    case ShadowReason::Warming:
        return "warming_history";
    case ShadowReason::Untrained:
        return "insufficient_training";
    case ShadowReason::Unexcited:
        return "insufficient_excitation";
    case ShadowReason::NumericalFault:
        return "numerical_fault_reset_required";
    case ShadowReason::Ready:
        return "ready_for_shadow_diagnostics";
    }
    return "unknown";
}

ShadowController::ShadowController( const ShadowConfig &config ) : m_config( config )
{
    if( config.m_accelChannels <= 0 || config.m_accelHistory <= 0 || !std::isfinite( config.m_lambda ) ||
        config.m_lambda <= 0 || !std::isfinite( config.m_rankTolerance ) || config.m_rankTolerance <= 0 ||
        config.m_rankTolerance >= 1 )
        throw std::invalid_argument( "Invalid shadow DDSPC configuration" );
    initialize( m_standard, false );
    initialize( m_assisted, true );
}

void ShadowController::initialize( Model &item, bool assisted )
{
    Model fresh;
    fresh.m_controller          = std::make_unique<PredictiveController>( m_config.m_modes,
                                                                          m_config.m_history,
                                                                          m_config.m_future,
                                                                          0,
                                                                          m_config.m_gamma,
                                                                          m_config.m_lambda,
                                                                          m_config.m_prior,
                                                                          assisted ? m_config.m_accelChannels : 0,
                                                                          assisted ? m_config.m_accelHistory : 0,
                                                                          HistoryMode::Causal );
    fresh.m_controller->use_qrd = m_config.m_useQrd;
    fresh.m_workspace           = fresh.m_controller->make_workspace();
    fresh.m_status.m_features   = fresh.m_controller->prediction_matrix_view().cols();
    const int p = fresh.m_status.m_features, d = m_config.m_future * m_config.m_modes;
    fresh.m_features = Matrix::Zero( fresh.m_status.m_features, fresh.m_status.m_features );
    fresh.m_context  = Matrix::Zero( p, 1 );
    fresh.m_error    = Matrix::Zero( d, 1 );
    fresh.m_scaled   = Matrix::Zero( p, p );
    fresh.m_rankQR   = std::make_unique<Eigen::ColPivHouseholderQR<Matrix>>( p, p );
    fresh.m_proposal = Matrix::Zero( m_config.m_modes, 1 );
    fresh.m_forecasts.resize( m_config.m_future + 1 );
    for( auto &forecast : fresh.m_forecasts )
    {
        forecast.m_theta = Matrix::Zero( d, p );
        forecast.m_state = Matrix::Zero( p - d, 1 );
    }
    item = std::move( fresh );
}

ShadowController::Model &ShadowController::model( ShadowMode mode )
{
    if( mode == ShadowMode::Standard )
        return m_standard;
    if( mode == ShadowMode::AccelAssisted )
        return m_assisted;
    throw std::invalid_argument( "Unknown shadow DDSPC mode" );
}

const ShadowController::Model &ShadowController::model( ShadowMode mode ) const
{
    if( mode == ShadowMode::Standard )
        return m_standard;
    if( mode == ShadowMode::AccelAssisted )
        return m_assisted;
    throw std::invalid_argument( "Unknown shadow DDSPC mode" );
}

void ShadowController::break_history( Model &item, bool resetBuffers )
{
    if( resetBuffers )
        item.m_controller->reset_buffers();
    const bool enabled              = item.m_status.m_trainingEnabled;
    const auto updates              = item.m_status.m_updates;
    const int  features             = item.m_status.m_features;
    item.m_status                   = ShadowStatus{};
    item.m_status.m_trainingEnabled = enabled;
    item.m_status.m_updates         = updates;
    item.m_status.m_features        = features;
    item.m_status.m_reason          = item.m_fault ? ShadowReason::NumericalFault : ShadowReason::InvalidData;
    item.m_featureRows = item.m_segmentUpdates = item.m_errorValues = 0;
    item.m_squaredError                                             = 0;
    item.m_proposal.setZero();
    item.m_features.setZero();
    item.m_rankDirty   = true;
    item.m_gainCurrent = false;
    for( auto &forecast : item.m_forecasts )
        forecast.m_valid = false;
}

void ShadowController::invalidate( const ShadowRecord &flags )
{
    m_validity                = ShadowRecord{};
    m_validity.m_wfsValid     = flags.m_wfsValid;
    m_validity.m_appliedValid = flags.m_appliedValid;
    m_validity.m_timingValid  = flags.m_timingValid;
    m_validity.m_accelValid   = flags.m_accelValid;
    break_history( m_standard );
    break_history( m_assisted );
    m_haveIndex = false;
}

bool ShadowController::process( const ShadowRecord &record )
{
    ShadowRecord flags;
    flags.m_wfsValid     = record.m_wfsValid && record.m_residual.rows() == m_config.m_modes &&
                           record.m_residual.cols() == 1 && record.m_residual.allFinite();
    flags.m_appliedValid = record.m_appliedValid && record.m_appliedCommand.rows() == m_config.m_modes &&
                           record.m_appliedCommand.cols() == 1 && record.m_appliedCommand.allFinite();
    flags.m_timingValid  = record.m_timingValid;
    flags.m_accelValid   = record.m_accelValid && record.m_acceleration.rows() == m_config.m_accelChannels &&
                           record.m_acceleration.cols() == 1 && record.m_acceleration.allFinite();
    if( !flags.m_wfsValid || !flags.m_appliedValid || !flags.m_timingValid ||
        record.m_index == std::numeric_limits<std::uint64_t>::max() )
    {
        invalidate( flags );
        return false;
    }
    if( m_haveIndex && record.m_index != m_lastIndex + 1 )
    {
        break_history( m_standard );
        break_history( m_assisted );
        if( record.m_index <= m_lastIndex )
        {
            m_validity = ShadowRecord{};
            return false;
        }
    }
    m_validity  = flags;
    m_haveIndex = true;
    m_lastIndex = record.m_index;
    step( m_standard, record, false );
    if( flags.m_accelValid )
        step( m_assisted, record, true );
    else
        break_history( m_assisted );
    return true;
}

void ShadowController::step( Model &item, const ShadowRecord &record, bool assisted )
{
    auto &state       = item.m_status;
    state.m_training  = false;
    state.m_dataValid = true;
    if( item.m_fault )
        return;
    try
    {
        const auto index = state.m_samples;
        item.m_controller->append_sample( index,
                                          record.m_residual,
                                          record.m_appliedCommand,
                                          assisted ? record.m_acceleration : item.m_emptyAcceleration );
        ++state.m_samples;
        auto         &workspace = *item.m_workspace;
        const bool    mature    = item.m_controller->prepare_training( workspace );
        const Matrix &phi = workspace.regressor(), &target = workspace.target();
        if( mature )
        {
            auto &origin = item.m_forecasts[( index - m_config.m_future ) % item.m_forecasts.size()];
            if( origin.m_valid )
            {
                item.m_context << origin.m_state, phi.topRows( m_config.m_future * m_config.m_modes );
                item.m_error.noalias() = origin.m_theta * item.m_context;
                item.m_error -= target;
                if( !item.m_error.allFinite() )
                    throw std::runtime_error( "Nonfinite shadow forecast" );
                item.m_squaredError += item.m_error.cast<double>().squaredNorm();
                item.m_errorValues += item.m_error.size();
                state.m_rmse = std::sqrt( item.m_squaredError / item.m_errorValues );
                ++state.m_predictions;
                state.m_predictionValid = true;
            }
            const auto row = item.m_featureRows % state.m_features;
            for( int j = 0; !item.m_rankDirty && j < state.m_features; ++j )
                if( std::memcmp( &item.m_features( row, j ), &phi( j ), sizeof( realT ) ) != 0 )
                    item.m_rankDirty = true;
            item.m_features.row( row ) = phi.transpose();
            ++item.m_featureRows;
            if( state.m_trainingEnabled )
            {
                item.m_gainCurrent = false;
                item.m_controller->update_system( workspace );
                if( !item.m_controller->prediction_matrix_view().allFinite() )
                    throw std::runtime_error( "Nonfinite shadow identification" );
                ++state.m_updates;
                ++item.m_segmentUpdates;
                state.m_training = true;
            }
        }
        state.m_ready  = false;
        state.m_reason = ShadowReason::Warming;
        if( mature )
        {
            state.m_reason = ShadowReason::Untrained;
            if( item.m_segmentUpdates >= std::max<std::uint64_t>( state.m_features, m_config.m_minUpdates ) )
            {
                if( item.m_rankDirty )
                {
                    item.m_scaled = item.m_features;
                    for( int j = 0; j < item.m_scaled.cols(); ++j )
                    {
                        const realT norm = item.m_scaled.col( j ).norm();
                        if( norm > 0 )
                            item.m_scaled.col( j ) /= norm;
                    }
                    item.m_rankQR->compute( item.m_scaled );
                    item.m_rankQR->setThreshold( m_config.m_rankTolerance );
                    state.m_rank     = item.m_rankQR->rank();
                    item.m_rankDirty = false;
                }
                state.m_reason = ShadowReason::Unexcited;
                if( state.m_rank == state.m_features )
                {
                    if( !item.m_gainCurrent )
                    {
                        item.m_controller->get_reference_controller( workspace, m_config.m_lambda );
                        item.m_gainCurrent = true;
                    }
                    else
                        item.m_controller->prepare_prediction_state( workspace );
                    const Matrix &gain        = workspace.gain();
                    item.m_proposal.noalias() = gain.bottomRows( m_config.m_modes ) * workspace.state();
                    if( !item.m_proposal.allFinite() )
                        throw std::runtime_error( "Nonfinite shadow proposal" );
                    state.m_ready  = true;
                    state.m_reason = ShadowReason::Ready;
                }
            }
        }
        auto &next   = item.m_forecasts[index % item.m_forecasts.size()];
        next.m_valid = state.m_ready;
        if( next.m_valid )
        {
            next.m_theta = workspace.theta();
            next.m_state = workspace.state();
        }
    }
    catch( const std::exception & )
    {
        item.m_fault = true;
        break_history( item );
    }
}

void ShadowController::training( ShadowMode mode, bool enabled )
{
    model( mode ).m_status.m_trainingEnabled = enabled;
    model( mode ).m_status.m_training        = false;
}

void ShadowController::reset( ShadowMode mode )
{
    auto &item = model( mode );
    item.m_controller->reset();
    item.m_fault = false;
    break_history( item, false );
    item.m_status.m_updates = 0;
    item.m_status.m_reason  = ShadowReason::AwaitingRecord;
}

bool ShadowController::select( ShadowMode mode )
{
    if( mode != ShadowMode::Standard && mode != ShadowMode::AccelAssisted )
        return false;
    if( mode == m_selected )
        return true;
    if( !model( mode ).m_status.m_ready || !model( mode ).m_status.m_dataValid )
        return false;
    m_selected = mode;
    return true;
}

ShadowMode ShadowController::selected() const
{
    return m_selected;
}
ShadowStatus ShadowController::status( ShadowMode mode ) const
{
    return model( mode ).m_status;
}
ShadowRecord ShadowController::validity() const
{
    return m_validity;
}
Matrix ShadowController::prediction_matrix( ShadowMode mode )
{
    return model( mode ).m_controller->get_reference_prediction_matrix();
}
Matrix ShadowController::proposal( ShadowMode mode ) const
{
    return model( mode ).m_status.m_ready ? model( mode ).m_proposal : Matrix();
}

bool ShadowController::predict_cached( ShadowMode    mode,
                                       std::uint64_t decisionIndex,
                                       std::uint64_t maxAgeFrames,
                                       const Matrix &state,
                                       Matrix       &output ) const
{
    if( output.rows() != m_config.m_modes || output.cols() != 1 || output.data() == state.data() )
        return false;
    output.setZero();
    if( mode != ShadowMode::Standard && mode != ShadowMode::AccelAssisted )
        return false;
    const auto &item = model( mode );
    if( !m_haveIndex || !item.m_status.m_ready || !item.m_status.m_dataValid || item.m_fault ||
        decisionIndex == std::numeric_limits<std::uint64_t>::max() || decisionIndex < m_lastIndex ||
        decisionIndex - m_lastIndex > maxAgeFrames || state.rows() != item.m_workspace->gain().cols() ||
        state.cols() != 1 || !state.allFinite() )
        return false;
    output.noalias() = item.m_workspace->gain().bottomRows( m_config.m_modes ) * state;
    if( !output.allFinite() )
    {
        output.setZero();
        return false;
    }
    return true;
}

} // namespace DDSPC
