/** \file shadow_controller.cpp
 * \brief Independent shadow DDSPC training and causal conditional prediction diagnostics.
 */

#include "shadow_controller.hpp"
#include <algorithm>
#include <cmath>
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
    fresh.m_status.m_features   = fresh.m_controller->get_prediction_matrix().cols();
    fresh.m_features            = Matrix::Zero( fresh.m_status.m_features, fresh.m_status.m_features );
    fresh.m_forecasts.resize( m_config.m_future + 1 );
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

void ShadowController::break_history( Model &item )
{
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
    item.m_proposal.resize( 0, 0 );
    item.m_features.setZero();
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
        item.m_controller->append_sample(
            index, record.m_residual, record.m_appliedCommand, assisted ? record.m_acceleration : Matrix() );
        ++state.m_samples;
        Matrix     phi, target;
        const bool mature = item.m_controller->get_training_example( phi, target );
        if( mature )
        {
            auto &origin = item.m_forecasts[( index - m_config.m_future ) % item.m_forecasts.size()];
            if( origin.m_valid )
            {
                Matrix context( origin.m_theta.cols(), 1 );
                context << origin.m_state, phi.topRows( m_config.m_future * m_config.m_modes );
                const Matrix error = origin.m_theta * context - target;
                if( !error.allFinite() )
                    throw std::runtime_error( "Nonfinite shadow forecast" );
                item.m_squaredError += error.cast<double>().squaredNorm();
                item.m_errorValues += error.size();
                state.m_rmse = std::sqrt( item.m_squaredError / item.m_errorValues );
                ++state.m_predictions;
                state.m_predictionValid = true;
            }
            item.m_features.row( item.m_featureRows % state.m_features ) = phi.transpose();
            ++item.m_featureRows;
            if( state.m_trainingEnabled )
            {
                item.m_controller->update_system();
                if( !item.m_controller->get_prediction_matrix().allFinite() )
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
                Matrix scaled = item.m_features;
                for( int j = 0; j < scaled.cols(); ++j )
                {
                    const realT norm = scaled.col( j ).norm();
                    if( norm > 0 )
                        scaled.col( j ) /= norm;
                }
                Eigen::ColPivHouseholderQR<Matrix> qr( scaled );
                qr.setThreshold( m_config.m_rankTolerance );
                state.m_rank   = qr.rank();
                state.m_reason = ShadowReason::Unexcited;
                if( state.m_rank == state.m_features )
                {
                    const Matrix theta = item.m_controller->get_reference_prediction_matrix();
                    if( !theta.allFinite() )
                        throw std::runtime_error( "Nonfinite shadow model" );
                    const Matrix gain = item.m_controller->get_reference_controller( m_config.m_lambda );
                    item.m_proposal   = gain.bottomRows( m_config.m_modes ) * item.m_controller->get_prediction_state();
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
            next.m_theta = item.m_controller->get_reference_prediction_matrix();
            next.m_state = item.m_controller->get_prediction_state();
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
    const bool enabled = model( mode ).m_status.m_trainingEnabled;
    initialize( model( mode ), mode == ShadowMode::AccelAssisted );
    model( mode ).m_status.m_trainingEnabled = enabled;
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

} // namespace DDSPC
