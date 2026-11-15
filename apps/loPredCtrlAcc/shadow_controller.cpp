/** \file shadow_controller.cpp
 * \brief Independent shadow DDSPC training and causal conditional prediction diagnostics.
 */

#include "shadow_controller.hpp"
#include "training_profile.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace DDSPC
{

ShadowPredictionQueue::ShadowPredictionQueue( std::size_t capacity )
{
    if( capacity > 65536 )
        throw std::invalid_argument( "Prediction queue exceeds bound" );
    if( capacity )
        m_rows.resize( capacity + 1 );
}
bool ShadowPredictionQueue::push( const ShadowPrediction &sample ) noexcept
{
    if( m_rows.empty() )
        return false;
    const auto head = m_head.load( std::memory_order_relaxed );
    const auto next = ( head + 1 ) % m_rows.size();
    if( next == m_tail.load( std::memory_order_acquire ) )
    {
        ++m_drops;
        return false;
    }
    m_rows[head] = sample;
    m_head.store( next, std::memory_order_release );
    return true;
}
bool ShadowPredictionQueue::pop( ShadowPrediction &sample ) noexcept
{
    if( m_rows.empty() )
        return false;
    const auto tail = m_tail.load( std::memory_order_relaxed );
    if( tail == m_head.load( std::memory_order_acquire ) )
        return false;
    sample = m_rows[tail];
    m_tail.store( ( tail + 1 ) % m_rows.size(), std::memory_order_release );
    return true;
}
std::size_t ShadowPredictionQueue::queued() const noexcept
{
    return m_rows.empty() ? 0 : ( m_head.load() + m_rows.size() - m_tail.load() ) % m_rows.size();
}
std::uint64_t ShadowPredictionQueue::drops() const noexcept
{
    return m_drops.load();
}
void ShadowPredictionQueue::reset() noexcept
{
    m_head.store( 0 );
    m_tail.store( 0 );
    m_drops.store( 0 );
}

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
        config.m_rankTolerance >= 1 || config.m_assistedCommandSlots < 1 || config.m_assistedCommandSlots > 4 ||
        config.m_standardCommandSlots < 1 || config.m_standardCommandSlots > 4 || !config.m_updateStride ||
        config.m_updateStride > 64 )
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
                                                                          HistoryMode::Causal,
                                                                          assisted ? m_config.m_assistedCommandSlots
                                                                                   : m_config.m_standardCommandSlots );
    fresh.m_controller->use_qrd = m_config.m_useQrd;
    fresh.m_workspace           = fresh.m_controller->make_workspace();
    fresh.m_status.m_features   = fresh.m_controller->prediction_matrix_view().cols();
    const int p = fresh.m_status.m_features, d = m_config.m_future * m_config.m_modes;
    const int q              = d * ( assisted ? m_config.m_assistedCommandSlots : m_config.m_standardCommandSlots );
    fresh.m_features         = Matrix::Zero( fresh.m_status.m_features, fresh.m_status.m_features );
    fresh.m_context          = Matrix::Zero( p, 1 );
    fresh.m_error            = Matrix::Zero( d, 1 );
    fresh.m_scoredPrediction = Matrix::Zero( d, 1 );
    fresh.m_scaled           = Matrix::Zero( p, p );
    fresh.m_rankQR           = std::make_unique<Eigen::ColPivHouseholderQR<Matrix>>( p, p );
    fresh.m_proposal         = Matrix::Zero(
        m_config.m_modes * ( assisted ? m_config.m_assistedCommandSlots : m_config.m_standardCommandSlots ), 1 );
    fresh.m_forecasts.resize( m_config.m_future + 1 );
    for( auto &forecast : fresh.m_forecasts )
    {
        forecast.m_theta               = Matrix::Zero( d, p );
        forecast.m_state               = Matrix::Zero( p - q, 1 );
        forecast.m_availablePrediction = Matrix::Zero( d, 1 );
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
    item.m_originSquaredError                                       = 0;
    item.m_originErrorValues                                        = 0;
    item.m_proposal.setZero();
    item.m_features.setZero();
    item.m_rankDirty   = true;
    item.m_gainCurrent = false;
    for( auto &forecast : item.m_forecasts )
        forecast.m_valid = false;
}

void ShadowController::invalidate( const ShadowRecord &flags )
{
    m_validity                        = ShadowRecord{};
    m_validity.m_wfsValid             = flags.m_wfsValid;
    m_validity.m_appliedValid         = flags.m_appliedValid;
    m_validity.m_assistedAppliedValid = flags.m_assistedAppliedValid;
    m_validity.m_timingValid          = flags.m_timingValid;
    m_validity.m_accelValid           = flags.m_accelValid;
    break_history( m_standard );
    break_history( m_assisted );
    m_haveIndex = false;
}

bool ShadowController::process( const ShadowRecord &record )
{
    DDSPC_PROFILE_SCOPE( Processing );
    ShadowRecord flags;
    flags.m_wfsValid     = record.m_wfsValid && record.m_residual.rows() == m_config.m_modes &&
                           record.m_residual.cols() == 1 && record.m_residual.allFinite();
    flags.m_appliedValid = record.m_appliedValid &&
                           record.m_appliedCommand.rows() == m_config.m_modes * m_config.m_standardCommandSlots &&
                           record.m_appliedCommand.cols() == 1 && record.m_appliedCommand.allFinite();
    flags.m_assistedAppliedValid =
        m_config.m_assistedCommandSlots == 1
            ? flags.m_appliedValid
            : record.m_assistedAppliedValid &&
                  record.m_assistedAppliedCommand.rows() == m_config.m_modes * m_config.m_assistedCommandSlots &&
                  record.m_assistedAppliedCommand.cols() == 1 && record.m_assistedAppliedCommand.allFinite();
    flags.m_timingValid = record.m_timingValid;
    flags.m_accelValid  = record.m_accelValid && record.m_acceleration.rows() == m_config.m_accelChannels &&
                          record.m_acceleration.cols() == 1 && record.m_acceleration.allFinite();
    if( !flags.m_wfsValid || ( !flags.m_appliedValid && !flags.m_assistedAppliedValid ) || !flags.m_timingValid ||
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
    if( flags.m_appliedValid )
        step( m_standard, record, false );
    else
        break_history( m_standard );
    if( flags.m_accelValid && flags.m_assistedAppliedValid )
        step( m_assisted, record, true );
    else
        break_history( m_assisted );
    return true;
}

void ShadowController::invalidate_acceleration()
{
    break_history( m_assisted );
    m_validity.m_accelValid = false;
}

void ShadowController::discard_next_forecasts()
{
    for( auto *item : { &m_standard, &m_assisted } )
        item->m_forecasts[item->m_status.m_samples % item->m_forecasts.size()].m_valid = false;
}

void ShadowController::mark_next_forecasts_late()
{
    for( auto *item : { &m_standard, &m_assisted } )
        item->m_forecasts[item->m_status.m_samples % item->m_forecasts.size()].m_late = true;
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
        DDSPC_PROFILE_BEGIN( History, historyWork );
        const auto index = state.m_samples;
        item.m_controller->append_sample(
            index,
            record.m_residual,
            assisted && m_config.m_assistedCommandSlots > 1 ? record.m_assistedAppliedCommand : record.m_appliedCommand,
            assisted ? record.m_acceleration : item.m_emptyAcceleration );
        ++state.m_samples;
        auto      &workspace = *item.m_workspace;
        const bool mature    = item.m_controller->prepare_training( workspace );
        DDSPC_PROFILE_END( historyWork );
        const Matrix &phi = workspace.regressor(), &target = workspace.target();
        if( mature )
        {
            auto &origin = item.m_forecasts[( index - m_config.m_future ) % item.m_forecasts.size()];
            if( origin.m_valid )
            {
                DDSPC_PROFILE_SCOPE( Diagnostics );
                item.m_context << origin.m_state,
                    phi.topRows( m_config.m_future * m_config.m_modes *
                                 ( assisted ? m_config.m_assistedCommandSlots : m_config.m_standardCommandSlots ) );
                item.m_error.noalias() = origin.m_theta * item.m_context;
                if( m_predictionSink )
                    item.m_scoredPrediction = item.m_error;
                item.m_error -= target;
                if( !item.m_error.allFinite() )
                    throw std::runtime_error( "Nonfinite shadow forecast" );
                if( origin.m_availableValid )
                {
                    for( int row = 0; row < target.rows(); ++row )
                    {
                        const double error = double( origin.m_availablePrediction( row ) ) - target( row );
                        item.m_originSquaredError += error * error;
                    }
                    item.m_originErrorValues += target.rows();
                    state.m_originRmse = std::sqrt( item.m_originSquaredError / item.m_originErrorValues );
                    ++state.m_originPredictions;
                }
                if( m_predictionSink )
                    for( int row = 0; row < target.rows(); ++row )
                    {
                        ShadowPrediction score{ assisted ? ShadowMode::AccelAssisted : ShadowMode::Standard,
                                                origin.m_index,
                                                record.m_index - unsigned( row / m_config.m_modes ),
                                                unsigned( m_config.m_future - row / m_config.m_modes ),
                                                unsigned( row % m_config.m_modes ),
                                                double( item.m_scoredPrediction( row ) ),
                                                double( target( row ) ),
                                                origin.m_late };
                        score.m_originValid = origin.m_availableValid;
                        if( score.m_originValid )
                            score.m_originPrediction = origin.m_availablePrediction( row );
                        m_predictionSink->prediction( score );
                    }
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
            if( state.m_trainingEnabled && index % m_config.m_updateStride == 0 )
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
                if( item.m_rankDirty && index % m_config.m_updateStride == 0 )
                {
                    DDSPC_PROFILE_SCOPE( Rank );
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
                        DDSPC_PROFILE_SCOPE( Reference );
                        item.m_controller->get_reference_controller( workspace, m_config.m_lambda );
                        item.m_gainCurrent = true;
                    }
                    else
                        item.m_controller->prepare_prediction_state( workspace );
                    const Matrix &gain        = workspace.gain();
                    item.m_proposal.noalias() = gain.bottomRows( item.m_proposal.rows() ) * workspace.state();
                    if( !item.m_proposal.allFinite() )
                        throw std::runtime_error( "Nonfinite shadow proposal" );
                    state.m_ready  = true;
                    state.m_reason = ShadowReason::Ready;
                }
            }
        }
        if( m_config.m_externalOrigins )
            return; // Native forecasts must be frozen before u_i exists, through predict_next().
        auto &next   = item.m_forecasts[index % item.m_forecasts.size()];
        next.m_valid = state.m_ready;
        if( next.m_valid )
        {
            next.m_index          = record.m_index;
            next.m_availableValid = false;
            next.m_late           = false;
            DDSPC_PROFILE_SCOPE( Diagnostics );
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
void ShadowController::prediction_sink( ShadowPredictionSink *sink )
{
    m_predictionSink = sink;
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

void ShadowController::available_forecast( Model &item, Forecast &origin )
{
    const Matrix lastInput                 = item.m_controller->get_current_command_past( 1 );
    const int    inputs                    = lastInput.rows();
    origin.m_availablePrediction.noalias() = origin.m_theta.leftCols( origin.m_state.rows() ) * origin.m_state;
    for( int h = 0; h < m_config.m_future; ++h )
        origin.m_availablePrediction.noalias() +=
            origin.m_theta.middleCols( origin.m_state.rows() + h * inputs, inputs ) * lastInput;
    if( !origin.m_availablePrediction.allFinite() )
        throw std::runtime_error( "Nonfinite origin-only DDSPC forecast" );
    origin.m_availableValid = true;
}

bool ShadowController::freeze_next(
    ShadowMode mode, std::uint64_t index, const Matrix &measurement, const Matrix &acceleration, bool late )
{
    auto &item = model( mode );
    if( !m_config.m_externalOrigins || !m_haveIndex || index != m_lastIndex + 1 || !item.m_status.m_ready ||
        !item.m_status.m_dataValid || item.m_fault )
        return false;
    try
    {
        auto &origin = item.m_forecasts[item.m_status.m_samples % item.m_forecasts.size()];
        item.m_controller->next_prediction_state( measurement, acceleration, origin.m_state );
        origin.m_theta = item.m_workspace->theta();
        available_forecast( item, origin );
        origin.m_index = index;
        origin.m_late  = late;
        origin.m_valid = true;
        return true;
    }
    catch( const std::exception & )
    {
        item.m_fault = true;
        break_history( item );
        return false;
    }
}

bool ShadowController::predict_next( ShadowMode    mode,
                                     std::uint64_t index,
                                     const Matrix &measurement,
                                     const Matrix &acceleration,
                                     Matrix       &plan,
                                     const Matrix *fixedPrefix )
{
    auto      &item = model( mode );
    const auto rows = m_config.m_future * item.m_proposal.rows();
    if( plan.rows() != rows || plan.cols() != 1 )
        return false;
    plan.setZero();
    if( !m_config.m_externalOrigins || !m_haveIndex || index != m_lastIndex + 1 || !item.m_status.m_ready ||
        !item.m_status.m_dataValid || item.m_fault )
        return false;
    try
    {
        auto &origin = item.m_forecasts[item.m_status.m_samples % item.m_forecasts.size()];
        item.m_controller->next_prediction_state( measurement, acceleration, origin.m_state );
        if( fixedPrefix && fixedPrefix->size() )
        {
            const int fixed = fixedPrefix->rows();
            if( fixedPrefix->cols() != 1 || fixed >= rows || fixed % m_config.m_modes || !fixedPrefix->allFinite() )
                return false;
            const auto &theta = item.m_workspace->theta();
            Matrix      known =
                theta.leftCols( origin.m_state.rows() ) * origin.m_state + theta.rightCols( fixed ) * *fixedPrefix;
            Matrix empty( theta.rows(), 0 );
            plan.topRows( rows - fixed ) = reference_controller(
                known, empty, empty, theta.middleCols( theta.cols() - rows, rows - fixed ), m_config.m_lambda );
            plan.bottomRows( fixed ) = *fixedPrefix;
        }
        else
            plan.noalias() = item.m_workspace->gain() * origin.m_state;
        if( !plan.allFinite() )
            throw std::runtime_error( "Nonfinite next-origin DDSPC plan" );
        origin.m_theta = item.m_workspace->theta();
        available_forecast( item, origin );
        origin.m_index = index;
        origin.m_late  = false;
        origin.m_valid = true;
        return true;
    }
    catch( const std::exception & )
    {
        item.m_fault = true;
        break_history( item );
        plan.setZero();
        return false;
    }
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
