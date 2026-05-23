/** \file ar_controller.cpp
 * \brief DDSPC predictive controller implementation.
 *
 * \ingroup loPredCtrl_files
 */

#include "ar_controller.hpp"
#include <algorithm>

namespace DDSPC
{

PredictiveController::PredictiveController( int   num_actuators,
                                            int   num_history,
                                            int   num_future,
                                            realT gain,
                                            realT gamma,
                                            realT initial_regularization,
                                            realT initial_covariance,
                                            int   num_accel_channels,
                                            int   accel_history )
{
    // Settable properties
    _num_modes          = num_actuators;
    _num_history        = num_history;
    _num_future         = num_future;
    _num_accel_channels = std::max( 0, num_accel_channels );
    _num_accel_history  = std::max( 0, accel_history );
    _gain               = gain;
    _delta_max          = 0.5;
    _regularization     = initial_regularization;

    // Derived properties
    num_predictors    = _num_future * _num_modes;
    num_correlations  = _num_future * _num_modes;
    num_accel_features = _num_accel_history * _num_accel_channels;
    num_features = ( _num_future - 1 + 2 * _num_history ) * _num_modes + num_accel_features;
    num_state_features = ( 2 * _num_history - 1 ) * _num_modes + num_accel_features;

    // Data buffers
    buffer_size = find_next_power_of_2( 2 * ( num_history + num_future + _num_accel_history + 1 ) );

    measurement_head = 0;
    measurement_buffer.resize( buffer_size, _num_modes );
    measurement_buffer.setZero();

    command_head = 0;
    command_buffer.resize( buffer_size, _num_modes );
    command_buffer.setZero();

    accel_head = 0;
    accel_buffer.resize( buffer_size, _num_accel_channels );
    accel_buffer.setZero();

    // The learner
    rls = new RecursiveLeastSquares( num_predictors, num_features, gamma, initial_covariance );

    // Initializing the controller
    controller.resize( _num_modes, num_state_features );
    controller.setZero();

    integrator.resize( _num_modes, num_state_features );
    integrator.setZero();

    // Architectural note:
    // state vector uses [u_p, a_p, y_p], so this index lands at the oldest y sample per mode.
    for( int i = 0; i < _num_modes; i++ )
    {
        int index = ( _num_history - 1 ) * _num_modes + num_accel_features + i;
        integrator( i, index ) = -_gain;
    }

    // Set the regularization matrix
    regularization_matrix_01.resize( num_correlations, num_correlations );
    regularization_matrix_01.setZero();

    regularization_matrix_02.resize( num_correlations, num_correlations );
    regularization_matrix_02.setZero();

    for( int i = 0; i < num_correlations; i++ )
    {
        regularization_matrix_01( i, i ) = initial_regularization;
        regularization_matrix_02( i, i ) = initial_regularization;
    }

    regularization_matrix = &regularization_matrix_01;
};


PredictiveController::~PredictiveController()
{
    delete rls;
}

void PredictiveController::reset()
{
    controller.resize( _num_modes, num_state_features );
    controller.setZero();
    integrator.setZero();

    for( int i = 0; i < _num_modes; i++ )
    {
        int index = ( _num_history - 1 ) * _num_modes + num_accel_features + i;
        integrator( i, index ) = -_gain;
    }

    rls->reset();

    measurement_head = 0;
    command_head     = 0;
    accel_head       = 0;

    measurement_buffer.setZero();
    command_buffer.setZero();
    accel_buffer.setZero();
}

void PredictiveController::set_regularization( realT new_regularization )
{
    if( use_regularization_matrix_01 )
    {
        for( int i = 0; i < num_correlations; i++ )
        {
            regularization_matrix_02( i, i ) = new_regularization;
        }
        use_regularization_matrix_01 = false;
    }
    else
    {
        for( int i = 0; i < num_correlations; i++ )
        {
            regularization_matrix_01( i, i ) = new_regularization;
        }
        use_regularization_matrix_01 = true;
    }
    do_switch_regularization_matrix = true;
}

Matrix PredictiveController::get_measurement_future()
{
    Matrix future_vec;
    future_vec.resize( _num_future * _num_modes, 1 );

    for( int i = 0; i < _num_future; i++ )
    {
        auto dat = measurement_buffer.row( ( measurement_head - i - 1 ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_modes; j++ )
        {
            future_vec( i * _num_modes + j, 0 ) = dat( j, 0 );
        }
    }

    return future_vec;
}

Matrix PredictiveController::get_measurement_past()
{
    Matrix past_vec;
    past_vec.resize( _num_history * _num_modes, 1 );

    for( int i = 0; i < _num_history; i++ )
    {
        auto dat = measurement_buffer.row( ( measurement_head - i - _num_future - 1 ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_modes; j++ )
        {
            past_vec( i * _num_modes + j, 0 ) = dat( j, 0 );
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_command_future( int skip_cmds )
{
    Matrix future_vec;
    future_vec.resize( ( _num_future - skip_cmds ) * _num_modes, 1 );

    for( int i = 0; i < ( _num_future - skip_cmds ); i++ )
    {
        // Architectural note:
        // keep existing indexing behavior to preserve legacy DDSPC command alignment.
        int offset = skip_cmds * _num_modes;
        auto dat   = command_buffer.row( ( command_head - i - 1 - offset ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_modes; j++ )
        {
            future_vec( i * _num_modes + j, 0 ) = dat( j, 0 );
        }
    }

    return future_vec;
}

Matrix PredictiveController::get_command_past()
{
    Matrix past_vec;
    past_vec.resize( _num_history * _num_modes, 1 );

    for( int i = 0; i < _num_history; i++ )
    {
        auto dat = command_buffer.row( ( command_head - i - _num_future - 1 ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_modes; j++ )
        {
            past_vec( i * _num_modes + j ) = dat( j, 0 );
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_accelerometer_past()
{
    Matrix past_vec;
    past_vec.resize( num_accel_features, 1 );
    past_vec.setZero();

    if( num_accel_features == 0 )
    {
        return past_vec;
    }

    for( int i = 0; i < _num_accel_history; i++ )
    {
        auto dat = accel_buffer.row( ( accel_head - i - 1 ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_accel_channels; j++ )
        {
            past_vec( i * _num_accel_channels + j, 0 ) = dat( j, 0 );
        }
    }

    return past_vec;
}

void PredictiveController::push_accelerometer_sample( const Matrix &new_acceleration )
{
    if( _num_accel_channels <= 0 )
    {
        return;
    }

    if( new_acceleration.rows() != _num_accel_channels || new_acceleration.cols() != 1 )
    {
        return;
    }

    accel_buffer.row( accel_head & ( buffer_size - 1 ) ) = new_acceleration.transpose();
    accel_head++;
}

Matrix PredictiveController::get_current_measurement_past( int num_steps )
{
    Matrix past_vec;
    past_vec.resize( num_steps * _num_modes, 1 );

    for( int i = 0; i < num_steps; i++ )
    {
        auto dat = measurement_buffer.row( ( measurement_head - i - 1 ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_modes; j++ )
        {
            past_vec( i * _num_modes + j ) = dat( j, 0 );
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_current_command_past( int num_steps )
{
    Matrix past_vec;
    past_vec.resize( num_steps * _num_modes, 1 );

    for( int i = 0; i < num_steps; i++ )
    {
        auto dat = command_buffer.row( ( command_head - i - 1 ) & ( buffer_size - 1 ) );
        for( int j = 0; j < _num_modes; j++ )
        {
            past_vec( i * _num_modes + j ) = dat( j, 0 );
        }
    }

    return past_vec;
}

void PredictiveController::update_system()
{
    Matrix future_cmd         = get_command_future( 1 );
    Matrix past_cmd           = get_command_past();
    Matrix past_accel         = get_accelerometer_past();
    Matrix future_measurement = get_measurement_future();
    Matrix past_measurement   = get_measurement_past();

    Matrix prediction_vector;
    prediction_vector.resize( future_cmd.rows() + past_cmd.rows() + past_accel.rows() + past_measurement.rows(), 1 );
    // Architectural note:
    // Feature order is [u_f, u_p, a_p, y_p], which is equivalent to the requested model partition
    // and keeps the legacy DDSPC block structure with u_f first.
    prediction_vector << future_cmd, past_cmd, past_accel, past_measurement;

    rls->update( &prediction_vector, &future_measurement );
}

void PredictiveController::update_controller()
{
    Matrix H   = rls->prediction_matrix.transpose() * rls->prediction_matrix;
    Matrix H11 = H.block( 0, 0, num_correlations, num_correlations );
    Matrix H21 = H.block( 0, num_correlations, num_correlations, H.cols() - num_correlations );

    if( do_switch_regularization_matrix )
    {
        if( use_regularization_matrix_01 )
        {
            regularization_matrix = &regularization_matrix_01;
        }
        else
        {
            regularization_matrix = &regularization_matrix_02;
        }

        do_switch_regularization_matrix = false;
    }

    Matrix full_controller = -1 * ( H11 + H11.maxCoeff() * ( *regularization_matrix ) ).inverse() * H21;
    controller =
        full_controller.block( full_controller.rows() - _num_modes, 0, _num_modes, full_controller.cols() );
}

Matrix PredictiveController::calculate_command( Matrix new_measurement, Matrix exploration_noise )
{
    measurement_buffer.row( measurement_head & ( buffer_size - 1 ) ) = new_measurement;
    measurement_head++;

    Matrix past_command     = get_current_command_past( _num_history - 1 );
    Matrix past_measurement = get_current_measurement_past( _num_history );
    Matrix past_accel       = get_accelerometer_past();

    Matrix past_vec;
    past_vec.resize( num_state_features, 1 );
    past_vec << past_command, past_accel, past_measurement;

    Matrix new_delta = ( controller + integrator ) * past_vec + exploration_noise;

    if( false )
    {
        for( int i = 0; i < _num_modes; i++ )
        {
            if( new_delta( i, 0 ) > _delta_max )
            {
                new_delta( i, 0 ) = _delta_max;
            }

            if( new_delta( i, 0 ) < -_delta_max )
            {
                new_delta( i, 0 ) = -_delta_max;
            }
        }
    }

    command_buffer.row( command_head & ( buffer_size - 1 ) ) = new_delta;
    command_head++;

    return new_delta;
}

}
