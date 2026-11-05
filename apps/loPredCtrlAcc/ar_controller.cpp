/** \file ar_controller.cpp
 * \brief Legacy-compatible DDSPC and offline causal history/reference-control calculations.
 */

#include "ar_controller.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace DDSPC
{

Matrix reference_controller( const Matrix &A, const Matrix &B, const Matrix &D, const Matrix &C, realT lambda )
{
    if( C.rows() <= 0 || C.cols() <= 0 || A.rows() != C.rows() || B.rows() != C.rows() || D.rows() != C.rows() ||
        !A.allFinite() || !B.allFinite() || !D.allFinite() || !C.allFinite() || !std::isfinite( lambda ) ||
        lambda <= 0 )
    {
        throw std::invalid_argument( "Invalid DDSPC reference blocks or command penalty" );
    }
    Matrix state( C.rows(), A.cols() + B.cols() + D.cols() );
    state << A, B, D;
    Matrix hessian = C.transpose() * C;
    hessian.diagonal().array() += lambda;
    const Matrix rhs = -C.transpose() * state;
    if( !hessian.allFinite() || !rhs.allFinite() )
    {
        throw std::runtime_error( "DDSPC control products overflowed" );
    }
    Eigen::LDLT<Matrix> solver( hessian );
    if( solver.info() != Eigen::Success || !solver.isPositive() || solver.vectorD().minCoeff() <= 0 )
    {
        throw std::runtime_error( "DDSPC regularized control factorization failed" );
    }
    Matrix result = solver.solve( rhs );
    if( solver.info() != Eigen::Success || !result.allFinite() )
    {
        throw std::runtime_error( "DDSPC regularized control solve failed" );
    }
    return result;
}

PredictiveController::PredictiveController( int         num_actuators,
                                            int         num_history,
                                            int         num_future,
                                            realT       gain,
                                            realT       gamma,
                                            realT       initial_regularization,
                                            realT       initial_covariance,
                                            int         num_accel_channels,
                                            int         accel_history,
                                            HistoryMode history_mode )
{
    m_historyMode = history_mode;
    if( m_historyMode == HistoryMode::Causal &&
        ( num_actuators < 1 || num_history < 1 || num_future < 1 || num_accel_channels < 0 || accel_history < 0 ||
          !std::isfinite( gamma ) || gamma <= 0 || gamma > 1 || !std::isfinite( initial_covariance ) ||
          initial_covariance <= 0 ) )
    {
        throw std::invalid_argument( "Invalid causal DDSPC dimensions or RLS prior" );
    }
    if( m_historyMode == HistoryMode::Causal )
    {
        // Validate before the legacy integer dimension and power-of-two calculations below.
        const double features = ( double( num_future ) - 1 + 2 * double( num_history ) ) * num_actuators +
                                double( num_accel_channels ) * accel_history;
        const double slots    = 2 * ( double( num_history ) + num_future + accel_history + 1 );
        if( features > std::numeric_limits<int>::max() || slots >= ( 1 << 30 ) )
        {
            throw std::invalid_argument( "Causal DDSPC dimensions exceed the legacy storage index range" );
        }
    }
    // Settable properties
    m_numModes         = num_actuators;
    m_numHistory       = num_history;
    m_numFuture        = num_future;
    m_numAccelChannels = std::max( 0, num_accel_channels );
    m_numAccelHistory  = std::max( 0, accel_history );
    m_gain             = gain;
    m_deltaMax         = 0.5;
    m_regularization   = initial_regularization;

    // Derived properties
    m_numPredictors    = m_numFuture * m_numModes;
    m_numAccelFeatures = m_numAccelHistory * m_numAccelChannels;
    m_numFeatures      = ( m_numFuture - 1 + 2 * m_numHistory ) * m_numModes + m_numAccelFeatures;
    m_numStateFeatures = ( 2 * m_numHistory - 1 ) * m_numModes + m_numAccelFeatures;
    m_numCorrelations  = m_numFuture * m_numModes;

    // Data buffers
    m_bufferSize = find_next_power_of_2( 2 * ( num_history + num_future + m_numAccelHistory + 1 ) );

    m_measurementHead = 0;
    m_measurementBuffer.resize( m_bufferSize, m_numModes );
    m_measurementBuffer.setZero();

    m_commandHead = 0;
    m_commandBuffer.resize( m_bufferSize, m_numModes );
    m_commandBuffer.setZero();

    m_accelHead = 0;
    m_accelBuffer.resize( m_bufferSize, m_numAccelChannels );
    m_accelBuffer.setZero();

    // The learner
    m_rls    = new RecursiveLeastSquares( m_numPredictors, m_numFeatures, gamma, initial_covariance );
    m_qrdRls = new QRDRecursiveLeastSquares(
        m_numPredictors, m_numFeatures, gamma, initial_covariance, m_historyMode == HistoryMode::Causal );

    // Initialize the legacy controller and its separate added-feedback gain.
    m_controller.resize( m_numModes, m_numStateFeatures );
    m_controller.setZero();

    m_integrator.resize( m_numModes, m_numStateFeatures );
    m_integrator.setZero();

    for( int i = 0; i < m_numModes; i++ )
    {
        int index                = ( m_numHistory - 1 ) * m_numModes + m_numAccelFeatures + i;
        m_integrator( i, index ) = m_gain;
    }

    // Set the regularization matrix
    m_regularizationMatrix01.resize( m_numCorrelations, m_numCorrelations );
    m_regularizationMatrix01.setZero();

    m_regularizationMatrix02.resize( m_numCorrelations, m_numCorrelations );
    m_regularizationMatrix02.setZero();

    for( int i = 0; i < m_numCorrelations; i++ )
    {
        m_regularizationMatrix01( i, i ) = initial_regularization;
        m_regularizationMatrix02( i, i ) = initial_regularization;
    }

    m_regularizationMatrix = &m_regularizationMatrix01;
};

PredictiveController::~PredictiveController()
{
    delete m_rls;
    delete m_qrdRls;
}

void PredictiveController::reset()
{
    m_controller.resize( m_numModes, m_numStateFeatures );
    m_controller.setZero();
    m_integrator.resize( m_numModes, m_numStateFeatures );
    m_integrator.setZero();

    for( int i = 0; i < m_numModes; i++ )
    {
        int index                = ( m_numHistory - 1 ) * m_numModes + m_numAccelFeatures + i;
        m_integrator( i, index ) = m_gain;
    }

    m_rls->reset();
    m_qrdRls->reset();

    reset_buffers();
}

void PredictiveController::reset_buffers()
{
    m_lastTrainingHead = 0;
    m_measurementHead  = 0;
    m_measurementBuffer.resize( m_bufferSize, m_numModes );
    m_measurementBuffer.setZero();

    m_commandHead = 0;
    m_commandBuffer.resize( m_bufferSize, m_numModes );
    m_commandBuffer.setZero();

    m_accelHead = 0;
    m_accelBuffer.resize( m_bufferSize, m_numAccelChannels );
    m_accelBuffer.setZero();
}

void PredictiveController::set_regularization( realT new_regularization )
{
    if( m_useRegularizationMatrix01 )
    {
        for( int i = 0; i < m_numCorrelations; i++ )
        {
            m_regularizationMatrix02( i, i ) = new_regularization;
        }
        m_useRegularizationMatrix01 = false;
    }
    else
    {
        for( int i = 0; i < m_numCorrelations; i++ )
        {
            m_regularizationMatrix01( i, i ) = new_regularization;
        }
        m_useRegularizationMatrix01 = true;
    }
    m_doSwitchRegularizationMatrix = true;
}

Matrix PredictiveController::get_measurement_future()
{
    Matrix future_vec;
    future_vec.resize( m_numFuture * m_numModes, 1 );

    for( int i = 0; i < m_numFuture; i++ )
    {
        auto dat = m_measurementBuffer.row( ( m_measurementHead - i - 1 ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numModes; j++ )
        {
            future_vec( i * m_numModes + j, 0 ) = dat( 0, j );
        }
    }

    return future_vec;
}

Matrix PredictiveController::get_measurement_past()
{
    Matrix past_vec;
    past_vec.resize( m_numHistory * m_numModes, 1 );

    for( int i = 0; i < m_numHistory; i++ )
    {
        auto dat = m_measurementBuffer.row( ( m_measurementHead - i - m_numFuture - 1 ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numModes; j++ )
        {
            past_vec( i * m_numModes + j, 0 ) = dat( 0, j );
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_command_future( int skip_cmds )
{
    if( m_historyMode == HistoryMode::Causal && ( skip_cmds < 0 || skip_cmds > m_numFuture ) )
    {
        throw std::invalid_argument( "Invalid future-command skip" );
    }
    Matrix future_vec( ( m_numFuture - skip_cmds ) * m_numModes, 1 );
    for( int i = 0; i < ( m_numFuture - skip_cmds ); i++ )
    {
        const int offset = m_historyMode == HistoryMode::Causal ? skip_cmds : skip_cmds * m_numModes;
        auto      dat    = m_commandBuffer.row( ( m_commandHead - i - 1 - offset ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numModes; j++ )
        {
            future_vec( i * m_numModes + j, 0 ) = dat( 0, j );
        }
    }

    return future_vec;
}

Matrix PredictiveController::get_command_past()
{
    Matrix past_vec;
    past_vec.resize( m_numHistory * m_numModes, 1 );

    for( int i = 0; i < m_numHistory; i++ )
    {
        auto dat = m_commandBuffer.row( ( m_commandHead - i - m_numFuture - 1 ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numModes; j++ )
        {
            past_vec( i * m_numModes + j ) = dat( 0, j );
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_accelerometer_past( int delay )
{
    if( delay < 0 || static_cast<std::uint64_t>( delay ) + m_numAccelHistory > m_bufferSize )
    {
        throw std::invalid_argument( "Invalid accelerometer history offset" );
    }
    Matrix past_vec;
    past_vec.resize( m_numAccelFeatures, 1 );
    past_vec.setZero();

    if( m_numAccelFeatures == 0 )
    {
        return past_vec;
    }

    for( int i = 0; i < m_numAccelHistory; i++ )
    {
        auto dat = m_accelBuffer.row( ( m_accelHead - delay - i - 1 ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numAccelChannels; j++ )
        {
            past_vec( i * m_numAccelChannels + j, 0 ) = dat( 0, j );
        }
    }

    return past_vec;
}

void PredictiveController::push_accelerometer_sample( const Matrix &new_acceleration )
{
    if( m_historyMode == HistoryMode::Causal )
    {
        throw std::logic_error( "Causal histories require append_sample records" );
    }
    if( m_numAccelChannels <= 0 )
    {
        return;
    }

    if( new_acceleration.rows() != m_numAccelChannels || new_acceleration.cols() != 1 )
    {
        return;
    }

    for( int j = 0; j < m_numAccelChannels; j++ )
    {
        m_accelBuffer( ( m_accelHead & ( m_bufferSize - 1 ) ), j ) = new_acceleration( j, 0 );
    }
    m_accelHead++;
}

Matrix PredictiveController::get_current_measurement_past( int num_steps )
{
    Matrix past_vec;
    past_vec.resize( num_steps * m_numModes, 1 );

    for( int i = 0; i < num_steps; i++ )
    {
        auto dat = m_measurementBuffer.row( ( m_measurementHead - i - 1 ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numModes; j++ )
        {
            past_vec( i * m_numModes + j ) = dat( 0, j );
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_current_command_past( int num_steps )
{
    Matrix past_vec;
    past_vec.resize( num_steps * m_numModes, 1 );

    for( int i = 0; i < num_steps; i++ )
    {
        auto dat = m_commandBuffer.row( ( m_commandHead - i - 1 ) & ( m_bufferSize - 1 ) );
        for( int j = 0; j < m_numModes; j++ )
        {
            past_vec( i * m_numModes + j ) = dat( 0, j );
        }
    }

    return past_vec;
}

void PredictiveController::append_sample( std::uint64_t index,
                                          const Matrix &measurement,
                                          const Matrix &applied_command,
                                          const Matrix &acceleration )
{
    if( m_historyMode != HistoryMode::Causal )
    {
        throw std::logic_error( "append_sample requires Causal history mode" );
    }
    if( index != m_measurementHead || measurement.rows() != m_numModes || measurement.cols() != 1 ||
        applied_command.rows() != m_numModes || applied_command.cols() != 1 || !measurement.allFinite() ||
        !applied_command.allFinite() || !acceleration.allFinite() ||
        ( m_numAccelFeatures > 0 && ( acceleration.rows() != m_numAccelChannels || acceleration.cols() != 1 ) ) ||
        ( m_numAccelFeatures == 0 && acceleration.size() != 0 ) )
    {
        throw std::invalid_argument( "Causal DDSPC requires finite contiguous synchronized records" );
    }
    const auto row                 = index & ( m_bufferSize - 1 );
    m_measurementBuffer.row( row ) = measurement.transpose();
    m_commandBuffer.row( row )     = applied_command.transpose();
    if( m_numAccelFeatures > 0 )
    {
        m_accelBuffer.row( row ) = acceleration.transpose();
    }
    ++m_measurementHead;
    ++m_commandHead;
    ++m_accelHead;
}

bool PredictiveController::get_training_example( Matrix &regressor, Matrix &target )
{
    const int accel_history = m_numAccelFeatures > 0 ? m_numAccelHistory : 0;
    if( m_historyMode == HistoryMode::Causal &&
        m_measurementHead < static_cast<std::uint64_t>( m_numFuture + std::max( m_numHistory, accel_history ) ) )
    {
        return false;
    }
    Matrix future_cmd       = get_command_future( 1 );
    Matrix past_cmd         = get_command_past();
    Matrix past_accel       = get_accelerometer_past( m_historyMode == HistoryMode::Causal ? m_numFuture : 0 );
    Matrix past_measurement = get_measurement_past();

    regressor.resize( m_numFeatures, 1 );
    regressor << future_cmd, past_cmd, past_accel, past_measurement;
    target = get_measurement_future();
    return true;
}

Matrix PredictiveController::get_prediction_state()
{
    const int accel_history = m_numAccelFeatures > 0 ? m_numAccelHistory : 0;
    if( m_historyMode != HistoryMode::Causal ||
        m_measurementHead < static_cast<std::uint64_t>( std::max( m_numHistory, accel_history ) ) )
    {
        throw std::logic_error( "Causal prediction state is not ready" );
    }
    Matrix state( m_numStateFeatures, 1 );
    state.topRows( m_numHistory * m_numModes ) = get_current_measurement_past( m_numHistory );
    // The command in the latest record is u_i, so known command history starts at u_(i-1).
    const int command_offset = m_numHistory * m_numModes;
    for( int i = 0; i < m_numHistory - 1; ++i )
    {
        state.middleRows( command_offset + i * m_numModes, m_numModes ) =
            m_commandBuffer.row( ( m_commandHead - i - 2 ) & ( m_bufferSize - 1 ) ).transpose();
    }
    state.bottomRows( m_numAccelFeatures ) = get_accelerometer_past();
    return state;
}

Matrix PredictiveController::get_prediction_matrix()
{
    return use_qrd ? m_qrdRls->prediction_matrix : m_rls->prediction_matrix;
}

Matrix PredictiveController::get_reference_prediction_matrix()
{
    Matrix    internal = get_prediction_matrix();
    Matrix    result( m_numPredictors, m_numFeatures );
    const int a_cols = m_numHistory * m_numModes;
    const int b_cols = ( m_numHistory - 1 ) * m_numModes;
    result << internal.rightCols( a_cols ), internal.middleCols( m_numCorrelations, b_cols ),
        internal.middleCols( m_numCorrelations + b_cols, m_numAccelFeatures ), internal.leftCols( m_numCorrelations );
    return result;
}

Matrix PredictiveController::get_reference_controller( realT lambda )
{
    Matrix    model  = get_reference_prediction_matrix();
    const int a_cols = m_numHistory * m_numModes;
    const int b_cols = ( m_numHistory - 1 ) * m_numModes;
    return reference_controller( model.leftCols( a_cols ),
                                 model.middleCols( a_cols, b_cols ),
                                 model.middleCols( a_cols + b_cols, m_numAccelFeatures ),
                                 model.rightCols( m_numCorrelations ),
                                 lambda );
}

void PredictiveController::update_system()
{
    if( m_historyMode == HistoryMode::Causal && m_lastTrainingHead == m_measurementHead )
    {
        return;
    }

    Matrix prediction_vector;
    Matrix future_measurement;
    if( !get_training_example( prediction_vector, future_measurement ) )
    {
        return;
    }

    if( use_qrd )
    {
        m_qrdRls->update( &prediction_vector, &future_measurement );
    }
    else
    {
        m_rls->update( &prediction_vector, &future_measurement );
    }
    m_lastTrainingHead = m_measurementHead;
}

void PredictiveController::update_controller()
{
    if( m_historyMode == HistoryMode::Causal )
    {
        throw std::logic_error( "Causal offline control uses get_reference_controller" );
    }
    Matrix H   = get_prediction_matrix().transpose() * get_prediction_matrix();
    Matrix H11 = H.block( 0, 0, m_numCorrelations, m_numCorrelations );
    Matrix H21 = H.block( 0, m_numCorrelations, m_numCorrelations, H.cols() - m_numCorrelations );

    if( m_doSwitchRegularizationMatrix )
    {
        if( m_useRegularizationMatrix01 )
        {
            m_regularizationMatrix = &m_regularizationMatrix01;
        }
        else
        {
            m_regularizationMatrix = &m_regularizationMatrix02;
        }

        m_doSwitchRegularizationMatrix = false;
    }

    Matrix full_controller = -1 * ( H11 + H11.maxCoeff() * ( *m_regularizationMatrix ) ).inverse() * H21;
    m_controller = full_controller.block( full_controller.rows() - m_numModes, 0, m_numModes, full_controller.cols() );
}

Matrix PredictiveController::calculate_command( Matrix new_measurement, Matrix exploration_noise )
{
    if( m_historyMode == HistoryMode::Causal )
    {
        throw std::logic_error( "Causal histories require explicit applied commands via append_sample" );
    }
    for( int j = 0; j < m_numModes; j++ )
    {
        m_measurementBuffer( ( m_measurementHead & ( m_bufferSize - 1 ) ), j ) = new_measurement( j, 0 );
    }
    m_measurementHead++;

    Matrix past_command     = get_current_command_past( m_numHistory - 1 );
    Matrix past_accel       = get_accelerometer_past();
    Matrix past_measurement = get_current_measurement_past( m_numHistory );

    Matrix past_vec;
    past_vec.resize( m_numStateFeatures, 1 );
    past_vec << past_command, past_accel, past_measurement;

    Matrix new_delta = ( m_controller + m_integrator ) * past_vec + exploration_noise;

    if( false )
    {
        for( int i = 0; i < m_numModes; i++ )
        {
            if( new_delta( i, 0 ) > m_deltaMax )
                new_delta( i, 0 ) = m_deltaMax;

            if( new_delta( i, 0 ) < -m_deltaMax )
                new_delta( i, 0 ) = -m_deltaMax;
        }
    }

    for( int j = 0; j < m_numModes; j++ )
    {
        m_commandBuffer( ( m_commandHead & ( m_bufferSize - 1 ) ), j ) = new_delta( j, 0 );
    }
    m_commandHead++;

    return new_delta;
}

void PredictiveController::save_state( const std::string &filename )
{
    if( m_historyMode == HistoryMode::Causal )
    {
        throw std::logic_error( "Causal model persistence is not implemented" );
    }
    // Save metadata in a JSON sidecar file
    std::string   metadata_file = filename + ".meta";
    std::ofstream ofs( metadata_file );
    if( !ofs.is_open() )
    {
        throw std::runtime_error( "Could not open file for save_state metadata: " + metadata_file );
    }

    ofs << "{\n";
    ofs << "  \"num_modes\": " << m_numModes << ",\n";
    ofs << "  \"num_future\": " << m_numFuture << ",\n";
    ofs << "  \"num_history\": " << m_numHistory << ",\n";
    ofs << "  \"gain\": " << m_gain << ",\n";
    ofs << "  \"delta_max\": " << m_deltaMax << ",\n";
    ofs << "  \"regularization\": " << m_regularization << ",\n";
    ofs << "}\n";
    ofs.close();

    // Save matrices using utils helpers
    DDSPC::save_matrix( filename + ".controller", m_controller );
    m_rls->save_state( filename + ".rls" );
    m_qrdRls->save_state( filename + ".qrd_rls" );
}

void PredictiveController::load_state( const std::string &filename )
{
    if( m_historyMode == HistoryMode::Causal )
    {
        throw std::logic_error( "Causal model persistence is not implemented" );
    }
    std::string   metadata_file = filename + ".meta";
    std::ifstream ifs( metadata_file );
    if( !ifs.is_open() )
    {
        throw std::runtime_error( "Could not open file for load_state metadata: " + metadata_file );
    }

    std::string line;
    std::getline( ifs, line ); // {

    std::getline( ifs, line );
    m_numModes = std::stoi( DDSPC::parse_json_value( line ) );
    std::getline( ifs, line );
    m_numFuture = std::stoi( DDSPC::parse_json_value( line ) );
    std::getline( ifs, line );
    m_numHistory = std::stoi( DDSPC::parse_json_value( line ) );
    std::getline( ifs, line );
    m_gain = static_cast<realT>( std::stod( DDSPC::parse_json_value( line ) ) );
    std::getline( ifs, line );
    m_deltaMax = static_cast<realT>( std::stod( DDSPC::parse_json_value( line ) ) );
    std::getline( ifs, line );
    m_regularization = static_cast<realT>( std::stod( DDSPC::parse_json_value( line ) ) );

    ifs.close();

    m_controller = DDSPC::load_matrix( filename + ".controller" );
    set_regularization( m_regularization );
    m_rls->load_state( filename + ".rls" );
    m_qrdRls->load_state( filename + ".qrd_rls" );
    reset_buffers();
}

} // namespace DDSPC
