/** \file ddspc_shadow_test.cpp
 * \brief Deterministic independent-model, selection, readiness, and frozen-origin diagnostics tests.
 * \ingroup loPredCtrlAcc_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../shadow_controller.hpp"
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace libXWCTest
{
namespace loPredCtrlAccTest
{

/// \cond DDSPC_SHADOW_HELPERS
/// Use a small excited MIMO-independent-input problem without depending on any simulator.
DDSPC::ShadowConfig shadow_config()
{
    DDSPC::ShadowConfig config;
    config.m_modes = config.m_history = config.m_future = config.m_accelChannels = config.m_accelHistory = 1;
    config.m_gamma                                                                                       = 0.99f;
    config.m_prior                                                                                       = 0.1f;
    config.m_lambda                                                                                      = 0.4f;
    config.m_minUpdates                                                                                  = 8;
    return config;
}

/// Build an explicitly synthetic applied-input record with deterministic independent channels.
DDSPC::ShadowRecord shadow_record( std::uint64_t index /**< [in] Synthetic decision index. */ )
{
    std::mt19937        random( static_cast<unsigned>( index + 7123 ) );
    DDSPC::ShadowRecord record;
    record.m_index = index;
    record.m_residual.resize( 1, 1 );
    record.m_appliedCommand.resize( 1, 1 );
    record.m_acceleration.resize( 1, 1 );
    record.m_residual( 0 )       = ( int( random() % 2001 ) - 1000 ) / 1000.0f;
    record.m_appliedCommand( 0 ) = ( int( random() % 2001 ) - 1000 ) / 1000.0f;
    record.m_acceleration( 0 )   = ( int( random() % 2001 ) - 1000 ) / 1000.0f;
    record.m_wfsValid = record.m_appliedValid = record.m_timingValid = record.m_accelValid = true;
    return record;
}
/// \endcond

/// Independent training, reset, and pause must never mutate the other model's coefficients or flags.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Shadow DDSPC models are independently owned and trained", "[ddspc][shadow]" )
{
    auto settings = shadow_config();
    for( bool qrd : { false, true } )
    {
        settings.m_useQrd = qrd;
        DDSPC::ShadowController pair( settings );
        pair.training( DDSPC::ShadowMode::Standard, true );
        for( int i = 0; i < 30; ++i )
            REQUIRE( pair.process( shadow_record( i ) ) );
        REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
        REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_updates == 29 );
        REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
        REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ).isZero( 0 ) );
        const auto frozen = pair.prediction_matrix( DDSPC::ShadowMode::Standard );
        pair.training( DDSPC::ShadowMode::Standard, false );
        pair.training( DDSPC::ShadowMode::AccelAssisted, true );
        for( int i = 30; i < 60; ++i )
            REQUIRE( pair.process( shadow_record( i ) ) );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard ).isApprox( frozen, 0 ) );
        REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard ).cols() == 2 );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ).cols() == 3 );
        pair.reset( DDSPC::ShadowMode::AccelAssisted );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ).isZero( 0 ) );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard ).isApprox( frozen, 0 ) );
        REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_trainingEnabled );
        REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_trainingEnabled );
    }
}

/// A mode request selects diagnostics only and cannot start training, reset a model, or bypass readiness.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Shadow selection is readiness-gated and preserves both models", "[ddspc][shadow]" )
{
    DDSPC::ShadowController pair( shadow_config() );
    REQUIRE( pair.selected() == DDSPC::ShadowMode::Standard );
    REQUIRE( pair.select( DDSPC::ShadowMode::Standard ) ); // Idempotent boot selection does not arm anything.
    REQUIRE_FALSE( pair.select( DDSPC::ShadowMode::AccelAssisted ) );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_trainingEnabled );
    pair.training( DDSPC::ShadowMode::Standard, true );
    pair.training( DDSPC::ShadowMode::AccelAssisted, true );
    for( int i = 0; i < 40; ++i )
        REQUIRE( pair.process( shadow_record( i ) ) );
    const auto standard      = pair.prediction_matrix( DDSPC::ShadowMode::Standard );
    const auto assisted      = pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted );
    const auto standardCount = pair.status( DDSPC::ShadowMode::Standard ).m_updates;
    REQUIRE( pair.select( DDSPC::ShadowMode::AccelAssisted ) );
    REQUIRE( pair.selected() == DDSPC::ShadowMode::AccelAssisted );
    REQUIRE( pair.select( DDSPC::ShadowMode::Standard ) );
    REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard ).isApprox( standard, 0 ) );
    REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ).isApprox( assisted, 0 ) );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_updates == standardCount );
    REQUIRE_FALSE( pair.select( static_cast<DDSPC::ShadowMode>( 55 ) ) );
}

/// Compare standard-model adaptation exactly with a standalone Phase 1 WFS/input learner.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Managed standard DDSPC equals the corrected standalone baseline", "[ddspc][shadow][baseline]" )
{
    auto config = shadow_config();
    for( bool qrd : { false, true } )
    {
        config.m_useQrd = qrd;
        DDSPC::ShadowController     pair( config );
        DDSPC::PredictiveController baseline(
            1, 1, 1, 0, config.m_gamma, config.m_lambda, config.m_prior, 0, 0, DDSPC::HistoryMode::Causal );
        baseline.use_qrd = qrd;
        pair.training( DDSPC::ShadowMode::Standard, true );
        pair.training( DDSPC::ShadowMode::AccelAssisted, true );
        for( int i = 0; i < 90; ++i )
        {
            auto record = shadow_record( i );
            REQUIRE( pair.process( record ) );
            baseline.append_sample( i, record.m_residual, record.m_appliedCommand, DDSPC::Matrix() );
            baseline.update_system();
            REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard )
                         .isApprox( baseline.get_reference_prediction_matrix(), 0 ) );
        }
        REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_predictions > 10 );
        REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_predictions > 10 );
        REQUIRE( pair.proposal( DDSPC::ShadowMode::Standard )
                     .isApprox( baseline.get_reference_controller( config.m_lambda ) * baseline.get_prediction_state(),
                                1e-6f ) );
    }
}

/// Missing accelerometers affect only the assisted history; unknown applied inputs/timing stop both.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Shadow validity and gaps never fabricate applied inputs", "[ddspc][shadow]" )
{
    DDSPC::ShadowController pair( shadow_config() );
    pair.training( DDSPC::ShadowMode::Standard, true );
    pair.training( DDSPC::ShadowMode::AccelAssisted, true );
    for( int i = 0; i < 30; ++i )
        pair.process( shadow_record( i ) );
    REQUIRE( pair.select( DDSPC::ShadowMode::AccelAssisted ) );
    const auto assisted = pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted );
    const auto count    = pair.status( DDSPC::ShadowMode::Standard ).m_updates;
    auto       record   = shadow_record( 30 );
    record.m_accelValid = false;
    record.m_acceleration.resize( 0, 0 );
    REQUIRE( pair.process( record ) );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_updates == count + 1 );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ).isApprox( assisted, 0 ) );
    REQUIRE( pair.selected() == DDSPC::ShadowMode::AccelAssisted ); // No automatic selection or output action.
    REQUIRE( pair.select( DDSPC::ShadowMode::Standard ) );
    REQUIRE_FALSE( pair.select( DDSPC::ShadowMode::AccelAssisted ) );
    const auto standard   = pair.prediction_matrix( DDSPC::ShadowMode::Standard );
    record                = shadow_record( 31 );
    record.m_appliedValid = false;
    REQUIRE_FALSE( pair.process( record ) );
    REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard ).isApprox( standard, 0 ) );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_dataValid );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_samples == 0 );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_trainingEnabled );
    for( int i = 32; i < 55; ++i )
        REQUIRE( pair.process( shadow_record( i ) ) );
    REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    record               = shadow_record( 55 );
    record.m_timingValid = false;
    REQUIRE_FALSE( pair.process( record ) );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( pair.process( shadow_record( 56 ) ) );
    REQUIRE_FALSE( pair.process( shadow_record( 56 ) ) ); // Duplicate does not become another RLS observation.
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_samples == 0 );
    REQUIRE( pair.process( shadow_record( 60 ) ) ); // Forward gap starts a fresh local history.
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_samples == 1 );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( pair.validity().m_appliedCommand.size() == 0 );
}

/// History length and observation count alone cannot qualify zero/unexcited regressors as ready.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Shadow readiness requires training and data excitation", "[ddspc][shadow]" )
{
    auto config         = shadow_config();
    config.m_minUpdates = 20;
    DDSPC::ShadowController pair( config );
    for( int i = 0; i < 30; ++i )
        pair.process( shadow_record( i ) );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_reason == DDSPC::ShadowReason::Untrained );
    pair.training( DDSPC::ShadowMode::Standard, true );
    for( int i = 30; i < 49; ++i )
        pair.process( shadow_record( i ) );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_updates == 19 );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
    pair.process( shadow_record( 49 ) );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_ready );
    pair.reset( DDSPC::ShadowMode::Standard );
    for( int i = 50; i < 90; ++i )
    {
        auto record = shadow_record( i );
        record.m_residual.setZero();
        record.m_appliedCommand.setZero();
        pair.process( record );
    }
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_reason == DDSPC::ShadowReason::Unexcited );
    REQUIRE_FALSE( pair.status( DDSPC::ShadowMode::Standard ).m_predictionValid );
    REQUIRE( pair.proposal( DDSPC::ShadowMode::Standard ).size() == 0 );
}

/// Verify error diagnostics against independently frozen matrices/states over a two-step future horizon.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Shadow diagnostics freeze model parameters at the prediction origin", "[ddspc][shadow][prediction]" )
{
    auto config     = shadow_config();
    config.m_future = config.m_history = 2;
    DDSPC::ShadowController pair( config );
    pair.training( DDSPC::ShadowMode::Standard, true );
    pair.training( DDSPC::ShadowMode::AccelAssisted, true );
    std::vector<DDSPC::Matrix>       standard, assisted;
    std::vector<bool>                standardReady, assistedReady;
    std::vector<DDSPC::ShadowRecord> records;
    double                           standardError = 0, assistedError = 0;
    std::uint64_t                    standardValues = 0, assistedValues = 0;
    for( int t = 0; t < 100; ++t )
    {
        records.push_back( shadow_record( t ) );
        REQUIRE( pair.process( records.back() ) );
        if( t >= 2 )
        {
            const int     origin = t - 2;
            DDSPC::Matrix target( 2, 1 ), uf( 2, 1 );
            target << records[t].m_residual, records[t - 1].m_residual;
            uf << records[t - 1].m_appliedCommand, records[t - 2].m_appliedCommand;
            if( standardReady[origin] )
            {
                DDSPC::Matrix context( 5, 1 );
                context << records[origin].m_residual, records[origin - 1].m_residual,
                    records[origin - 1].m_appliedCommand, uf;
                standardError += ( standard[origin] * context - target ).cast<double>().squaredNorm();
                standardValues += 2;
            }
            if( assistedReady[origin] )
            {
                DDSPC::Matrix context( 6, 1 );
                context << records[origin].m_residual, records[origin - 1].m_residual,
                    records[origin - 1].m_appliedCommand, records[origin].m_acceleration, uf;
                assistedError += ( assisted[origin] * context - target ).cast<double>().squaredNorm();
                assistedValues += 2;
            }
        }
        const auto s = pair.status( DDSPC::ShadowMode::Standard ), a = pair.status( DDSPC::ShadowMode::AccelAssisted );
        standardReady.push_back( s.m_ready );
        assistedReady.push_back( a.m_ready );
        standard.push_back( pair.prediction_matrix( DDSPC::ShadowMode::Standard ) );
        assisted.push_back( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ) );
        if( standardValues )
            REQUIRE( s.m_rmse == Approx( std::sqrt( standardError / standardValues ) ).margin( 1e-7 ) );
        if( assistedValues )
            REQUIRE( a.m_rmse == Approx( std::sqrt( assistedError / assistedValues ) ).margin( 1e-7 ) );
    }
    REQUIRE( standardValues > 100 );
    REQUIRE( assistedValues > 100 );
    REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_predictions == standardValues / 2 );
    REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_predictions == assistedValues / 2 );
}

/** \brief Invalid numerical payloads never advance a learner, and missing acceleration preserves standard updates.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Shadow numerical validity checks precede both RLS updates", "[loPredCtrlAcc][shadow]" )
{
    for( bool qr : { false, true } )
    {
        auto config     = shadow_config();
        config.m_useQrd = qr;
        DDSPC::ShadowController pair( config );
        pair.training( DDSPC::ShadowMode::Standard, true );
        pair.training( DDSPC::ShadowMode::AccelAssisted, true );
        for( std::uint64_t i = 0; i < 40; ++i )
            REQUIRE( pair.process( shadow_record( i ) ) );
        auto record                = shadow_record( 40 );
        record.m_acceleration( 0 ) = std::numeric_limits<float>::quiet_NaN();
        const auto assisted        = pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted );
        const auto standardUpdates = pair.status( DDSPC::ShadowMode::Standard ).m_updates;
        REQUIRE( pair.process( record ) );
        REQUIRE( pair.status( DDSPC::ShadowMode::Standard ).m_updates == standardUpdates + 1 );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ) == assisted );
        REQUIRE_FALSE( pair.validity().m_accelValid );
        const auto standard = pair.prediction_matrix( DDSPC::ShadowMode::Standard );
        record              = shadow_record( 41 );
        record.m_appliedCommand.resize( 2, 1 );
        REQUIRE_FALSE( pair.process( record ) );
        REQUIRE_FALSE( pair.validity().m_appliedValid );
        record                 = shadow_record( 42 );
        record.m_residual( 0 ) = std::numeric_limits<float>::infinity();
        REQUIRE_FALSE( pair.process( record ) );
        REQUIRE_FALSE( pair.validity().m_wfsValid );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::Standard ) == standard );
        REQUIRE( pair.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ) == assisted );
        REQUIRE( pair.proposal( DDSPC::ShadowMode::Standard ).size() == 0 );
        REQUIRE( pair.proposal( DDSPC::ShadowMode::AccelAssisted ).size() == 0 );
    }
    for( float lambda : { 0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN() } )
    {
        auto config     = shadow_config();
        config.m_lambda = lambda;
        REQUIRE_THROWS_AS( DDSPC::ShadowController( config ), std::invalid_argument );
    }
}

} // namespace loPredCtrlAccTest
} // namespace libXWCTest
