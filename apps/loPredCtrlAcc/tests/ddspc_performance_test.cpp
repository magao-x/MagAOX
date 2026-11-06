/** \file ddspc_performance_test.cpp
 * \brief Independent pre-optimization replay and causal workspace/prediction safety regression tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "phase6_replay.hpp"
#include "../shadow_controller.hpp"
#include <iostream>
#include <limits>
#include <random>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
/// Construct portable independent synthetic WFS, applied-input and already synchronized sensor samples.
DDSPC::ShadowRecord performance_record( int           modes /**< [in] Controlled modes. */,
                                        std::uint64_t index /**< [in] External identity. */,
                                        unsigned      seed /**< [in] Deterministic exogenous signal identity. */ )
{
    DDSPC::ShadowRecord record;
    record.m_index = index;
    record.m_residual.resize( modes, 1 );
    record.m_appliedCommand.resize( modes, 1 );
    record.m_acceleration.resize( 2, 1 );
    record.m_wfsValid = record.m_appliedValid = record.m_timingValid = record.m_accelValid = true;
    std::mt19937 rng( seed );
    for( auto *v : { &record.m_residual, &record.m_appliedCommand, &record.m_acceleration } )
        for( int j = 0; j < v->size(); ++j )
            ( *v )( j ) = ( int( rng() % 2001 ) - 1000 ) / 1000.f;
    return record;
}
/// \endcond

/// Execute the committed Phase 6 code alongside optimized managers on identical interrupted causal sequences.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Optimized shadow traces preserve the independent Phase 6 baseline", "[ddspc][performance][baseline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process();
    DDSPC::PredictiveController::update_system();
#endif
    // clang-format on
    double maxProposal = 0, maxRmse = 0;
    for( int modes : { 2, 10 } )
        for( bool qrd : { false, true } )
            for( float gamma : { 1.f, .99f } )
            {
                DDSPC::ShadowConfig cfg;
                cfg.m_modes  = modes;
                cfg.m_useQrd = qrd;
                cfg.m_prior  = .1f;
                cfg.m_lambda = .3f;
                cfg.m_gamma  = gamma;
                DDSPC::ShadowController current( cfg );
                auto                    old = phase6_replay( modes, qrd, gamma );
                for( bool assisted : { false, true } )
                {
                    auto mode = assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
                    old->training( assisted, true );
                    current.training( mode, true );
                }
                for( int t = 0; t < 720; ++t )
                {
                    CAPTURE( modes, qrd, gamma, t );
                    const std::uint64_t index  = t + ( t >= 450 ? 9 : 0 );
                    auto                record = performance_record( modes, index, 8800 + t );
                    if( t >= 200 && t < 400 ) // Identical windows can reuse rank; renewed excitation must recompute it.
                        record = performance_record( modes, index, 800 );
                    record.m_accelValid  = t != 280;
                    record.m_timingValid = t != 400;
                    if( t == 300 || t == 304 )
                    {
                        old->training( true, t == 304 );
                        current.training( DDSPC::ShadowMode::AccelAssisted, t == 304 );
                    }
                    if( t == 680 || t == 684 )
                        for( bool assisted : { false, true } )
                        {
                            old->training( assisted, t == 684 );
                            current.training( assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard,
                                              t == 684 );
                        }
                    if( t == 520 )
                    {
                        old->reset( false );
                        current.reset( DDSPC::ShadowMode::Standard );
                    }
                    REQUIRE( current.process( record ) == old->process( index,
                                                                        record.m_residual,
                                                                        record.m_appliedCommand,
                                                                        record.m_acceleration,
                                                                        record.m_timingValid,
                                                                        record.m_accelValid ) );
                    for( bool assisted : { false, true } )
                    {
                        CAPTURE( assisted );
                        auto       mode     = assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
                        const auto expected = old->inspect( assisted );
                        const auto s        = current.status( mode );
                        const std::array<bool, 5> flags{
                            s.m_ready, s.m_trainingEnabled, s.m_training, s.m_dataValid, s.m_predictionValid };
                        const std::array<std::uint64_t, 3> counts{ s.m_samples, s.m_updates, s.m_predictions };
                        const std::array<int, 3> codes{ s.m_features, s.m_rank, static_cast<int>( s.m_reason ) };
                        REQUIRE( flags == expected.m_flags );
                        REQUIRE( counts == expected.m_counts );
                        REQUIRE( codes == expected.m_codes );
                        REQUIRE( current.prediction_matrix( mode ).isApprox( expected.m_theta, 0 ) );
                        const auto proposal = current.proposal( mode );
                        REQUIRE( proposal.rows() == expected.m_proposal.rows() );
                        REQUIRE( proposal.cols() == expected.m_proposal.cols() );
                        if( proposal.size() )
                        {
                            maxProposal = std::max(
                                maxProposal, double( ( proposal - expected.m_proposal ).cwiseAbs().maxCoeff() ) );
                            REQUIRE( proposal.isApprox( expected.m_proposal, 0 ) );
                        }
                        maxRmse = std::max( maxRmse, std::abs( s.m_rmse - expected.m_rmse ) );
                        REQUIRE( s.m_rmse == expected.m_rmse );
                        if( t == 250 || t == 719 )
                            REQUIRE( current.select( mode ) == old->select( assisted ) );
                    }
                }
            }
    std::cout << "Phase 6/7 matched traces: max proposal difference=" << maxProposal << " RMSE difference=" << maxRmse
              << '\n';
}

/// Reused pivoted QR must preserve the original normalized threshold on both sides of near-collinearity.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Rank workspace preserves readiness near the numerical threshold", "[ddspc][performance][rank]" )
{
    bool sawReady = false, sawUnexcited = false;
    for( bool qrd : { false, true } )
        for( float epsilon : { 0.f, 1e-7f, 1e-5f, 2e-5f, 1e-3f } )
        {
            DDSPC::ShadowConfig cfg;
            cfg.m_modes = cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
            cfg.m_useQrd                                                    = qrd;
            cfg.m_prior                                                     = .1f;
            cfg.m_lambda                                                    = .3f;
            DDSPC::ShadowController current( cfg );
            auto                    old = phase6_replay( 1, qrd, 1, 1, 1, 1 );
            current.training( DDSPC::ShadowMode::AccelAssisted, true );
            old->training( true, true );
            for( int i = 0; i < 160; ++i )
            {
                CAPTURE( qrd, epsilon, i );
                auto record                = performance_record( 1, i, 6100 + i );
                record.m_acceleration( 1 ) = record.m_acceleration( 0 ) + epsilon * record.m_acceleration( 1 );
                REQUIRE(
                    current.process( record ) ==
                    old->process( i, record.m_residual, record.m_appliedCommand, record.m_acceleration, true, true ) );
                const auto expected = old->inspect( true );
                const auto actual   = current.status( DDSPC::ShadowMode::AccelAssisted );
                REQUIRE( actual.m_rank == expected.m_codes[1] );
                REQUIRE( actual.m_ready == expected.m_flags[0] );
                REQUIRE( static_cast<int>( actual.m_reason ) == expected.m_codes[2] );
                sawReady |= actual.m_ready;
                sawUnexcited |= actual.m_reason == DDSPC::ShadowReason::Unexcited;
            }
        }
    REQUIRE( sawReady );
    REQUIRE( sawUnexcited );
}

/// Workspace identity, record freshness and reset epochs prevent duplicated or stale identification examples.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Causal workspace rejects stale records and reset epochs", "[ddspc][performance][history]" )
{
    DDSPC::PredictiveController controller( 1, 1, 1, 0, 1, .3f, .1f, 0, 0, DDSPC::HistoryMode::Causal );
    auto                        scratch = controller.make_workspace();
    DDSPC::Matrix               y = DDSPC::Matrix::Ones( 1, 1 ), u = .2f * y;
    controller.append_sample( 0, y, u, DDSPC::Matrix() );
    REQUIRE_FALSE( controller.prepare_training( *scratch ) );
    REQUIRE_THROWS_AS( controller.update_system( *scratch ), std::invalid_argument );
    controller.append_sample( 1, y, u, DDSPC::Matrix() );
    REQUIRE( controller.prepare_training( *scratch ) );
    REQUIRE( controller.update_system( *scratch ) );
    REQUIRE_FALSE( controller.update_system( *scratch ) );
    controller.append_sample( 2, y, u, DDSPC::Matrix() );
    REQUIRE_THROWS_AS( controller.update_system( *scratch ), std::invalid_argument );
    controller.reset_buffers();
    controller.append_sample( 0, 2 * y, u, DDSPC::Matrix() );
    controller.append_sample( 1, 2 * y, u, DDSPC::Matrix() );
    REQUIRE_THROWS_AS( controller.update_system( *scratch ), std::invalid_argument ); // Same head, different segment.
    REQUIRE( controller.prepare_training( *scratch ) );
    REQUIRE( controller.update_system( *scratch ) );
    DDSPC::PredictiveController legacy( 1, 1, 1, 0, 1, .3f, .1f, 0, 0 );
    REQUIRE_THROWS_AS( legacy.make_workspace(), std::logic_error );
    REQUIRE_THROWS_AS( legacy.prepare_training( *scratch ), std::invalid_argument );
}

/// Cached diagnostics retain readiness and explicit frame-age gates, and never adapt or publish a command.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Cached prediction is bounded by readiness and explicit coefficient age",
           "[ddspc][performance][prediction]" )
{
    DDSPC::ShadowConfig cfg;
    cfg.m_modes   = 2;
    cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
    cfg.m_prior                                       = .1f;
    cfg.m_lambda                                      = .3f;
    DDSPC::ShadowController pair( cfg );
    pair.training( DDSPC::ShadowMode::Standard, true );
    pair.training( DDSPC::ShadowMode::AccelAssisted, true );
    DDSPC::Matrix output = DDSPC::Matrix::Ones( 2, 1 ), state( 4, 1 );
    REQUIRE_FALSE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 0, 0, state, output ) );
    REQUIRE( output.isZero( 0 ) );
    auto last = performance_record( 2, 0, 9800 );
    for( int i = 0; i < 100; ++i )
    {
        last = performance_record( 2, i, 9800 + i );
        REQUIRE( pair.process( last ) );
    }
    state << last.m_residual, last.m_acceleration;
    REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    const auto status = pair.status( DDSPC::ShadowMode::AccelAssisted );
    REQUIRE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 99, 0, state, output ) );
    REQUIRE( output.isApprox( pair.proposal( DDSPC::ShadowMode::AccelAssisted ), 0 ) );
    REQUIRE_FALSE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 98, 1, state, output ) );
    REQUIRE_FALSE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 100, 0, state, output ) );
    REQUIRE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 100, 1, state, output ) );
    REQUIRE_FALSE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted,
                                        std::numeric_limits<std::uint64_t>::max(),
                                        std::numeric_limits<std::uint64_t>::max(),
                                        state,
                                        output ) );
    REQUIRE( pair.status( DDSPC::ShadowMode::AccelAssisted ).m_updates == status.m_updates );
    state( 0 ) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 99, 0, state, output ) );
    REQUIRE( output.isZero( 0 ) );
    auto gap = performance_record( 2, 110, 9990 );
    REQUIRE( pair.process( gap ) );
    state.setZero();
    REQUIRE_FALSE( pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 110, 0, state, output ) );
}
} // namespace libXWCTest::loPredCtrlAccTest
