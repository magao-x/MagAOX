/** \file ddspc_stability_test.cpp
 * \brief Pinned Phase 10 equivalence, weighted-LS covariance checks and causal offline recovery validation.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "stability.hpp"
#include <Eigen/Eigenvalues>
#include <cstring>
#include <tuple>
#undef PCRLS_HPP
#undef PCUTILS_HPP
#define DDSPC DDSPCSymmetric
#include "symmetric/recursive_least_squares.hpp"
#undef DDSPC

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
/// Compare every float bit on unchanged paths, including signed zero.
bool stability_bits( const DDSPC::Matrix &a /**< [in] Actual. */, const DDSPC::Matrix &b /**< [in] Reference. */ )
{
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           !std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) );
}
/// Compare original Phase 10 fields without inventing values for added recovery metadata.
void same_frame( const VibrationFrame &a /**< [in] Current. */, const VibrationFrame &b /**< [in] Pinned reference. */ )
{
    REQUIRE( stability_bits( a.m_residual, b.m_residual ) );
    REQUIRE( stability_bits( a.m_measured, b.m_measured ) );
    REQUIRE( stability_bits( a.m_requested, b.m_requested ) );
    REQUIRE( stability_bits( a.m_applied, b.m_applied ) );
    REQUIRE( std::tie( a.m_valid,
                       a.m_ready,
                       a.m_newCommand,
                       a.m_version,
                       a.m_ageNs,
                       a.m_modelSequence,
                       a.m_origin,
                       a.m_reason,
                       a.m_predictionSquared,
                       a.m_predictionCount,
                       a.m_thetaNorm,
                       a.m_dNorm,
                       a.m_rank ) == std::tie( b.m_valid,
                                               b.m_ready,
                                               b.m_newCommand,
                                               b.m_version,
                                               b.m_ageNs,
                                               b.m_modelSequence,
                                               b.m_origin,
                                               b.m_reason,
                                               b.m_predictionSquared,
                                               b.m_predictionCount,
                                               b.m_thetaNorm,
                                               b.m_dNorm,
                                               b.m_rank ) );
}
/// Verify independent recovery provenance and strict amplitude/rate limits on every action.
void recovery_safety( const VibrationConfig &c /**< [in] Explicit simulation contract. */,
                      const VibrationResult &result /**< [in] Actual trajectories. */ )
{
    for( int model = 3; model < 5; ++model )
    {
        int probes = 0;
        REQUIRE( result.m_invalidChanges[model] == 0 );
        for( int i = c.m_train; i < c.m_train + c.m_evaluate; ++i )
        {
            const auto &f        = result.m_frames[model][i];
            const auto &previous = result.m_frames[model][i - 1];
            REQUIRE( f.m_requested.allFinite() );
            REQUIRE( f.m_applied.allFinite() );
            REQUIRE( f.m_requested.cwiseAbs().maxCoeff() <= c.m_limit );
            REQUIRE( f.m_applied.cwiseAbs().maxCoeff() <= c.m_limit );
            if( f.m_action == int( SimAction::Model ) )
            {
                REQUIRE( f.m_valid );
                REQUIRE( f.m_ageNs <= c.m_maxAgeNs );
                REQUIRE( f.m_modelSequence < std::uint64_t( f.m_origin ) );
                REQUIRE( f.m_correctionAccepted );
            }
            else
                REQUIRE_FALSE( f.m_correctionAccepted );
            if( f.m_action == int( SimAction::Hold ) )
                REQUIRE( stability_bits( f.m_requested, previous.m_requested ) );
            if( f.m_action == int( SimAction::Neutral ) )
                for( int j = 0; j < 2; ++j )
                    REQUIRE( std::abs( f.m_requested( j ) ) <= std::abs( previous.m_requested( j ) ) );
            if( c.m_controlGain != 0 )
                for( int j = 0; j < 2; ++j )
                    REQUIRE( std::abs( double( f.m_requested( j ) ) - previous.m_requested( j ) ) <=
                             ( j ? f.m_slewY : f.m_slewX ) );
            probes += f.m_action == int( SimAction::Probe );
        }
        REQUIRE( probes <= c.m_train );
    }
}
/// \endcond

/// A real-data regressor replay separates covariance roundoff from feedback and matches independent weighted LS.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Symmetric covariance candidate restores the RLS invariant on identical calibration data",
           "[ddspc][stability]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::RecursiveLeastSquares::update_preallocated();
#endif
    // clang-format on
    VibrationConfig c;
    c.m_train                          = 2400; // Exactly the Phase 10 calibration; early covariance checks remain.
    c.m_evaluate                       = 100;
    c.m_broadband                      = true;
    const auto                  source = vibration_source( c );
    const auto                  trace  = committed_vibration_run( c, source );
    DDSPC::PredictiveController history(
        2, 5, 3, 0, float( c.m_gamma ), float( c.m_lambda ), float( c.m_prior ), 2, 5, DDSPC::HistoryMode::Causal );
    auto                                  workspace = history.make_workspace();
    DDSPC::RecursiveLeastSquares          original( 6, 34, float( c.m_gamma ), float( c.m_prior ) );
    DDSPCSymmetric::RecursiveLeastSquares candidate( 6, 34, float( c.m_gamma ), float( c.m_prior ) );
    DDSPC::Matrix                         normal( 1, 1 ), weights( 6, 34 ), covariance( 34, 34 );
    Eigen::MatrixXd                       info = double( float( c.m_prior ) ) * Eigen::MatrixXd::Identity( 34, 34 ),
                                          rhs  = Eigen::MatrixXd::Zero( 6, 34 );
    bool                                  originalIndefinite = false;
    for( int i = 0; i < c.m_train; ++i )
    {
        const auto &r = trace.m_frames[4][i];
        history.append_sample( i, r.m_measured, r.m_applied, source.m_acceleration[source.m_sensorIndex[i]] );
        if( !history.prepare_training( *workspace ) )
            continue;
        const auto &x = workspace->regressor(), &y = workspace->target();
        original.update_preallocated( x, y, normal, weights, covariance );
        candidate.update_preallocated( x, y, normal, weights, covariance );
        info = double( float( c.m_gamma ) ) * info + x.cast<double>() * x.cast<double>().transpose();
        rhs  = double( float( c.m_gamma ) ) * rhs + y.cast<double>() * x.cast<double>().transpose();
        REQUIRE( stability_bits( candidate.inverse_covariance, candidate.inverse_covariance.transpose().eval() ) );
        if( i % 20 == 0 )
        {
            const Eigen::MatrixXd                          po = original.inverse_covariance.cast<double>();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig( ( .5 * ( po + po.transpose() ) ).eval() );
            originalIndefinite = originalIndefinite || eig.eigenvalues().minCoeff() < 0;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> good( candidate.inverse_covariance.cast<double>() );
            REQUIRE( good.eigenvalues().minCoeff() > 0 );
        }
    }
    const Eigen::MatrixXd exact = rhs * info.ldlt().solve( Eigen::MatrixXd::Identity( 34, 34 ) );
    REQUIRE( originalIndefinite );
    REQUIRE( ( candidate.prediction_matrix.cast<double>() - exact ).norm() <
             .01 * ( original.prediction_matrix.cast<double>() - exact ).norm() );
}

/// The compatibility path is the actual committed implementation, with exact old payloads and diagnostics.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Phase 11 preserves the committed Phase 10 default and QR baseline exactly", "[ddspc][stability]" )
{
    VibrationConfig c;
    c.m_train         = 200;
    c.m_evaluate      = 120;
    c.m_broadband     = true;
    c.m_qrd           = GENERATE( false, true );
    const auto      s = vibration_source( c ), beforeSource = s;
    const auto      old = committed_vibration_run( c, s ), now = vibration_run( c, s );
    StabilityPolicy symmetric;
    symmetric.m_symmetricRls = true;
    c.m_stability            = &symmetric;
    const auto candidate     = vibration_run( c, s );
    for( int model = 0; model < 5; ++model )
        for( int i = 0; i < c.m_train + c.m_evaluate; ++i )
        {
            same_frame( now.m_frames[model][i], old.m_frames[model][i] );
            if( c.m_qrd || model < 3 )
                same_frame( candidate.m_frames[model][i], old.m_frames[model][i] );
        }
}

/// Explicit recovery actions require calibrated authority and never derive a command from invalid or future evidence.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline recovery rejects stale models and handles reset dropout overflow and lost excitation",
           "[ddspc][stability]" )
{
    const bool qrd = GENERATE( false, true );
    for( int fault = 0; fault < 8; ++fault )
    {
        CAPTURE( qrd, fault );
        VibrationConfig c;
        c.m_train       = 600;
        c.m_evaluate    = 400;
        c.m_broadband   = true;
        c.m_qrd         = qrd;
        const int event = c.m_train + 80;
        if( fault == 0 )
            c.m_resetFrame = event;
        if( fault == 1 )
        {
            c.m_missingFrame  = event;
            c.m_missingFrames = 20;
        }
        if( fault == 2 )
        {
            c.m_stallFrame  = event;
            c.m_stallFrames = 32;
            c.m_capacity    = 4;
        }
        if( fault == 3 )
            c.m_publicationFrames = 6;
        if( fault == 4 )
            c.m_limit = .2;
        if( fault == 5 )
        {
            c.m_freezeFrame  = event;
            c.m_freezeFrames = 100;
        }
        if( fault == 6 )
            c.m_controlGain = 0;
        StabilityPolicy policy;
        policy.m_symmetricRls  = !qrd;
        policy.m_neutralReturn = true;
        policy.m_retrain       = true;
        policy.m_confidence    = true;
        c.m_stability          = &policy;
        const auto result      = vibration_run( c, vibration_source( c ) );
        recovery_safety( c, result );
        if( fault == 3 || fault == 6 )
            for( int model = 3; model < 5; ++model )
                for( int i = c.m_train + 20; i < c.m_train + c.m_evaluate; ++i )
                {
                    REQUIRE( result.m_frames[model][i].m_requested.isZero( 0 ) );
                    REQUIRE( result.m_frames[model][i].m_action != int( SimAction::Probe ) );
                }
        if( fault < 3 || fault == 5 )
            for( int model = 3; model < 5; ++model )
            {
                int consecutive = 0, best = 0;
                for( int i = event + 100; i < c.m_train + c.m_evaluate; ++i )
                {
                    consecutive = result.m_frames[model][i].m_correctionAccepted ? consecutive + 1 : 0;
                    best        = std::max( best, consecutive );
                }
                REQUIRE( best >= 24 );
            }
    }
}

/// Changing future disturbance/sensor values cannot change earlier corrections or independent retraining actions.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Recovery and symmetric RLS preserve future-prefix invariance", "[ddspc][stability]" )
{
    StabilityPolicy policy;
    policy.m_symmetricRls  = true;
    policy.m_neutralReturn = true;
    policy.m_retrain       = true;
    policy.m_confidence    = true;
    VibrationConfig c;
    c.m_train         = 600;
    c.m_evaluate      = 400;
    c.m_broadband     = true;
    c.m_resetFrame    = 650;
    c.m_stability     = &policy;
    auto       s      = vibration_source( c );
    const auto a      = vibration_run( c, s );
    const auto repeat = vibration_run( c, s );
    for( std::size_t i = 800; i < s.m_disturbance.size(); ++i )
    {
        s.m_disturbance[i].array() += 10;
        s.m_acceleration[i].array() -= 20;
        s.m_exploration[i].array() += 5;
    }
    const auto b = vibration_run( c, s );
    for( int model : { 2, 3, 4 } )
        for( int i = 0; i < 1000; ++i )
        {
            same_frame( a.m_frames[model][i], repeat.m_frames[model][i] );
            if( i < 800 )
            {
                REQUIRE( stability_bits( a.m_frames[model][i].m_requested, b.m_frames[model][i].m_requested ) );
                REQUIRE( a.m_frames[model][i].m_action == b.m_frames[model][i].m_action );
            }
        }
}

/// A zero empirical noise bound remains valid calibration evidence, with finite retraining budget.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Recovery confidence handles exact zero errors without weakening its gate", "[ddspc][stability]" )
{
    VibrationConfig c;
    StabilityPolicy p;
    p.m_neutralReturn = p.m_retrain = p.m_confidence = true;
    RecoverySupervisor supervisor( c, 4, p );
    DDSPC::Matrix      positive = DDSPC::Matrix::Ones( 2, 1 ), zero = positive * 0;
    for( int i = 0; i < 8; ++i )
    {
        supervisor.calibrate( i % 2 ? positive : -positive );
        supervisor.observe( 0, true );
    }
    SimAction     action;
    DDSPC::Matrix command = positive;
    for( int i = 0; i < 30; ++i )
        command = supervisor.command( false, false, positive, command, positive, 4, action );
    REQUIRE( command.isZero( 0 ) );
    for( int i = 0; i < 4; ++i )
        supervisor.observe( 0, false );
    for( int i = 0; i < 20; ++i )
        command = supervisor.command( true, true, zero, command, positive, 4, action );
    REQUIRE( supervisor.state() == SimRecovery::Tracking );
    REQUIRE( action == SimAction::Model );
    c.m_train = 8;
    RecoverySupervisor bounded( c, 4, p );
    for( int i = 0; i < 8; ++i )
    {
        bounded.calibrate( i % 2 ? positive : -positive );
        bounded.observe( 0, true );
    }
    int probes = 0;
    command    = positive;
    for( int i = 0; i < 100; ++i )
    {
        command = bounded.command( false, true, positive, command, positive, 4, action );
        probes += action == SimAction::Probe;
        REQUIRE( action != SimAction::Model );
    }
    REQUIRE( probes == c.m_train );
    REQUIRE( bounded.state() == SimRecovery::Disabled );
    REQUIRE( command.isZero( 0 ) );
    for( bool symmetric : { false, true } )
    {
        DDSPC::ShadowConfig      config;
        StabilityWorker          worker( config, 4, symmetric );
        const auto               invalid = static_cast<DDSPC::ShadowMode>( 999 );
        DDSPC::OfflinePrediction prediction;
        DDSPC::Matrix            state = positive, output = DDSPC::Matrix::Ones( config.m_modes, 1 );
        REQUIRE_FALSE( worker.predict( invalid, 1, 0, 1, 1, 100, state, output, prediction ) );
        REQUIRE( prediction.m_reason == DDSPC::OfflinePredictionReason::InvalidState );
        REQUIRE( output.isZero( 0 ) );
        REQUIRE_THROWS_AS( worker.reset( invalid ), std::invalid_argument );
    }
}
} //namespace libXWCTest::loPredCtrlAccTest
