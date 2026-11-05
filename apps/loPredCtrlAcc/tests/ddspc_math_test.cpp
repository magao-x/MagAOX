/** \file ddspc_math_test.cpp
 * \brief Independent double-precision numerical oracles for DDSPC prediction, RLS, and control.
 * \ingroup loPredCtrlAcc_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../ar_controller.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace libXWCTest
{
namespace loPredCtrlAccTest
{

/// \cond DDSPC_TEST_HELPERS
/// Generate portable deterministic values without a platform-dependent random distribution.
Eigen::MatrixXd samples( int      rows /**< [in] Row count. */,
                         int      cols /**< [in] Column count. */,
                         unsigned seed /**< [in] Fixed MT19937 seed. */ )
{
    std::mt19937    generator( seed );
    Eigen::MatrixXd result( rows, cols );
    for( int j = 0; j < cols; ++j )
        for( int i = 0; i < rows; ++i )
            result( i, j ) = ( static_cast<int>( generator() % 2001 ) - 1000 ) / 1000.0;
    return result;
}

/// Measure scaled error without dividing by a near-zero reference.
double relative_error( const Eigen::MatrixXd &actual /**< [in] Calculated values. */,
                       const Eigen::MatrixXd &expected /**< [in] Independent reference values. */ )
{
    return ( actual - expected ).norm() / std::max( 1.0, expected.norm() );
}

/// Build the whole weighted regression design, including the decaying zero-model prior.
Eigen::MatrixXd weighted_design( const Eigen::MatrixXd &inputs /**< [in] Regressors by column. */,
                                 int                    count /**< [in] Number of observations included. */,
                                 double                 gamma /**< [in] Forgetting factor. */,
                                 double                 rho /**< [in] Initial information strength. */ )
{
    const int       p                      = inputs.rows();
    Eigen::MatrixXd design                 = Eigen::MatrixXd::Zero( p + count, p );
    design.topRows( p ).diagonal().array() = std::sqrt( rho * std::pow( gamma, count ) );
    for( int j = 0; j < count; ++j )
        design.row( p + j ) = std::sqrt( std::pow( gamma, count - 1 - j ) ) * inputs.col( j ).transpose();
    return design;
}

/// Solve the stated cost by a double-precision augmented least-squares QR, independently of the production LDLT.
Eigen::MatrixXd control_oracle( const Eigen::MatrixXd &C /**< [in] Future input response. */,
                                const Eigen::MatrixXd &state /**< [in] Predicted output from known histories. */,
                                double                 lambda /**< [in] Literal positive command cost. */ )
{
    Eigen::MatrixXd design( C.rows() + C.cols(), C.cols() );
    design << C, std::sqrt( lambda ) * Eigen::MatrixXd::Identity( C.cols(), C.cols() );
    Eigen::MatrixXd target     = Eigen::MatrixXd::Zero( design.rows(), state.cols() );
    target.topRows( C.rows() ) = -state;
    return design.colPivHouseholderQr().solve( target );
}
/// \endcond

/// Verify the explicit extended prediction equation and literal-lambda optimum with independently formed blocks.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Extended DDSPC control matches the quadratic objective", "[ddspc][control]" )
{
    const DDSPC::Matrix   A      = samples( 6, 3, 11 ).cast<float>();
    const DDSPC::Matrix   B      = samples( 6, 2, 12 ).cast<float>();
    const DDSPC::Matrix   D      = samples( 6, 4, 13 ).cast<float>();
    const DDSPC::Matrix   full_C = samples( 6, 4, 14 ).cast<float>();
    const Eigen::VectorXd y      = samples( 3, 1, 15 );
    const Eigen::VectorXd u      = samples( 2, 1, 16 );
    const Eigen::VectorXd a      = samples( 4, 1, 17 );
    const Eigen::VectorXd known  = A.cast<double>() * y + B.cast<double>() * u + D.cast<double>() * a;
    Eigen::VectorXd       state( 9 );
    state << y, u, a;
    Eigen::MatrixXd theta( 6, 13 );
    theta << A.cast<double>(), B.cast<double>(), D.cast<double>(), full_C.cast<double>();
    const Eigen::VectorXd future = samples( 4, 1, 18 );
    Eigen::VectorXd       phi( 13 );
    phi << state, future;
    REQUIRE( relative_error( theta * phi, known + full_C.cast<double>() * future ) < 1e-14 );

    for( int rank_case : { 0, 1, 2 } )
        for( float lambda : { 0.01f, 0.7f, 3.0f } )
        {
            CAPTURE( rank_case, lambda );
            DDSPC::Matrix C = full_C;
            if( rank_case == 1 )
                C.col( 3 ) = C.col( 1 );
            if( rank_case == 2 )
                C.setZero();
            const auto gain = DDSPC::reference_controller( A, B, D, C, lambda );
            REQUIRE( gain.rows() == 4 );
            REQUIRE( gain.cols() == 9 );
            const Eigen::VectorXd actual   = gain.cast<double>() * state;
            const auto            expected = control_oracle( C.cast<double>(), known, lambda );
            REQUIRE( relative_error( actual, expected ) < 2e-5 );
            const Eigen::VectorXd gradient =
                C.cast<double>().transpose() * ( C.cast<double>() * actual + known ) + lambda * actual;
            REQUIRE( gradient.norm() / std::max( 1.0, known.norm() ) < 3e-6 );
            const double optimum_cost =
                ( known + C.cast<double>() * actual ).squaredNorm() + lambda * actual.squaredNorm();
            for( int j = 0; j < 4; ++j )
            {
                auto perturbed = actual;
                perturbed( j ) += 0.1;
                const double cost =
                    ( known + C.cast<double>() * perturbed ).squaredNorm() + lambda * perturbed.squaredNorm();
                REQUIRE( cost > optimum_cost );
            }
            if( rank_case == 2 )
                REQUIRE( actual.isZero( 0 ) );
        }
}

/// Establish the scalar sign and show that model-scaled regularization/additional feedback are distinct laws.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Reference cost is distinct from legacy scaled feedback", "[ddspc][control]" )
{
    DDSPC::Matrix A( 1, 1 ), B( 1, 1 ), D( 1, 1 ), C( 1, 1 );
    A << 2;
    B << -1;
    D << 3;
    C << 2;
    Eigen::Vector3d state( 0.5, 0.25, -0.5 );
    const double    known             = 2 * 0.5 - 0.25 + 3 * -0.5;
    const auto      reference         = DDSPC::reference_controller( A, B, D, C, 0.5f ).cast<double>().eval();
    const double    reference_command = ( reference * state )( 0 );
    REQUIRE( reference_command == Approx( -2 * known / 4.5 ).epsilon( 1e-6 ) );
    const double legacy_scaled_command = -2 * known / ( 4 + 4 * 0.5 );
    REQUIRE( std::abs( reference_command - legacy_scaled_command ) > 0.05 );
    REQUIRE( std::abs( reference_command - ( legacy_scaled_command + 0.25 * state( 0 ) ) ) > 0.01 );
    const DDSPC::Matrix empty( 1, 0 );
    REQUIRE( DDSPC::reference_controller( A, empty, empty, C, 0.5f ).cols() == 1 );
    REQUIRE_THROWS_AS( DDSPC::reference_controller( A, B, D, C, 0 ), std::invalid_argument );
    REQUIRE_THROWS_AS( DDSPC::reference_controller( A, B, D, C, -1 ), std::invalid_argument );
    REQUIRE_THROWS_AS( DDSPC::reference_controller( A, B, D, C, std::numeric_limits<float>::infinity() ),
                       std::invalid_argument );
    REQUIRE_THROWS_AS( DDSPC::reference_controller( DDSPC::Matrix::Zero( 2, 1 ), B, D, C, 1 ), std::invalid_argument );
    C( 0 ) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS( DDSPC::reference_controller( A, B, D, C, 1 ), std::invalid_argument );
    C( 0 ) = std::numeric_limits<float>::max();
    REQUIRE_THROWS_AS( DDSPC::reference_controller( A, B, D, C, 1 ), std::runtime_error );
}

/// Compare every conventional/QR RLS update to a freshly assembled weighted batch regression and SVD information.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "RLS and QR RLS agree with independent weighted batch least squares", "[ddspc][rls]" )
{
    constexpr int       p = 12, d = 4, steps = 72;
    const DDSPC::Matrix A = samples( d, 4, 31 ).cast<float>();
    const DDSPC::Matrix B = samples( d, 2, 32 ).cast<float>();
    const DDSPC::Matrix D = samples( d, 2, 33 ).cast<float>();
    const DDSPC::Matrix C = samples( d, 4, 34 ).cast<float>();
    DDSPC::Matrix       truth( d, p );
    truth << C, B, D, A;
    const Eigen::MatrixXd input = samples( p, steps, 35 ).cast<float>().cast<double>();
    const DDSPC::Matrix   noise = samples( d, steps, 36 ).cast<float>();
    Eigen::MatrixXd       output( d, steps );
    for( int j = 0; j < steps; ++j )
        output.col( j ) = ( truth * input.col( j ).cast<float>() + 0.01f * noise.col( j ) ).cast<double>();

    for( float gamma : { 1.0f, 0.97f } )
    {
        CAPTURE( gamma );
        constexpr float                 rho = 0.8f;
        DDSPC::RecursiveLeastSquares    classic( d, p, gamma, rho ), image_classic( d, p, gamma, rho );
        DDSPC::QRDRecursiveLeastSquares qrd( d, p, gamma, rho, true ), image_qrd( d, p, gamma, rho, true );
        DDSPC::QRDRecursiveLeastSquares legacy_qrd( d, p, gamma, rho ), explicit_legacy( d, p, gamma, rho, false );
        double                          max_weights = 0, max_covariance = 0, max_factor = 0, max_gain = 0;
        for( int k = 1; k <= steps; ++k )
        {
            CAPTURE( k );
            DDSPC::Matrix                 x = input.col( k - 1 ).cast<float>(), z = output.col( k - 1 ).cast<float>();
            mx::improc::eigenImage<float> image_x = x.array(), image_z = z.array();
            classic.update( &x, &z );
            qrd.update( &x, &z );
            legacy_qrd.update( &x, &z );
            explicit_legacy.update( &x, &z );
            image_classic.update( &image_x, &image_z );
            image_qrd.update( &image_x, &image_z );
            const auto      design = weighted_design( input, k, gamma, rho );
            Eigen::MatrixXd labels = Eigen::MatrixXd::Zero( p + k, d );
            for( int j = 0; j < k; ++j )
                labels.row( p + j ) = std::sqrt( std::pow( double( gamma ), k - 1 - j ) ) * output.col( j ).transpose();
            const Eigen::MatrixXd                   weights = design.colPivHouseholderQr().solve( labels ).transpose();
            const Eigen::JacobiSVD<Eigen::MatrixXd> svd( design, Eigen::ComputeThinV );
            const Eigen::MatrixXd covariance  = svd.matrixV() *
                                                svd.singularValues().array().square().inverse().matrix().asDiagonal() *
                                                svd.matrixV().transpose();
            const Eigen::MatrixXd information = design.transpose() * design;
            const Eigen::MatrixXd gain        = covariance * input.col( k - 1 );
            max_weights = std::max( { max_weights,
                                      relative_error( classic.prediction_matrix.cast<double>(), weights ),
                                      relative_error( qrd.prediction_matrix.cast<double>(), weights ) } );
            max_covariance =
                std::max( max_covariance, relative_error( classic.inverse_covariance.cast<double>(), covariance ) );
            max_factor = std::max(
                max_factor, relative_error( qrd.R_.cast<double>().transpose() * qrd.R_.cast<double>(), information ) );
            max_gain = std::max( { max_gain,
                                   relative_error( classic.K.cast<double>().transpose(), gain ),
                                   relative_error( qrd.g.cast<double>(), gain ) } );
            REQUIRE( relative_error( classic.prediction_matrix.cast<double>(), weights ) < 3e-5 );
            REQUIRE( relative_error( qrd.prediction_matrix.cast<double>(), weights ) < 3e-5 );
            REQUIRE( relative_error( legacy_qrd.prediction_matrix.cast<double>(), weights ) < 3e-5 );
            REQUIRE( legacy_qrd.prediction_matrix.isApprox( explicit_legacy.prediction_matrix, 0 ) );
            REQUIRE( relative_error( classic.inverse_covariance.cast<double>(), covariance ) < 3e-5 );
            REQUIRE( relative_error( qrd.R_.cast<double>().transpose() * qrd.R_.cast<double>(), information ) < 3e-5 );
            REQUIRE( relative_error( classic.K.cast<double>().transpose(), gain ) < 3e-5 );
            REQUIRE( relative_error( qrd.g.cast<double>(), gain ) < 3e-5 );
            REQUIRE( image_classic.prediction_matrix.isApprox( classic.prediction_matrix, 1e-6f ) );
            REQUIRE( image_qrd.prediction_matrix.isApprox( qrd.prediction_matrix, 1e-6f ) );
            REQUIRE( ( classic.inverse_covariance - classic.inverse_covariance.transpose() ).norm() < 2e-6f );
            const Eigen::MatrixXd symmetric =
                ( classic.inverse_covariance.cast<double>() + classic.inverse_covariance.cast<double>().transpose() ) /
                2;
            REQUIRE( Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>( symmetric ).eigenvalues().minCoeff() > 0 );
        }
        std::cout << "RLS oracle gamma=" << gamma << " max scaled errors: weights=" << max_weights
                  << " covariance=" << max_covariance << " QR information=" << max_factor << " gain=" << max_gain
                  << '\n';
        classic.reset();
        qrd.reset();
        REQUIRE( classic.prediction_matrix.isZero( 0 ) );
        REQUIRE( qrd.prediction_matrix.isZero( 0 ) );
        REQUIRE( classic.inverse_covariance.isApprox( DDSPC::Matrix::Identity( p, p ) / rho, 1e-7f ) );
        REQUIRE( qrd.R_.isApprox( std::sqrt( rho ) * DDSPC::Matrix::Identity( p, p ), 1e-7f ) );
        DDSPC::Matrix zero = DDSPC::Matrix::Zero( p, 1 ), label = DDSPC::Matrix::Ones( d, 1 );
        classic.update( &zero, &label );
        qrd.update( &zero, &label );
        REQUIRE( classic.prediction_matrix.isZero( 0 ) );
        REQUIRE( qrd.prediction_matrix.isZero( 0 ) );
        REQUIRE( classic.inverse_covariance.isApprox( DDSPC::Matrix::Identity( p, p ) / ( rho * gamma ), 1e-6f ) );
    }
}

/// Diagnose QR stability on an independently assembled regression with unused accelerometer columns.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "QR RLS preserves the information factor with zero feature columns", "[ddspc][rls]" )
{
    constexpr int   p = 16, d = 4, steps = 320;
    constexpr float gamma = 0.99f, rho = 0.5f;
    Eigen::MatrixXd input = samples( p, steps, 41 ).cast<float>().cast<double>();
    input.middleRows( 6, 6 ).setZero();
    const Eigen::MatrixXd           truth  = samples( d, p, 42 );
    const Eigen::MatrixXd           output = ( truth * input ).cast<float>().cast<double>();
    DDSPC::QRDRecursiveLeastSquares learner( d, p, gamma, rho, true );
    for( int j = 0; j < steps; ++j )
    {
        DDSPC::Matrix x = input.col( j ).cast<float>(), y = output.col( j ).cast<float>();
        learner.update( &x, &y );
    }
    const auto      design = weighted_design( input, steps, gamma, rho );
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero( p + steps, d );
    for( int j = 0; j < steps; ++j )
        labels.row( p + j ) = std::sqrt( std::pow( double( gamma ), steps - 1 - j ) ) * output.col( j ).transpose();
    const Eigen::MatrixXd weights = design.colPivHouseholderQr().solve( labels ).transpose();
    const DDSPC::Matrix   lower   = learner.R_.triangularView<Eigen::StrictlyLower>();
    INFO( "QR lower triangle norm=" << lower.norm() );
    INFO( "QR information error=" << relative_error( learner.R_.cast<double>().transpose() * learner.R_.cast<double>(),
                                                     design.transpose() * design ) );
    INFO( "QR weights error=" << relative_error( learner.prediction_matrix.cast<double>(), weights ) );
    REQUIRE( lower.isZero( 0 ) );
    REQUIRE( relative_error( learner.prediction_matrix.cast<double>(), weights ) < 3e-5 );
    REQUIRE( relative_error( learner.R_.cast<double>().transpose() * learner.R_.cast<double>(),
                             design.transpose() * design ) < 3e-5 );
    std::cout << "Zero-column QR: weights error=" << relative_error( learner.prediction_matrix.cast<double>(), weights )
              << " information error="
              << relative_error( learner.R_.cast<double>().transpose() * learner.R_.cast<double>(),
                                 design.transpose() * design )
              << '\n';
}

/// Learn a known causal MIMO plant using explicitly synthetic held/clipped applied inputs, not proposals.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Causal identification learns the accelerometer D block", "[ddspc][identification]" )
{
    DDSPC::Matrix A0( 2, 2 ), A1( 2, 2 ), B( 2, 2 ), D( 2, 2 ), C( 2, 2 );
    A0 << 0.25f, 0.04f, -0.03f, 0.2f;
    A1 << -0.06f, 0.01f, 0.02f, -0.04f;
    B << 0.08f, -0.02f, 0.03f, 0.06f;
    D << 0.5f, -0.15f, 0.12f, 0.4f;
    C << 0.65f, 0.1f, -0.08f, 0.55f;
    DDSPC::Matrix truth( 2, 10 );
    truth << A0, A1, B, D, C; // Canonical [A B D C], with A=[A0 A1].
    constexpr int steps      = 2400;
    const auto    excitation = samples( 4, steps, 51 ).cast<float>().eval();
    for( bool use_qrd : { false, true } )
    {
        CAPTURE( use_qrd );
        DDSPC::PredictiveController model( 2, 2, 1, 0, 1, 1, 0.02f, 2, 1, DDSPC::HistoryMode::Causal );
        DDSPC::PredictiveController baseline( 2, 2, 1, 0, 1, 1, 0.02f, 0, 0, DDSPC::HistoryMode::Causal );
        model.use_qrd = baseline.use_qrd = use_qrd;
        DDSPC::Matrix   y = DDSPC::Matrix::Zero( 2, 1 ), previous_y = y, previous_u = y;
        Eigen::MatrixXd regressors( 10, steps - 2 ), labels( 2, steps - 2 );
        int             count           = 0;
        double          augmented_error = 0, baseline_error = 0;
        for( int t = 0; t < steps; ++t )
        {
            // Synthetic actuator contract: apply a held or clipped proposal over [t,t+1).
            const DDSPC::Matrix proposal = 2 * excitation.topRows( 2 ).col( t );
            const DDSPC::Matrix applied =
                t % 7 == 0 ? previous_u : proposal.array().max( -0.7f ).min( 0.7f ).matrix().eval();
            const DDSPC::Matrix a = excitation.bottomRows( 2 ).col( t );
            model.append_sample( t, y, applied, a );
            baseline.append_sample( t, y, applied, DDSPC::Matrix() );
            DDSPC::Matrix phi, target;
            if( model.get_training_example( phi, target ) )
            {
                // F=1: the matured target y_t must use u_(t-1), never proposal_t or u_t.
                REQUIRE( phi.topRows( 2 ).isApprox( previous_u, 0 ) );
                Eigen::VectorXd canonical( 10 );
                canonical << phi.bottomRows( 4 ).cast<double>(), phi.middleRows( 2, 2 ).cast<double>(),
                    phi.middleRows( 4, 2 ).cast<double>(), phi.topRows( 2 ).cast<double>();
                regressors.col( count ) = canonical;
                labels.col( count++ )   = target.cast<double>();
                REQUIRE( relative_error( truth.cast<double>() * canonical, target.cast<double>() ) < 2e-7 );
            }
            if( t < 1800 )
            {
                model.update_system();
                baseline.update_system();
            }
            else
            {
                const DDSPC::Matrix state      = model.get_prediction_state();
                const DDSPC::Matrix base_state = baseline.get_prediction_state();
                DDSPC::Matrix       canonical_phi( 10, 1 ), base_phi( 8, 1 );
                canonical_phi << state, applied;
                base_phi << base_state, applied;
                const DDSPC::Matrix next_y = A0 * y + A1 * previous_y + B * previous_u + D * a + C * applied;
                augmented_error += ( model.get_reference_prediction_matrix() * canonical_phi - next_y ).squaredNorm();
                baseline_error += ( baseline.get_reference_prediction_matrix() * base_phi - next_y ).squaredNorm();
            }
            const DDSPC::Matrix next_y = A0 * y + A1 * previous_y + B * previous_u + D * a + C * applied;
            previous_y                 = y;
            previous_u                 = applied;
            y                          = next_y;
        }
        const DDSPC::Matrix learned           = model.get_reference_prediction_matrix();
        const double        coefficient_error = relative_error( learned.cast<double>(), truth.cast<double>() );
        const double        d_error = relative_error( learned.middleCols( 6, 2 ).cast<double>(), D.cast<double>() );
        REQUIRE( coefficient_error < 0.003 );
        REQUIRE( d_error < 0.001 );
        REQUIRE( augmented_error < baseline_error * 1e-4 );
        // Independently verify the extended predictor on every matured synthetic applied-command record.
        REQUIRE( relative_error( truth.cast<double>() * regressors.leftCols( count ), labels.leftCols( count ) ) <
                 2e-7 );
        const auto      gain = model.get_reference_controller( 0.3f );
        Eigen::MatrixXd known( 2, 8 );
        known << learned.leftCols( 8 ).cast<double>();
        const auto oracle = control_oracle( learned.rightCols( 2 ).cast<double>(), known, 0.3f );
        REQUIRE( relative_error( gain.cast<double>(), oracle ) < 2e-6 );
        std::cout << "Synthetic MIMO " << ( use_qrd ? "QR" : "classic" ) << ": coefficient error=" << coefficient_error
                  << " D error=" << d_error << " held-out augmented RMSE=" << std::sqrt( augmented_error / 1200 )
                  << " baseline RMSE=" << std::sqrt( baseline_error / 1200 ) << '\n';
    }
}

/// Removing accelerometer columns must reproduce the same corrected WFS/DM fit, predictions, and commands.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Disabled accelerometers preserve the corrected WFS DM baseline", "[ddspc][baseline]" )
{
    const auto data = samples( 4, 320, 71 ).cast<float>().eval();
    for( bool use_qrd : { false, true } )
    {
        CAPTURE( use_qrd );
        DDSPC::PredictiveController disabled( 2, 2, 2, 0, 0.99f, 1, 0.5f, 0, 0, DDSPC::HistoryMode::Causal );
        DDSPC::PredictiveController zero_history( 2, 2, 2, 0, 0.99f, 1, 0.5f, 2, 0, DDSPC::HistoryMode::Causal );
        DDSPC::PredictiveController unused_columns( 2, 2, 2, 0, 0.99f, 1, 0.5f, 2, 3, DDSPC::HistoryMode::Causal );
        disabled.use_qrd = zero_history.use_qrd = unused_columns.use_qrd = use_qrd;
        for( int t = 0; t < data.cols(); ++t )
        {
            const DDSPC::Matrix y = data.topRows( 2 ).col( t ), applied = data.bottomRows( 2 ).col( t );
            disabled.append_sample( t, y, applied, DDSPC::Matrix() );
            zero_history.append_sample( t, y, applied, DDSPC::Matrix() );
            unused_columns.append_sample( t, y, applied, DDSPC::Matrix::Zero( 2, 1 ) );
            // Use identical mature examples: extra zero columns need a longer initial history.
            if( t >= 4 )
            {
                disabled.update_system();
                zero_history.update_system();
                unused_columns.update_system();
            }
        }
        const auto base  = disabled.get_reference_prediction_matrix();
        const auto extra = unused_columns.get_reference_prediction_matrix();
        REQUIRE( base.cols() == 10 );
        REQUIRE( extra.cols() == 16 );
        REQUIRE( base.isApprox( zero_history.get_reference_prediction_matrix(), 0 ) );
        DDSPC::Matrix stripped( 4, 10 );
        stripped << extra.leftCols( 6 ), extra.rightCols( 4 );
        INFO( "Baseline zero-column scaled error=" << relative_error( stripped.cast<double>(), base.cast<double>() ) );
        // Use the same max(1,norm) error convention as the batch oracle; this random fit is near zero.
        REQUIRE( relative_error( stripped.cast<double>(), base.cast<double>() ) < 3e-5 );
        REQUIRE( extra.middleCols( 6, 6 ).isZero( 0 ) );
        const DDSPC::Matrix gain           = disabled.get_reference_controller( 0.7f );
        const DDSPC::Matrix augmented_gain = unused_columns.get_reference_controller( 0.7f );
        REQUIRE( relative_error( gain.cast<double>(), augmented_gain.leftCols( 6 ).cast<double>() ) < 3e-5 );
        const auto state = disabled.get_prediction_state();
        REQUIRE( relative_error( ( gain * state ).cast<double>(),
                                 ( augmented_gain * unused_columns.get_prediction_state() ).cast<double>() ) < 3e-5 );
        const auto oracle =
            control_oracle( base.rightCols( 4 ).cast<double>(), base.leftCols( 6 ).cast<double>(), 0.7f );
        REQUIRE( relative_error( gain.cast<double>(), oracle ) < 2e-6 );
        // Newest-first future ordering: u_i is the last two rows, not the most distant first block.
        REQUIRE( relative_error( ( gain.bottomRows( 2 ) * state ).cast<double>(),
                                 oracle.bottomRows( 2 ) * state.cast<double>() ) < 2e-6 );
        REQUIRE( ( gain.topRows( 2 ) * state - gain.bottomRows( 2 ) * state ).norm() > 1e-6 );
    }
}

/// Legacy constructor callers retain model-scaled penalties, added feedback, and calculated-command recording.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Legacy operational law remains an explicit compatibility path", "[ddspc][baseline][compatibility]" )
{
    constexpr int               N = 3, F = 2;
    constexpr float             gain = -0.2f, penalty = 0.4f;
    DDSPC::PredictiveController legacy( 1, N, F, gain, 1, penalty, 1, 0, 0 );
    DDSPC::PredictiveController causal( 1, N, F, gain, 1, penalty, 1, 0, 0, DDSPC::HistoryMode::Causal );
    const auto                  data         = samples( 2, 120, 81 ).cast<float>().eval();
    Eigen::VectorXd             old_commands = Eigen::VectorXd::Zero( 120 );
    for( int t = 0; t < 120; ++t )
    {
        const DDSPC::Matrix residual = data.topRows( 1 ).col( t );
        const DDSPC::Matrix noise    = 0.05f * data.bottomRows( 1 ).col( t );
        Eigen::VectorXd     state    = Eigen::VectorXd::Zero( 2 * N - 1 );
        for( int j = 0; j < N - 1; ++j )
            if( t > j )
                state( j ) = old_commands( t - 1 - j );
        for( int j = 0; j < N; ++j )
            if( t >= j )
                state( N - 1 + j ) = data( 0, t - j );
        double expected = gain * residual( 0 ) + noise( 0 );
        if( t >= N + F )
        {
            const Eigen::MatrixXd theta         = legacy.get_prediction_matrix().cast<double>();
            const Eigen::MatrixXd C             = theta.leftCols( F );
            const double          scaled_lambda = ( C.transpose() * C ).maxCoeff() * penalty;
            expected += ( control_oracle( C, theta.rightCols( 2 * N - 1 ), scaled_lambda ) * state )( F - 1 );
        }
        const auto applied = legacy.calculate_command( residual, noise );
        REQUIRE( applied( 0 ) == Approx( expected ).margin( 3e-6 ) );
        old_commands( t ) = applied( 0 );
        causal.append_sample( t, residual, applied, DDSPC::Matrix() );
        if( t + 1 >= N + F )
        {
            legacy.update_system();
            causal.update_system();
            legacy.update_controller();
            REQUIRE( legacy.get_prediction_matrix().isApprox( causal.get_prediction_matrix(), 1e-6f ) );
        }
    }
}

} // namespace loPredCtrlAccTest
} // namespace libXWCTest
