/** \file ddspc_history_test.cpp
 * \brief Deterministic DDSPC history-origin regression tests without application startup.
 * \ingroup loPredCtrlAcc_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../ar_controller.hpp"
#include <limits>
#include <vector>

namespace libXWCTest
{
namespace loPredCtrlAccTest
{

/// \cond DDSPC_TEST_HELPERS
/// Build uniquely tagged modal/channel values from an independent frame number.
DDSPC::Matrix tagged( int count /**< [in] Number of modes/channels. */,
                      int frame /**< [in] Known synthetic frame index. */,
                      int offset /**< [in] Signal-kind tag. */ )
{
    DDSPC::Matrix result( count, 1 );
    for( int j = 0; j < count; ++j )
    {
        result( j ) = offset + 10 * frame + j;
    }
    return result;
}
/// \endcond

/// Reproduce the legacy leakage, then assert that corrected training uses the frozen origin.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Accelerometer history uses the training origin", "[ddspc][history]" )
{
    DDSPC::PredictiveController legacy( 1, 3, 2, 0, 1, 1, 1, 2, 2 );
    DDSPC::PredictiveController causal( 1, 3, 2, 0, 1, 1, 1, 2, 2, DDSPC::HistoryMode::Causal );
    DDSPC::Matrix               frozen;
    for( int t = 0; t <= 7; ++t )
    {
        auto acceleration = tagged( 2, t, 100 );
        auto measurement  = tagged( 1, t, 0 );
        auto applied      = tagged( 1, t, 1000 );
        legacy.push_accelerometer_sample( acceleration );
        legacy.calculate_command( measurement, applied ); // Gain zero: exploration is the exact recorded command.
        causal.append_sample( t, measurement, applied, acceleration );
        if( t == 5 )
        {
            frozen = causal.get_prediction_state();
        }
    }
    DDSPC::Matrix old_phi, old_target, phi, target;
    REQUIRE( legacy.get_training_example( old_phi, old_target ) );
    REQUIRE( causal.get_training_example( phi, target ) );
    REQUIRE( causal.get_measurement_past()( 0 ) == 50 );
    DDSPC::Matrix expected_origin( 4, 1 ), leaked_target_period( 4, 1 );
    expected_origin << 150, 151, 140, 141;      // a_5,a_4, available at origin i=5.
    leaked_target_period << 170, 171, 160, 161; // a_7,a_6, from the target period.
    REQUIRE( old_phi.middleRows( 4, 4 ).isApprox( leaked_target_period, 0 ) );
    REQUIRE_FALSE( old_phi.middleRows( 4, 4 ).isApprox( expected_origin, 0 ) );
    REQUIRE( phi.middleRows( 4, 4 ).isApprox( expected_origin, 0 ) );
    DDSPC::Matrix reconstructed_state( 9, 1 );
    reconstructed_state << phi.bottomRows( 3 ), phi.middleRows( 2, 2 ), phi.middleRows( 4, 4 );
    REQUIRE( reconstructed_state.isApprox( frozen, 0 ) );
    REQUIRE( target.isApprox( old_target, 0 ) );
}

/// Ring rows represent times, so skipping a command must not depend on mode count.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Multichannel command history skips one frame", "[ddspc][history]" )
{
    DDSPC::PredictiveController legacy( 2, 3, 3, 0, 1, 1, 1, 0, 0 );
    DDSPC::PredictiveController causal( 2, 3, 3, 0, 1, 1, 1, 0, 0, DDSPC::HistoryMode::Causal );
    for( int t = 0; t <= 7; ++t )
    {
        auto measurement = tagged( 2, t, 100 );
        auto applied     = tagged( 2, t, 0 );
        legacy.calculate_command( measurement, applied );
        causal.append_sample( t, measurement, applied, DDSPC::Matrix() );
    }
    DDSPC::Matrix expected( 4, 1 );
    expected << 60, 61, 50, 51;
    REQUIRE( causal.get_command_future( 1 ).isApprox( expected, 0 ) );
    DDSPC::Matrix old_phi, old_target, phi, target;
    REQUIRE( legacy.get_training_example( old_phi, old_target ) );
    REQUIRE( causal.get_training_example( phi, target ) );
    DDSPC::Matrix wrong( 6, 1 ), right( 6, 1 );
    wrong << 50, 51, 40, 41, 40, 41;
    right << 60, 61, 50, 51, 40, 41;
    REQUIRE( old_phi.topRows( 6 ).isApprox( wrong, 0 ) );
    REQUIRE( phi.topRows( 6 ).isApprox( right, 0 ) );
    REQUIRE( target.isApprox( old_target, 0 ) );
}

/// Check whole histories, frozen prediction states, warm-up, and wrap using external frame tags.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Causal records preserve every frame and block", "[ddspc][history]" )
{
    for( int m : { 1, 2, 3 } )
        for( int horizon : { 1, 2, 3 } )
            for( int history : { 1, 3 } )
                for( int accel_history : { 0, 2, 5 } )
                {
                    CAPTURE( m, horizon, history, accel_history );
                    const int                   channels     = accel_history == 0 ? 0 : 2;
                    const int                   accel_size   = channels * accel_history;
                    const int                   state_size   = ( 2 * history - 1 ) * m + accel_size;
                    const int                   feature_size = state_size + horizon * m;
                    DDSPC::PredictiveController controller(
                        m, history, horizon, 0, 1, 1, 1, channels, accel_history, DDSPC::HistoryMode::Causal );
                    std::vector<DDSPC::Matrix> snapshots;
                    DDSPC::Matrix              phi = DDSPC::Matrix::Constant( 1, 1, -1 ), target = phi;
                    REQUIRE_FALSE( controller.get_training_example( phi, target ) );
                    REQUIRE( phi( 0 ) == -1 );
                    REQUIRE( target( 0 ) == -1 );
                    REQUIRE_THROWS_AS( controller.get_prediction_state(), std::logic_error );
                    for( int t = 0; t < 130; ++t )
                    {
                        const auto acceleration = channels == 0 ? DDSPC::Matrix() : tagged( channels, t, 2000 );
                        controller.append_sample( t, tagged( m, t, 0 ), tagged( m, t, 10000 ), acceleration );
                        if( t + 1 >= std::max( history, accel_history ) )
                        {
                            snapshots.push_back( controller.get_prediction_state() );
                        }
                        else
                        {
                            snapshots.emplace_back();
                        }
                        const bool ready = t + 1 >= horizon + std::max( history, accel_history );
                        REQUIRE( controller.get_training_example( phi, target ) == ready );
                        if( !ready )
                            continue;
                        const int     origin = t - horizon;
                        DDSPC::Matrix expected( feature_size, 1 ), expected_target( horizon * m, 1 );
                        int           row = 0;
                        for( int j = 0; j < horizon; ++j )
                        {
                            expected.middleRows( row, m )          = tagged( m, t - 1 - j, 10000 );
                            expected_target.middleRows( j * m, m ) = tagged( m, t - j, 0 );
                            row += m;
                        }
                        for( int j = 0; j < history - 1; ++j )
                        {
                            expected.middleRows( row, m ) = tagged( m, origin - 1 - j, 10000 );
                            row += m;
                        }
                        for( int j = 0; j < accel_history; ++j )
                        {
                            expected.middleRows( row, channels ) = tagged( channels, origin - j, 2000 );
                            row += channels;
                        }
                        for( int j = 0; j < history; ++j )
                        {
                            expected.middleRows( row, m ) = tagged( m, origin - j, 0 );
                            row += m;
                        }
                        REQUIRE( phi.rows() == feature_size );
                        REQUIRE( phi.isApprox( expected, 0 ) );
                        REQUIRE( target.isApprox( expected_target, 0 ) );
                        DDSPC::Matrix state( state_size, 1 );
                        state << phi.bottomRows( history * m ), phi.middleRows( horizon * m, ( history - 1 ) * m ),
                            phi.middleRows( ( horizon + history - 1 ) * m, accel_size );
                        REQUIRE( state.isApprox( snapshots[origin], 0 ) );
                    }
                }
}

/// Reject partial/invalid records and implicit candidate-command recording on the offline path.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Causal recording requires explicit valid applied commands", "[ddspc][history]" )
{
    DDSPC::PredictiveController controller( 2, 2, 1, 0, 1, 1, 1, 2, 1, DDSPC::HistoryMode::Causal );
    auto                        y = tagged( 2, 0, 0 ), applied = tagged( 2, 0, 100 ), a = tagged( 2, 0, 200 );
    REQUIRE_THROWS_AS( controller.append_sample( 1, y, applied, a ), std::invalid_argument );
    REQUIRE_THROWS_AS( controller.append_sample( 0, y, DDSPC::Matrix(), a ), std::invalid_argument );
    auto nonfinite = a;
    nonfinite( 0 ) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS( controller.append_sample( 0, y, applied, nonfinite ), std::invalid_argument );
    REQUIRE_THROWS_AS( controller.calculate_command( y, applied ), std::logic_error );
    REQUIRE_THROWS_AS( controller.push_accelerometer_sample( a ), std::logic_error );
    REQUIRE_THROWS_AS( controller.update_controller(), std::logic_error );
    REQUIRE_THROWS_AS( controller.save_state( "unused-causal-artifact" ), std::logic_error );
    REQUIRE_THROWS_AS( controller.load_state( "unused-causal-artifact" ), std::logic_error );
    controller.append_sample( 0, y, applied, a );
    REQUIRE_THROWS_AS( controller.append_sample( 0, y, applied, a ), std::invalid_argument );
    REQUIRE_THROWS_AS( controller.append_sample( 2, y, applied, a ), std::invalid_argument );
    controller.update_system();
    REQUIRE( controller.get_prediction_matrix().isZero( 0 ) );
    controller.append_sample( 1, y, applied, a );
    controller.update_system();
    REQUIRE( controller.get_prediction_matrix().isZero( 0 ) );
    controller.append_sample( 2, y + DDSPC::Matrix::Ones( 2, 1 ), applied, a );
    controller.update_system();
    const auto learned = controller.get_prediction_matrix();
    REQUIRE_FALSE( learned.isZero( 0 ) );
    controller.update_system();
    REQUIRE( controller.get_prediction_matrix().isApprox( learned, 0 ) );
    controller.reset_buffers();
    REQUIRE( controller.get_prediction_matrix().isApprox( learned, 0 ) );
    controller.append_sample( 0, y, applied, a );
    controller.reset();
    REQUIRE( controller.get_prediction_matrix().isZero( 0 ) );
    REQUIRE_THROWS_AS( controller.get_prediction_state(), std::logic_error );
    controller.append_sample( 0, y, applied, a );
}

/// Reject invalid dimensions, nonphysical RLS priors, and storage-index overflow before allocation.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Causal constructor validates numerical dimensions and priors", "[ddspc][history]" )
{
    using DDSPC::HistoryMode;
    using DDSPC::PredictiveController;
    REQUIRE_THROWS_AS( PredictiveController( 0, 2, 1, 0, 1, 1, 1, 0, 0, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 0, 1, 0, 1, 1, 1, 0, 0, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 2, 0, 0, 1, 1, 1, 0, 0, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 2, 1, 0, 0, 1, 1, 0, 0, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 2, 1, 0, 1.1f, 1, 1, 0, 0, HistoryMode::Causal ),
                       std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 2, 1, 0, 1, 1, 0, 0, 0, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 2, 1, 0, 1, 1, 1, -1, 0, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS( PredictiveController( 1, 2, 1, 0, 1, 1, 1, 2, -1, HistoryMode::Causal ), std::invalid_argument );
    REQUIRE_THROWS_AS(
        PredictiveController( std::numeric_limits<int>::max(), 2, 2, 0, 1, 1, 1, 0, 0, HistoryMode::Causal ),
        std::invalid_argument );
    REQUIRE_THROWS_AS(
        PredictiveController( 1, std::numeric_limits<int>::max(), 2, 0, 1, 1, 1, 0, 0, HistoryMode::Causal ),
        std::invalid_argument );
    REQUIRE_THROWS_AS(
        PredictiveController( 1, 2, 1, 0, std::numeric_limits<float>::quiet_NaN(), 1, 1, 0, 0, HistoryMode::Causal ),
        std::invalid_argument );
}

} // namespace loPredCtrlAccTest
} // namespace libXWCTest
