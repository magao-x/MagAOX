/** \file ddspc_operational_test.cpp
 * \brief Verify operational parameter meanings against actual pinned Legacy algebra without modifying production.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "legacy_replay.hpp"
#include "operational_law.hpp"
#include <random>

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief Scaled Legacy regularization is not the literal causal lambda; gain is additive current-WFS feedback. */
TEST_CASE( "Operational penalty scaling and feedback retain their original meanings", "[ddspc][operational]" )
{
    DDSPC::Matrix theta( 1, 2 );
    theta << 3, 2;
    auto actual = DDSPC::operational_legacy_gain( theta, 1, 1, 100, .3f );
    REQUIRE( actual( 0 ) == Approx( .3f - 6.f / 404.f ).margin( 1e-7 ) );
    auto scaled = DDSPC::operational_legacy_gain( 2 * theta, 1, 1, 100, .3f );
    REQUIRE( ( scaled.array() == actual.array() ).all() );
    auto literal = DDSPC::reference_controller(
        theta.leftCols( 1 ), DDSPC::Matrix( 1, 0 ), DDSPC::Matrix( 1, 0 ), theta.rightCols( 1 ), 100 );
    REQUIRE( literal( 0 ) == Approx( -6.f / 104.f ).margin( 1e-7 ) );
    REQUIRE( std::abs( literal( 0 ) - actual( 0 ) ) > .3f );
    theta( 0, 1 ) = 0;
    REQUIRE_THROWS( DDSPC::operational_legacy_gain( theta, 1, 1, 100, .3f ) );
}

/** \brief Compare the causal-state law re-expression with real original command calculations on their own Legacy
 * histories. */
TEST_CASE( "Operational law adapter agrees with actual original Legacy controller", "[ddspc][operational]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::update_controller;
    DDSPC::PredictiveController::calculate_command;
#endif
    // clang-format on
    LegacySettings settings;
    settings.m_modes                             = 2;
    settings.m_history                           = 15;
    settings.m_future                            = 3;
    settings.m_gain                              = .3f;
    settings.m_gamma                             = 1;
    settings.m_lambda                            = 100;
    settings.m_prior                             = 1e-5f;
    settings.m_qrd                               = true;
    auto                                  legacy = original_model( settings );
    std::mt19937                          random( 1701 );
    std::uniform_real_distribution<float> draw( -.2f, .2f );
    for( int i = 0; i < 240; ++i )
    {
        ReplayMatrix y( 2, 1 ), noise( 2, 1 );
        y << draw( random ), draw( random );
        noise << draw( random ), draw( random );
        ReplayMatrix expected;
        if( i >= 64 )
        {
            const auto   before = legacy->snapshot();
            ReplayMatrix theta( 6, 64 );
            theta << before.m_theta.rightCols( 30 ), before.m_theta.middleCols( 6, 28 ), before.m_theta.leftCols( 6 );
            ReplayMatrix state( 58, 1 );
            state << y, before.m_currentY.topRows( 28 ), before.m_currentU.topRows( 28 );
            auto gain = DDSPC::operational_legacy_gain( theta, 2, 15, 100, .3f );
            expected  = gain.bottomRows( 2 ) * state + noise;
        }
        auto actual = legacy->command( y, noise );
        if( i >= 64 )
        {
            REQUIRE( actual.allFinite() );
            REQUIRE( expected.allFinite() );
            REQUIRE( ( actual - expected ).cwiseAbs().maxCoeff() < 2e-6f );
        }
        // Original inverse is undefined while C is still zero. Match a fully trained comparison precondition.
        legacy->learn( i >= 63 );
    }
}

/** \brief The actual checked-out clone ignores copy_gain and the parameter snippet does not enable
 * prediction/integration. */
TEST_CASE( "Operational configuration exposes existing copy gain loading and runtime defaults", "[ddspc][operational]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl;
#endif
    // clang-format on
    auto                               app = current_legacy_app();
    std::map<std::string, std::string> config{
        { "parameters.gain", "0.3" },
        { "parameters.copy_gain", "1.0" },
        { "parameters.regularization", "100" },
        { "parameters.gamma", "1" },
        { "parameters.covariance", "1e-5" },
        { "parameters.history", "15" },
        { "parameters.future", "3" },
        { "parameters.num_modes", "2" },
        { "parameters.qrd", "1" }, // Lifecycle double uses numeric boolean strings.
        { "parameters.own_shmim", "1" },
        { "parameters.fpsSource", "camwfs" } };
    REQUIRE( app->initialize( config, 2 ) == 0 );
    const auto state = app->state_snapshot();
    REQUIRE( state.m_gain == .3f );
    REQUIRE( state.m_copyGain == .3f ); // Existing parser bug, deliberately characterized rather than repaired.
    REQUIRE( state.m_qrd );
    REQUIRE_FALSE( state.m_predict );
    REQUIRE_FALSE( state.m_learn );
    REQUIRE_FALSE( state.m_integrate );
}
} //namespace libXWCTest::loPredCtrlAccTest
