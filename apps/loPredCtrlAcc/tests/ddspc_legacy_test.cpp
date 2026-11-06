/** \file ddspc_legacy_test.cpp
 * \brief Execute pinned original and current legacy DDSPC against identical deterministic records.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "legacy_replay.hpp"
#include <cstring>
#include <cstdint>
#include <limits>
#include <random>
#include <filesystem>
#include <cstdlib>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond LEGACY_TEST_SUPPORT
namespace
{
/// Own a unique temporary directory for numerical state artifacts, never the instrument calibration tree.
class LegacyTempDir
{
  public:
    /// Newly created test-owned directory.
    std::filesystem::path m_path;
    /// Create an isolated temporary directory.
    LegacyTempDir();
    /// Remove only this fixture's newly created artifacts.
    ~LegacyTempDir();
};

LegacyTempDir::LegacyTempDir()
{
    std::random_device random;
    for( int attempt = 0; attempt < 64; ++attempt )
    {
        auto candidate =
            std::filesystem::temp_directory_path() / ( "ddspc-legacy-review-" + std::to_string( random() ) );
        if( std::filesystem::create_directory( candidate ) )
        {
            m_path = std::move( candidate );
            return;
        }
    }
    throw std::runtime_error( "Cannot create legacy replay temporary directory" );
}

LegacyTempDir::~LegacyTempDir()
{
    std::error_code ignored;
    std::filesystem::remove_all( m_path, ignored );
}

/// Compare float representations exactly, including signed zero; no tolerance hides drift.
bool same_bits( const ReplayMatrix &a /**< [in] Original. */, const ReplayMatrix &b /**< [in] Comparison. */ )
{
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           ( a.size() == 0 || std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) ) == 0 );
}

/// Inspect exponent bits so the test oracle still detects NaNs/Infs under finite-only compilation.
bool finite_bits( const ReplayMatrix &values /**< [in] Numerical state to inspect. */ )
{
    static_assert( sizeof( float ) == sizeof( std::uint32_t ) );
    static_assert( std::numeric_limits<float>::is_iec559 );
    for( Eigen::Index i = 0; i < values.size(); ++i )
    {
        std::uint32_t bits;
        std::memcpy( &bits, values.data() + i, sizeof( bits ) );
        if( ( bits & 0x7f800000u ) == 0x7f800000u )
            return false;
    }
    return true;
}

/// Require exact public history and coefficient equality, independent of internal buffer capacities.
void require_snapshot( const LegacySnapshot &a /**< [in] Original state. */,
                       const LegacySnapshot &b /**< [in] Current state. */ )
{
    REQUIRE( same_bits( a.m_futureY, b.m_futureY ) );
    REQUIRE( same_bits( a.m_pastY, b.m_pastY ) );
    REQUIRE( same_bits( a.m_futureU, b.m_futureU ) );
    REQUIRE( same_bits( a.m_pastU, b.m_pastU ) );
    REQUIRE( same_bits( a.m_currentY, b.m_currentY ) );
    REQUIRE( same_bits( a.m_currentU, b.m_currentU ) );
    REQUIRE( same_bits( a.m_theta, b.m_theta ) );
}

/// Supply deterministic exogenous WFS/command/exploration traces without simulator state or hardware.
ReplayMatrix replay_signal( int modes /**< [in] Vector length. */, std::uint64_t seed /**< [in] Trace identity. */ )
{
    std::mt19937 random( seed );
    ReplayMatrix value( modes, 1 );
    for( int j = 0; j < modes; ++j )
        value( j ) = float( int( random() % 2001 ) - 1000 ) / 1000;
    return value;
}

/// Match all operational parameters explicitly, including the historical QR choice and disabled acceleration.
std::map<std::string, std::string> replay_config( bool qrd /**< [in] RLS algorithm. */ )
{
    return { { "parameters.num_modes", "2" },
             { "parameters.history", "5" },
             { "parameters.future", "2" },
             { "parameters.gain", "-0.25" },
             { "parameters.regularization", "100" },
             { "parameters.gamma", "0.99" },
             { "parameters.covariance", "1" },
             { "parameters.qrd", qrd ? "1" : "0" },
             { "parameters.accel_enabled", "0" },
             { "parameters.is_integrating", "0" } };
}
} //namespace
/// \endcond

/// The test oracle must detect injected IEEE infinities/NaNs even when fast-math assumes finite arithmetic.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Legacy finite-state oracle survives finite-only compiler flags", "[loPredCtrlAcc][legacy]" )
{
    ReplayMatrix values = ReplayMatrix::Zero( 2, 1 );
    REQUIRE( finite_bits( values ) );
    for( std::uint32_t bits : { 0x7f800000u, 0xff800000u, 0x7fc00000u } )
    {
        std::memcpy( values.data(), &bits, sizeof( bits ) );
        REQUIRE_FALSE( finite_bits( values ) );
    }
    values.setZero();
    REQUIRE( finite_bits( values ) );
}

/// Isolate the historical reciprocal-overflow failure; run explicitly to inspect compiler/FP-mode behavior.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Historical tiny Givens rotations must remain finite", "[.legacy-tiny-rotation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::QRDRecursiveLeastSquares::apply_givens();
#endif
    // clang-format on
    auto         old       = original_qr( 3, 4, 1, 1 );
    ReplayMatrix workspace = ReplayMatrix::Zero( 5, 4 );
    workspace.row( 0 ) << 1.e-40f, .123f, -.578f, -.578f;
    workspace.row( 1 ) << -2.e-40f, .456f, .382f, .382f;
    const auto result = old->rotate( workspace );
    REQUIRE( finite_bits( result ) );
}

/// Quantify compiler-dependent drift separately; the default exact regression requirements are retained.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Measure legacy compiler drift without relaxing exact regressions", "[.legacy-drift]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::calculate_command();
    DDSPC::PredictiveController::update_system();
    DDSPC::PredictiveController::get_prediction_matrix();
#endif
    // clang-format on
    for( bool qr : { false, true } )
    {
        double      commandError = 0, fitError = 0, restoredError = 0;
        std::size_t commandDifferences = 0, fitDifferences = 0;
        for( int modes : { 1, 2 } )
            for( int history : { 1, 5 } )
                for( int future : { 1, 2, 3 } )
                    for( float gamma : { 1.f, .97f } )
                    {
                        LegacySettings s;
                        s.m_modes   = modes;
                        s.m_history = history;
                        s.m_future  = future;
                        s.m_qrd     = qr;
                        s.m_gamma   = gamma;
                        auto a = original_model( s ), b = restored_model( s ), c = current_legacy_model( s );
                        for( int t = 0; t < 192; ++t )
                        {
                            if( t == 80 || t == 160 )
                                for( auto *model : { a.get(), b.get(), c.get() } )
                                    model->reset( t == 160 );
                            if( t == 32 || t == 112 )
                                for( auto *model : { a.get(), b.get(), c.get() } )
                                    model->regularization( t == 32 ? 25.f : 100.f );
                            const auto         y     = replay_signal( modes, 7811 + t );
                            const ReplayMatrix noise = .1f * replay_signal( modes, 1783 + t );
                            const auto         x = a->command( y, noise ), z = b->command( y, noise ),
                                               v = c->command( y, noise );
                            REQUIRE( x.allFinite() );
                            REQUIRE( z.allFinite() );
                            REQUIRE( v.allFinite() );
                            commandError  = std::max( commandError, double( ( x - v ).cwiseAbs().maxCoeff() ) );
                            restoredError = std::max( restoredError, double( ( x - z ).cwiseAbs().maxCoeff() ) );
                            commandDifferences += !same_bits( x, v );
                            const bool solve = t >= 64 && !( t >= 80 && t < 96 );
                            for( auto *model : { a.get(), b.get(), c.get() } )
                                model->learn( solve );
                            const auto ta = a->snapshot().m_theta, tb = b->snapshot().m_theta,
                                       tc = c->snapshot().m_theta;
                            REQUIRE( ta.allFinite() );
                            REQUIRE( tb.allFinite() );
                            REQUIRE( tc.allFinite() );
                            fitError      = std::max( fitError, double( ( ta - tc ).cwiseAbs().maxCoeff() ) );
                            restoredError = std::max( restoredError, double( ( ta - tb ).cwiseAbs().maxCoeff() ) );
                            fitDifferences += !same_bits( ta, tc );
                        }
                    }
        std::cout << "Legacy compiler drift qrd=" << qr << " max_abs_command=" << commandError
                  << " max_abs_fit=" << fitError << " differing_commands=" << commandDifferences
                  << " differing_fits=" << fitDifferences << " original_vs_restored=" << restoredError << '\n';
        REQUIRE( restoredError == 0 );
    }
}

/** \brief Identical explicit DM traces and subsequent predictive operation reproduce the original numerical model.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Pinned original and current legacy numerical controllers are identical on matched traces",
           "[loPredCtrlAcc][legacy]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::calculate_command();
    DDSPC::PredictiveController::update_system();
    DDSPC::PredictiveController::update_controller();
    DDSPC::PredictiveController::reset();
    DDSPC::PredictiveController::reset_buffers();
#endif
    // clang-format on
    std::size_t comparisons = 0;
    for( int modes : { 1, 2 } )
        for( int history : { 1, 5 } )
            for( int future : { 1, 2, 3 } )
                for( bool qr : { false, true } )
                    for( float gamma : { 1.0f, 0.97f } )
                    {
                        LegacySettings s;
                        s.m_modes     = modes;
                        s.m_history   = history;
                        s.m_future    = future;
                        s.m_qrd       = qr;
                        s.m_gamma     = gamma;
                        auto original = original_model( s ), restored = restored_model( s ),
                             current = current_legacy_model( s );
                        for( int t = 0; t < 192; ++t )
                        {
                            CAPTURE( modes, history, future, qr, gamma, t );
                            if( t == 80 )
                            {
                                original->reset( false );
                                restored->reset( false );
                                current->reset( false );
                            }
                            if( t == 160 )
                            {
                                original->reset( true );
                                restored->reset( true );
                                current->reset( true );
                            }
                            if( t == 32 || t == 112 )
                                for( auto *model : { original.get(), restored.get(), current.get() } )
                                    model->regularization( t == 32 ? 25.0f : 100.0f );
                            auto y = replay_signal( modes, 7811 + t );
                            // With zero feedback and an unsolved zero controller, exploration exactly inserts the
                            // supplied DM trace.
                            ReplayMatrix suppliedDm = 0.1f * replay_signal( modes, 1783 + t );
                            const auto   a = original->command( y, suppliedDm ), b = restored->command( y, suppliedDm ),
                                         c = current->command( y, suppliedDm );
                            REQUIRE( a.allFinite() );
                            REQUIRE( same_bits( a, b ) );
                            REQUIRE( same_bits( a, c ) );
                            if( t < 64 || ( t >= 80 && t < 96 ) )
                                REQUIRE( same_bits( a, suppliedDm ) );
                            const bool solve = t >= 64 && !( t >= 80 && t < 96 );
                            for( auto *model : { original.get(), restored.get(), current.get() } )
                                model->learn( solve );
                            const auto snapshot = original->snapshot();
                            REQUIRE( snapshot.m_theta.allFinite() );
                            require_snapshot( snapshot, restored->snapshot() );
                            require_snapshot( snapshot, current->snapshot() );
                            ++comparisons;
                        }
                    }
    std::cout << "Legacy exact numerical comparisons: " << comparisons
              << " frames; max output/fit/history difference=0\n";
}

/** \brief The larger copied ring changes historical modulo aliasing at unusually high mode counts; characterize it.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Legacy clone buffer capacity has an explicit high-mode compatibility limit", "[loPredCtrlAcc][legacy]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::get_command_future();
#endif
    // clang-format on
    LegacySettings s;
    s.m_modes     = 17;
    s.m_history   = 5;
    s.m_future    = 2;
    auto original = original_model( s ), restored = restored_model( s ), current = current_legacy_model( s );
    for( int t = 0; t < 48; ++t )
    {
        const auto y = ReplayMatrix::Constant( 17, 1, 1000 + t ), u = ReplayMatrix::Constant( 17, 1, 100 + t );
        REQUIRE( same_bits( original->command( y, u ), restored->command( y, u ) ) );
        REQUIRE( same_bits( current->command( y, u ), u ) );
    }
    const auto a = original->snapshot(), b = restored->snapshot(), c = current->snapshot();
    require_snapshot( a, b );
    REQUIRE( a.m_futureU( 0 ) == 146 ); // skip=17 wraps modulo the original 16-row ring to one frame.
    REQUIRE( c.m_futureU( 0 ) == 130 ); // the clone's 32-row ring retains the actual 17-frame-old command.
    REQUIRE_FALSE( same_bits( a.m_futureU, c.m_futureU ) );
}

/** \brief Run actual legacy WFS/control/publication callbacks, including integration, exploration, resets and skipped
 * copies.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Original and current legacy app outputs match with explicit configuration", "[loPredCtrlAcc][legacy][app]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processImage();
    MagAOX::app::loPredCtrlAcc::loadImageIntoStream();
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcMode();
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcTrain();
#endif
    // clang-format on
    for( bool qr : { false, true } )
    {
        auto       original = original_app(), restored = restored_app(), current = current_legacy_app();
        const auto config = replay_config( qr );
        for( auto *app : { original.get(), restored.get(), current.get() } )
        {
            REQUIRE( app->initialize( config, 4 ) == 0 );
            REQUIRE( app->control( "predict", "on" ) == 0 );
        }
        bool sawGeneratedButUnpublished = false;
        for( int t = 0; t < 192; ++t )
        {
            CAPTURE( qr, t );
            for( auto *app : { original.get(), restored.get(), current.get() } )
            {
                if( t == 24 )
                {
                    REQUIRE( app->control( "exploration_sequence", "40,0.03,10,40,0.01,100" ) == 0 );
                    REQUIRE( app->control( "learn", "on" ) == 0 );
                }
                if( t == 64 )
                    REQUIRE( app->control( "integrate", "on" ) == 0 );
                if( t == 80 )
                {
                    REQUIRE( app->control( "learn", "off" ) == 0 );
                    REQUIRE( app->control( "predict", "off" ) == 0 );
                }
                if( t == 88 )
                    REQUIRE( app->control( "predict", "on" ) == 0 );
                if( t == 96 )
                    REQUIRE( app->control( "reset_model", "on" ) == 0 );
                if( t == 120 )
                    REQUIRE( app->control( "learn", "on" ) == 0 );
                if( t == 128 )
                    REQUIRE( app->control( "integrate", "off" ) == 0 );
                if( t == 144 )
                    REQUIRE( app->control( "learn", "off" ) == 0 );
            }
            if( t == 40 || t == 100 )
            {
                REQUIRE( current->control( "ddspc_mode", t == 40 ? "standard" : "accel_assisted" ) == -1 );
                REQUIRE( current->control( "ddspc_train", "on" ) == -1 );
            }
            const auto y       = replay_signal( 4, 2431 + t );
            const bool publish = t % 5 != 0;
            const auto a = original->frame( y, publish ), b = restored->frame( y, publish ),
                       c = current->frame( y, publish );
            REQUIRE( a.m_output.allFinite() );
            REQUIRE( finite_bits( a.m_output ) );
            REQUIRE( same_bits( a.m_candidate, b.m_candidate ) );
            REQUIRE( same_bits( a.m_candidate, c.m_candidate ) );
            REQUIRE( same_bits( a.m_output, b.m_output ) );
            INFO( "Historical output=" << std::hexfloat << a.m_output.transpose()
                                       << " clone output=" << c.m_output.transpose() << " max absolute difference="
                                       << ( a.m_output - c.m_output ).cwiseAbs().maxCoeff() );
            REQUIRE( same_bits( a.m_output, c.m_output ) );
            REQUIRE( same_bits( a.m_published, b.m_published ) );
            REQUIRE( same_bits( a.m_published, c.m_published ) );
            REQUIRE( same_bits( a.m_output.bottomRows( 2 ), y.bottomRows( 2 ) ) );
            require_snapshot( a.m_model, b.m_model );
            require_snapshot( a.m_model, c.m_model );
            REQUIRE( finite_bits( a.m_model.m_theta ) );
            if( !publish && t > 0 && t < 80 && !same_bits( a.m_candidate, a.m_published.topRows( 2 ) ) )
            {
                REQUIRE( same_bits( a.m_model.m_currentU.topRows( 2 ), a.m_candidate ) );
                sawGeneratedButUnpublished = true;
            }
            const auto state = current->state_snapshot();
            REQUIRE_FALSE( state.m_shadow );
            REQUIRE_FALSE( state.m_accel );
            REQUIRE_FALSE( state.m_shadowAllocated );
        }
        REQUIRE( sawGeneratedButUnpublished );
    }
    std::cout << "Legacy exact application comparisons: 384 callbacks; output/published/fit/history difference=0\n";
}

/** \brief Characterize default differences introduced before Phases 1-3, including ignored copy_gain configuration.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Legacy clone defaults differ from the restored on-sky app", "[loPredCtrlAcc][legacy][app]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl();
#endif
    // clang-format on
    auto original = original_app(), restored = restored_app(), current = current_legacy_app();
    // Member-default characterization only: the fixture does not validate required deployed config keys.
    for( auto *app : { original.get(), restored.get(), current.get() } )
        REQUIRE( app->initialize( {}, 1 ) == 0 );
    REQUIRE( original->state_snapshot().m_qrd );
    REQUIRE( restored->state_snapshot().m_qrd );
    REQUIRE_FALSE( current->state_snapshot().m_qrd );
    REQUIRE( original->state_snapshot().m_copyGain == 0 );
    REQUIRE( current->state_snapshot().m_copyGain == 1 );
    const auto y = ReplayMatrix::Constant( 1, 1, 0.25f );
    REQUIRE( original->frame( y, true ).m_output( 0 ) == 0 );
    REQUIRE( restored->frame( y, true ).m_output( 0 ) == 0 );
    REQUIRE( current->frame( y, true ).m_output( 0 ) == 0.25f );
    auto configured                = current_legacy_app();
    auto config                    = replay_config( true );
    config["parameters.copy_gain"] = "9";
    REQUIRE( configured->initialize( config, 2 ) == 0 );
    REQUIRE( configured->state_snapshot().m_copyGain == -0.25f ); // Existing code reads parameters.gain twice.
    REQUIRE_FALSE( configured->state_snapshot().m_shadowAllocated );
}

/** \brief A WFS callback is the only software clock; prediction-off publication does not advance model histories.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Legacy timing is callback ordered and independent of publication acknowledgement",
           "[loPredCtrlAcc][legacy][app]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processImage();
#endif
    // clang-format on
    for( auto factory : { original_app, restored_app, current_legacy_app } )
    {
        auto app = factory();
        REQUIRE( app->initialize( replay_config( false ), 2 ) == 0 );
        REQUIRE( app->control( "predict", "on" ) == 0 );
        LegacyAppFrame before;
        for( int t = 0; t < 12; ++t )
            before = app->frame( replay_signal( 2, 6503 + t ), true );
        REQUIRE( app->control( "predict", "off" ) == 0 );
        for( int t = 0; t < 5; ++t )
        {
            const auto after = app->frame( replay_signal( 2, 8101 + t ), true );
            require_snapshot( before.m_model, after.m_model );
            REQUIRE_FALSE( same_bits( before.m_published, after.m_published ) );
        }
    }
}

/** \brief Restoring a different feedback gain exposes a reset difference in the clone without changing either
 * implementation.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Legacy gain restored from disk has an explicit reset compatibility difference", "[loPredCtrlAcc][legacy]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::load_state();
    DDSPC::PredictiveController::reset();
#endif
    // clang-format on
    LegacyTempDir  directory;
    const auto     artifact = ( directory.m_path / "same-dimension-state" ).string();
    LegacySettings constructed;
    constructed.m_history = constructed.m_future = 1;
    constructed.m_gain                           = 0.125f;
    auto written                                 = constructed;
    written.m_gain                               = -0.5f;
    original_model( written )->save( artifact );
    auto               original = original_model( constructed ), restored = restored_model( constructed ),
                       current = current_legacy_model( constructed );
    const ReplayMatrix y = ReplayMatrix::Ones( 1, 1 ), noise = ReplayMatrix::Zero( 1, 1 );
    for( auto *model : { original.get(), restored.get(), current.get() } )
    {
        model->load( artifact );
        REQUIRE( model->command( y, noise )( 0 ) ==
                 0.125f ); // All still have the constructed feedback matrix immediately after load.
        model->reset( false );
    }
    REQUIRE( original->command( y, noise )( 0 ) == 0.125f );
    REQUIRE( restored->command( y, noise )( 0 ) == 0.125f );
    REQUIRE( current->command( y, noise )( 0 ) == -0.5f ); // Current reset rebuilds feedback from the restored gain.
}

} // namespace libXWCTest::loPredCtrlAccTest
