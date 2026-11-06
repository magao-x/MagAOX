/** \file ddspc_qr_compatibility_test.cpp
 * \brief Exact historical QR comparisons isolating vectorized rotations from controller/history construction.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "legacy_replay.hpp"
#include "../qrd_rls.hpp"
#include <cstring>
#include <random>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
namespace
{
/// Compare every float bit, including signed zeros; no numerical tolerance accepts drift.
bool qr_same_bits( const ReplayMatrix &a /**< [in] Historical result. */,
                   const ReplayMatrix &b /**< [in] Current result. */ )
{
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) ) == 0;
}
} // namespace
/// \endcond

/// The fourth unrolled column reproduces the GCC complex-vector contraction regression before the correction.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Legacy Givens rotation is bit exact with the historical kernel", "[loPredCtrlAcc][legacy][qr]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::QRDRecursiveLeastSquares::apply_givens();
#endif
    // clang-format on
    auto                            old = original_qr( 3, 4, 1, 1 );
    DDSPC::QRDRecursiveLeastSquares current( 3, 4, 1, 1 ); // Default remains Legacy, without structural zeroing.
    DDSPC::Matrix                   workspace = DDSPC::Matrix::Zero( 5, 4 );
    workspace.row( 0 ) << .731f, .123f, -.578f, -.578f;
    workspace.row( 1 ) << -.27f, .456f, .382f, .382f;
    const auto expected = old->rotate( workspace );
    current.A_          = workspace;
    current.apply_givens();
    REQUIRE( qr_same_bits( expected, current.A_ ) );
    // Causal opt-in still enforces the exact structural invariant rather than reproducing rounding remnants.
    DDSPC::QRDRecursiveLeastSquares causal( 3, 4, 1, 1, true );
    causal.A_ = workspace;
    causal.apply_givens();
    REQUIRE( causal.A_( 1, 0 ) == 0 );
    REQUIRE( causal.A_.allFinite() );
}

/// Explicit sparse-startup and dense regressors compare factors, gain and coefficients after every update.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Legacy QR identification is bit exact without controller history code", "[loPredCtrlAcc][legacy][qr]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::QRDRecursiveLeastSquares::update();
#endif
    // clang-format on
    for( int features : { 4, 10 } )
        for( float gamma : { 1.f, .97f } )
            for( float prior : { 1.f, .02f } )
            {
                auto                            old = original_qr( 3, features, gamma, prior );
                DDSPC::QRDRecursiveLeastSquares current( 3, features, gamma, prior, false );
                std::mt19937                    rng( 681 );
                DDSPC::Matrix                   x( features, 1 ), y( 3, 1 );
                for( int t = 0; t < 192; ++t )
                {
                    CAPTURE( features, gamma, prior, t );
                    for( int j = 0; j < features; ++j )
                        x( j ) = t < j ? 0.f : ( int( rng() % 2001 ) - 1000 ) / 1000.f;
                    for( int j = 0; j < y.size(); ++j )
                        y( j ) = ( int( rng() % 2001 ) - 1000 ) / 1000.f;
                    const auto expected = old->update( x, y );
                    current.update( &x, &y );
                    REQUIRE( qr_same_bits( expected.m_factor, current.R_ ) );
                    REQUIRE( qr_same_bits( expected.m_augmented, current.A_ ) );
                    REQUIRE( qr_same_bits( expected.m_gain, current.g ) );
                    REQUIRE( qr_same_bits( expected.m_theta, current.prediction_matrix ) );
                }
            }
}
} // namespace libXWCTest::loPredCtrlAccTest
