/** \file dmReconMath_test.cpp
 * \brief Offline dmRecon mask, inverse and nonfinite-value regressions under production compiler flags.
 * \ingroup dmRecon_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include <algorithm>
#include <bit>
#include <limits>

#include <mx/improc/eigenImage.hpp>
#include <mx/math/eigenLapack.hpp>

/// \cond DMRECON_MATH_TEST_HARNESS
namespace dmReconMathHarness
{
int inverseFailure = 0; ///< Injected decomposition status; zero forwards to the real mxlib implementation.

/// Restore ordinary decomposition after a fault-injection section, including failed assertions.
class InverseFailure
{
  public:
    /// Select the failure status for this scope.
    explicit InverseFailure( int status /**< [in] Nonzero mxlib decomposition status. */ );

    /// Restore real decomposition.
    ~InverseFailure();
};

InverseFailure::InverseFailure( int status )
{
    inverseFailure = status;
}

InverseFailure::~InverseFailure()
{
    inverseFailure = 0;
}
} // namespace dmReconMathHarness

namespace mx
{
namespace math
{
/// Forward to real mxlib decomposition except when testing failure propagation.
int dmReconTestPseudoInverse( mx::improc::eigenImage<float> &inverse /**< [out] Calculated inverse. */,
                              float                         &condition /**< [out] Condition estimate. */,
                              int                           &rejected /**< [out] Rejected singular mode count. */,
                              mx::improc::eigenImage<float> &modes /**< [in,out] Matrix consumed by decomposition. */,
                              float                         &maxCondition /**< [in,out] Truncation control. */ )
{
    if( dmReconMathHarness::inverseFailure != 0 )
    {
        return dmReconMathHarness::inverseFailure;
    }

    return eigenPseudoInverse( inverse, condition, rejected, modes, maxCondition );
}
} // namespace math
} // namespace mx

#define eigenPseudoInverse dmReconTestPseudoInverse
#include "../dmReconMath.hpp"
#undef eigenPseudoInverse
/// \endcond

namespace libXWCTest
{
namespace dmReconTest
{

namespace math = MagAOX::app::dmReconMath;

/// \cond DMRECON_MATH_TEST_HARNESS
/// Initialize two independent modes with unequal amplitudes on asymmetric active pixels.
void independentModes( mx::improc::eigenCube<float> &modes /**< [out] Private two-mode cube. */ )
{
    modes.resize( 2, 2, 2 );
    std::fill_n( modes.data(), 8, 0.0F );
    modes.image( 0 ).data()[0] = 2;
    modes.image( 1 ).data()[2] = 4;
}
/// \endcond

/// Reject every infinity and NaN encoding while preserving ordinary finite and subnormal values.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon classifies nonfinite bits under fast math", "[dmRecon][math]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmReconMath::finite;
    MagAOX::app::dmReconMath::allFinite;
    #endif
    // clang-format on

    for( uint32_t bits : { 0U, 0x80000000U, 1U, 0x80000001U, 0x3f800000U, 0x7f7fffffU, 0xff7fffffU } )
    {
        const float value = std::bit_cast<float>( bits );
        CHECK( math::finite( value ) );
        CHECK( math::allFinite( &value, 1 ) );
    }

    for( uint32_t bits : { 0x7f800000U, 0xff800000U, 0x7fc00000U, 0xffc00000U, 0x7f800001U } )
    {
        const float value = std::bit_cast<float>( bits );
        CHECK_FALSE( math::finite( value ) );
        CHECK_FALSE( math::allFinite( &value, 1 ) );
    }

    CHECK( math::allFinite( nullptr, 0 ) );
    CHECK_FALSE( math::allFinite( nullptr, 1 ) );
}

/// Select asymmetric masks in storage order and retain the previous selection after invalid updates.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon preserves column-major mask selection", "[dmRecon][math]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmReconMath::maskIndices;
    #endif
    // clang-format on

    mx::improc::eigenImage<float> mask( 3, 3 );
    mask.setZero();
    mask( 0, 1 ) = 1;
    mask( 2, 0 ) = 1;
    mask( 2, 2 ) = 0.5F;
    mask( 1, 2 ) = -1;
    std::vector<size_t>       indices;
    const std::vector<size_t> expected{ 2, 3 };

    REQUIRE( math::maskIndices( indices, mask.data(), mask.size() ) );
    CHECK( indices == expected );
    CHECK_FALSE( math::maskIndices( indices, nullptr, 0 ) );
    CHECK( indices == expected );
    mask.setZero();
    CHECK_FALSE( math::maskIndices( indices, mask.data(), mask.size() ) );
    CHECK( indices == expected );
    mask( 0, 0 ) = 1;
    mask( 2, 2 ) = std::bit_cast<float>( 0x7fc00000U );
    CHECK_FALSE( math::maskIndices( indices, mask.data(), mask.size() ) );
    CHECK( indices == expected );
}

/// Require nonempty perfect-square response output images before converting their dimensions.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon validates square response dimensions", "[dmRecon][math]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmReconMath::squareSize;
    #endif
    // clang-format on

    uint32_t side = 7;
    CHECK_FALSE( math::squareSize( side, 0 ) );
    CHECK_FALSE( math::squareSize( side, 8 ) );
    CHECK( side == 7 );
    REQUIRE( math::squareSize( side, 9 ) );
    CHECK( side == 3 );
    REQUIRE( math::squareSize( side, size_t( 65535 ) * 65535 ) );
    CHECK( side == 65535 );
    CHECK_FALSE( math::squareSize( side, std::numeric_limits<size_t>::max() ) );
}

/// Reconstruct a known asymmetric basis with and without the existing response normalization.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon builds finite inverses using real mxlib decomposition", "[dmRecon][math]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmReconMath::buildInverse;
    mx::math::eigenPseudoInverse;
    #endif
    // clang-format on

    mx::improc::eigenCube<float> modes;
    independentModes( modes );
    mx::improc::eigenImage<float> inverse, response;
    float                         condition = -1;
    int                           rejected  = -1;
    std::string                   error;
    float                         factor   = 1;
    int                           retained = 0;

    SECTION( "direct DM modes" )
    {
    }

    SECTION( "identity response retains existing RMS scaling" )
    {
        response = Eigen::Matrix4f::Identity().array();
        factor   = std::sqrt( 2.0F );
    }

    SECTION( "singular mode truncation retains the larger independent mode" )
    {
        retained = 1;
    }

    REQUIRE( math::buildInverse( inverse, condition, rejected, modes, { 0, 2 }, response, retained, error ) == 0 );
    CHECK( error.empty() );
    REQUIRE( inverse.rows() == 2 );
    REQUIRE( inverse.cols() == 2 );
    CHECK( inverse( 0, 0 ) == Approx( retained == 0 ? 0.5F * factor : 0 ).margin( 1e-6 ) );
    CHECK( inverse( 1, 1 ) == Approx( 0.25F * factor ) );
    CHECK( inverse( 0, 1 ) == Approx( 0 ).margin( 1e-6 ) );
    CHECK( inverse( 1, 0 ) == Approx( 0 ).margin( 1e-6 ) );
    CHECK( condition == Approx( retained == 0 ? 2 : 1 ) );
    CHECK( rejected == retained );
}

/// Keep the previous inverse intact when calibration inputs or normalization are invalid.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon rejects malformed inverse inputs", "[dmRecon][math]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmReconMath::buildInverse;
    #endif
    // clang-format on

    mx::improc::eigenCube<float> modes;
    independentModes( modes );
    mx::improc::eigenImage<float> inverse( 1, 1 ), response;
    inverse( 0, 0 ) = 123;
    std::vector<size_t> indices{ 0, 2 };
    int                 retained  = 0;
    float               condition = -1;
    int                 rejected  = -1;
    std::string         error;

    SECTION( "empty mask" )
    {
        indices.clear();
    }

    SECTION( "too few active pixels for mxlib full SVD" )
    {
        indices = { 0 };
    }

    SECTION( "out-of-bounds mask" )
    {
        indices = { 0, 4 };
    }

    SECTION( "duplicate indices" )
    {
        indices = { 0, 0 };
    }

    SECTION( "negative truncation" )
    {
        retained = -1;
    }

    SECTION( "excess truncation" )
    {
        retained = 3;
    }

    SECTION( "nonfinite input mode" )
    {
        modes.data()[0] = std::bit_cast<float>( 0x7f800000U );
    }

    SECTION( "response input dimension mismatch" )
    {
        response.setOnes( 4, 3 );
    }

    SECTION( "nonsquare response output" )
    {
        response.setOnes( 3, 4 );
    }

    SECTION( "zero response normalization" )
    {
        response.setZero( 4, 4 );
    }

    SECTION( "nonfinite response" )
    {
        response.setOnes( 4, 4 );
        response( 0, 0 ) = std::bit_cast<float>( 0x7fc00000U );
    }

    SECTION( "zero singular values produce an invalid inverse" )
    {
        std::fill_n( modes.data(), 8, 0.0F );
    }

    SECTION( "finite response overflows during normalization" )
    {
        response.setConstant( 4, 4, std::numeric_limits<float>::max() );
    }

    REQUIRE( math::buildInverse( inverse, condition, rejected, modes, indices, response, retained, error ) == -1 );
    CHECK_FALSE( error.empty() );
    REQUIRE( inverse.rows() == 1 );
    REQUIRE( inverse.cols() == 1 );
    CHECK( inverse( 0, 0 ) == 123 );
}

/// Treat positive nonconvergence and negative argument errors identically without publishing a candidate.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon rejects every nonzero inverse status", "[dmRecon][math]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmReconMath::buildInverse;
    mx::math::eigenPseudoInverse;
    #endif
    // clang-format on

    const int                          status = GENERATE( -4, 1, 7 );
    dmReconMathHarness::InverseFailure failure( status );
    mx::improc::eigenCube<float>       modes;
    independentModes( modes );
    mx::improc::eigenImage<float> inverse( 1, 1 ), response;
    inverse( 0, 0 )       = 123;
    float       condition = -1;
    int         rejected  = -1;
    std::string error;

    REQUIRE( math::buildInverse( inverse, condition, rejected, modes, { 0, 2 }, response, 0, error ) == -1 );
    CHECK( error == "eigenPseudoInverse failed with status " + std::to_string( status ) );
    CHECK( condition == 1 );
    CHECK( rejected == 0 );
    REQUIRE( inverse.rows() == 1 );
    REQUIRE( inverse.cols() == 1 );
    CHECK( inverse( 0, 0 ) == 123 );
}

} // namespace dmReconTest
} // namespace libXWCTest
