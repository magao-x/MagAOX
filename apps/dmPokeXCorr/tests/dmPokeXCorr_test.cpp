/** \file dmPokeXCorr_test.cpp
 * \brief Catch2 tests for the dmPokeXCorr app.
 *
 * \ingroup dmPokeXCorr_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmPokeXCorr.hpp"

using namespace MagAOX::app;

/// \cond
namespace MagAOX
{
namespace app
{

// Test harness with access to protected dmPokeXCorr members.
class dmPokeXCorr_test
{
  public:
    dmPokeXCorr m_app;

    void maxDelta( float md )
    {
        m_app.m_maxDelta = md;
    }

    bool measurementInRange( float deltaX, float deltaY )
    {
        return m_app.measurementInRange( deltaX, deltaY );
    }

    void minXCorr( float mx )
    {
        m_app.m_minXCorr = mx;
    }

    bool peakValid( float normPeak )
    {
        return m_app.peakValid( normPeak );
    }
};

} // namespace app
} // namespace MagAOX
/// \endcond

namespace libXWCTest
{

/** \defgroup dmPokeXCorr_unit_test dmPokeXCorr Unit Tests
 * \brief Unit tests for the dmPokeXCorr application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `dmPokeXCorr` unit tests.
/** \ingroup dmPokeXCorr_unit_test
 */
namespace dmPokeXCorrTest
{

/// Verify the placeholder dmPokeXCorr test harness instantiates the app cleanly.
/**
 * \ingroup dmPokeXCorr_unit_test
 */
TEST_CASE( "dmPokeXCorr placeholder harness instantiates the app", "[dmPokeXCorr]" )
{
    // clang-format off
    #ifdef DMPOKEXCORR_TEST_DOXYGEN_REF
    dmPokeXCorr();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        dmPokeXCorr app;

        REQUIRE( true );
    }
}

/// Verify that out-of-range delta measurements are detected.
/**
 * \ingroup dmPokeXCorr_unit_test
 */
TEST_CASE( "dmPokeXCorr rejects deltas larger than maxDelta", "[dmPokeXCorr]" )
{
    // clang-format off
    #ifdef DMPOKEXCORR_TEST_DOXYGEN_REF
    dev::dmPokeWFS<dmPokeXCorr>::measurementInRange( 0, 0 );
    #endif
    // clang-format on

    dmPokeXCorr_test harness;

    SECTION( "default maxDelta is 10 pixels" )
    {
        REQUIRE( harness.measurementInRange( 0, 0 ) );
        REQUIRE( harness.measurementInRange( 10, -10 ) );
        REQUIRE_FALSE( harness.measurementInRange( 10.5, 0 ) );
        REQUIRE_FALSE( harness.measurementInRange( 0, -10.5 ) );
    }

    SECTION( "maxDelta is configurable" )
    {
        harness.maxDelta( 2 );
        REQUIRE( harness.measurementInRange( 1.5, -2 ) );
        REQUIRE_FALSE( harness.measurementInRange( 2.5, 0 ) );
    }

    SECTION( "maxDelta <= 0 disables the check" )
    {
        harness.maxDelta( 0 );
        REQUIRE( harness.measurementInRange( 1000, -1000 ) );
    }
}

/// Verify that low normalized cross-correlation peaks are detected.
/**
 * \ingroup dmPokeXCorr_unit_test
 */
TEST_CASE( "dmPokeXCorr rejects peaks below minPeak", "[dmPokeXCorr]" )
{
    // clang-format off
    #ifdef DMPOKEXCORR_TEST_DOXYGEN_REF
    dmPokeXCorr::peakValid( 0 );
    #endif
    // clang-format on

    dmPokeXCorr_test harness;

    SECTION( "default minPeak is 0.3" )
    {
        REQUIRE( harness.peakValid( 0.68 ) );
        REQUIRE( harness.peakValid( 0.3 ) );
        REQUIRE_FALSE( harness.peakValid( 0.1 ) );
        REQUIRE_FALSE( harness.peakValid( -0.5 ) );
    }

    SECTION( "minPeak <= 0 disables the check" )
    {
        harness.minXCorr( 0 );
        REQUIRE( harness.peakValid( -1 ) );
    }
}

} // namespace dmPokeXCorrTest

} // namespace libXWCTest
