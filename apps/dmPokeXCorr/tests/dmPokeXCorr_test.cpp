/** \file dmPokeXCorr_test.cpp
 * \brief Catch2 tests for the dmPokeXCorr app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup dmPokeXCorr_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmPokeXCorr.hpp"

using namespace MagAOX::app;

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

} // namespace dmPokeXCorrTest

} // namespace libXWCTest
