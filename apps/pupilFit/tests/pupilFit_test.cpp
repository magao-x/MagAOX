/** \file pupilFit_test.cpp
 * \brief Catch2 tests for the pupilFit app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup pupilFit_files
 */

#include "../../../tests/testXWC.hpp"

#include "../pupilFit.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup pupilFit_unit_test pupilFit Unit Tests
 * \brief Unit tests for the pupilFit application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `pupilFit` unit tests.
/** \ingroup pupilFit_unit_test
 */
namespace pupilFitTest
{

/// Verify the placeholder pupilFit test harness instantiates the app cleanly.
/**
 * \ingroup pupilFit_unit_test
 */
TEST_CASE( "pupilFit placeholder harness instantiates the app", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        pupilFit app;

        REQUIRE( true );
    }
}

} // namespace pupilFitTest

} // namespace libXWCTest
