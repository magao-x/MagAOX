/** \file magAOXMaths_test.cpp
 * \brief Catch2 tests for the magAOXMaths app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup magAOXMaths_files
 */

#include "../../../tests/testXWC.hpp"

#include "../magAOXMaths.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup magAOXMaths_unit_test magAOXMaths Unit Tests
 * \brief Unit tests for the magAOXMaths application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `magAOXMaths` unit tests.
/** \ingroup magAOXMaths_unit_test
 */
namespace magAOXMathsTest
{

/// Verify the placeholder magAOXMaths test harness instantiates the app cleanly.
/**
 * \ingroup magAOXMaths_unit_test
 */
TEST_CASE( "magAOXMaths placeholder harness instantiates the app", "[magAOXMaths]" )
{
    // clang-format off
    #ifdef MAGAOXMATHS_TEST_DOXYGEN_REF
    magAOXMaths();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        magAOXMaths app;

        REQUIRE( true );
    }
}

} // namespace magAOXMathsTest

} // namespace libXWCTest
