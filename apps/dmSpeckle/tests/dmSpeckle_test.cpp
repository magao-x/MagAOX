/** \file dmSpeckle_test.cpp
 * \brief Catch2 tests for the dmSpeckle app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup dmSpeckle_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmSpeckle.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup dmSpeckle_unit_test dmSpeckle Unit Tests
 * \brief Unit tests for the dmSpeckle application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `dmSpeckle` unit tests.
/** \ingroup dmSpeckle_unit_test
 */
namespace dmSpeckleTest
{

/// Verify the placeholder dmSpeckle test harness instantiates the app cleanly.
/**
 * \ingroup dmSpeckle_unit_test
 */
TEST_CASE( "dmSpeckle placeholder harness instantiates the app", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        dmSpeckle app;

        REQUIRE( true );
    }
}

} // namespace dmSpeckleTest

} // namespace libXWCTest
