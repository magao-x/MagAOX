/** \file dmMode_test.cpp
 * \brief Catch2 tests for the dmMode app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup dmMode_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmMode.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup dmMode_unit_test dmMode Unit Tests
 * \brief Unit tests for the dmMode application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `dmMode` unit tests.
/** \ingroup dmMode_unit_test
 */
namespace dmModeTest
{

/// Verify the placeholder dmMode test harness instantiates the app cleanly.
/**
 * \ingroup dmMode_unit_test
 */
TEST_CASE( "dmMode placeholder harness instantiates the app", "[dmMode]" )
{
    // clang-format off
    #ifdef DMMODE_TEST_DOXYGEN_REF
    dmMode();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        dmMode app;

        REQUIRE( true );
    }
}

} // namespace dmModeTest

} // namespace libXWCTest
