/** \file hwpTracker_test.cpp
 * \brief Catch2 tests for the hwpTracker app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup hwpTracker_files
 */

#include "../../../tests/testXWC.hpp"

#include "../hwpTracker.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup hwpTracker_unit_test hwpTracker Unit Tests
 * \brief Unit tests for the hwpTracker application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `hwpTracker` unit tests.
/** \ingroup hwpTracker_unit_test
 */
namespace hwpTrackerTest
{

/// Verify the placeholder hwpTracker test harness instantiates the app cleanly.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker placeholder harness instantiates the app", "[hwpTracker]" )
{
    // clang-format off
    #ifdef HWPTRACKER_TEST_DOXYGEN_REF
    hwpTracker();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        hwpTracker app;

        REQUIRE( true );
    }
}

} // namespace hwpTrackerTest

} // namespace libXWCTest
