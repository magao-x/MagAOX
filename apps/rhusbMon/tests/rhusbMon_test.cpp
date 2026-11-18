/** \file rhusbMon_test.cpp
 * \brief Catch2 tests for the rhusbMon app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup rhusbMon_files
 */

#include "../../../tests/testXWC.hpp"

#include "../rhusbMon.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup rhusbMon_unit_test rhusbMon Unit Tests
 * \brief Unit tests for the rhusbMon application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `rhusbMon` unit tests.
/** \ingroup rhusbMon_unit_test
 */
namespace rhusbMonTest
{

/// Verify the placeholder rhusbMon test harness instantiates the app cleanly.
/**
 * \ingroup rhusbMon_unit_test
 */
TEST_CASE( "rhusbMon placeholder harness instantiates the app", "[rhusbMon]" )
{
    // clang-format off
    #ifdef RHUSBMON_TEST_DOXYGEN_REF
    rhusbMon();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        rhusbMon app;

        REQUIRE( true );
    }
}

} // namespace rhusbMonTest

} // namespace libXWCTest
