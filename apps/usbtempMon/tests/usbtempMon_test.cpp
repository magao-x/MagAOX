/** \file usbtempMon_test.cpp
 * \brief Catch2 tests for the usbtempMon app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup usbtempMon_files
 */

#include "../../../tests/testXWC.hpp"

#include "../usbtempMon.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup usbtempMon_unit_test usbtempMon Unit Tests
 * \brief Unit tests for the usbtempMon application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `usbtempMon` unit tests.
/** \ingroup usbtempMon_unit_test
 */
namespace usbtempMonTest
{

/// Verify the placeholder usbtempMon test harness instantiates the app cleanly.
/**
 * \ingroup usbtempMon_unit_test
 */
TEST_CASE( "usbtempMon placeholder harness instantiates the app", "[usbtempMon]" )
{
    // clang-format off
    #ifdef USBTEMPMON_TEST_DOXYGEN_REF
    usbtempMon();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        usbtempMon app;

        REQUIRE( true );
    }
}

} // namespace usbtempMonTest

} // namespace libXWCTest
