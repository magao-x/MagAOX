/** \file sparkleClock_test.cpp
 * \brief Catch2 tests for the sparkleClock app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup sparkleClock_files
 */

#include "../../../tests/testXWC.hpp"

#include "../sparkleClock.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup sparkleClock_unit_test sparkleClock Unit Tests
 * \brief Unit tests for the sparkleClock application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `sparkleClock` unit tests.
/** \ingroup sparkleClock_unit_test
 */
namespace sparkleClockTest
{

/// Verify the placeholder sparkleClock test harness instantiates the app cleanly.
/**
 * \ingroup sparkleClock_unit_test
 */
TEST_CASE( "sparkleClock placeholder harness instantiates the app", "[sparkleClock]" )
{
    // clang-format off
    #ifdef SPARKLECLOCK_TEST_DOXYGEN_REF
    sparkleClock();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        sparkleClock app;

        REQUIRE( true );
    }
}

} // namespace sparkleClockTest

} // namespace libXWCTest
