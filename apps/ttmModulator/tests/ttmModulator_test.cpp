/** \file ttmModulator_test.cpp
 * \brief Catch2 tests for the ttmModulator app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup ttmModulator_files
 */

#include "../../../tests/testXWC.hpp"

#include "../ttmModulator.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup ttmModulator_unit_test ttmModulator Unit Tests
 * \brief Unit tests for the ttmModulator application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `ttmModulator` unit tests.
/** \ingroup ttmModulator_unit_test
 */
namespace ttmModulatorTest
{

/// Verify the placeholder ttmModulator test harness instantiates the app cleanly.
/**
 * \ingroup ttmModulator_unit_test
 */
TEST_CASE( "ttmModulator placeholder harness instantiates the app", "[ttmModulator]" )
{
    // clang-format off
    #ifdef TTMMODULATOR_TEST_DOXYGEN_REF
    ttmModulator();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        ttmModulator app;

        REQUIRE( true );
    }
}

} // namespace ttmModulatorTest

} // namespace libXWCTest
