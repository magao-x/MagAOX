/** \file pwfsSlopeCalc_test.cpp
 * \brief Catch2 tests for the pwfsSlopeCalc app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup pwfsSlopeCalc_files
 */

#include "../../../tests/testXWC.hpp"

#include "../pwfsSlopeCalc.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup pwfsSlopeCalc_unit_test pwfsSlopeCalc Unit Tests
 * \brief Unit tests for the pwfsSlopeCalc application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `pwfsSlopeCalc` unit tests.
/** \ingroup pwfsSlopeCalc_unit_test
 */
namespace pwfsSlopeCalcTest
{

/// Verify the placeholder pwfsSlopeCalc test harness instantiates the app cleanly.
/**
 * \ingroup pwfsSlopeCalc_unit_test
 */
TEST_CASE( "pwfsSlopeCalc placeholder harness instantiates the app", "[pwfsSlopeCalc]" )
{
    // clang-format off
    #ifdef PWFSSLOPECALC_TEST_DOXYGEN_REF
    pwfsSlopeCalc();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        pwfsSlopeCalc app;

        REQUIRE( true );
    }
}

} // namespace pwfsSlopeCalcTest

} // namespace libXWCTest
