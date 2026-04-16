/** \file indiTSAccumulator_test.cpp
 * \brief Catch2 tests for the indiTSAccumulator app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup indiTSAccumulator_files
 */

#include "../../../tests/testXWC.hpp"

#include "../indiTSAccumulator.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup indiTSAccumulator_unit_test indiTSAccumulator Unit Tests
 * \brief Unit tests for the indiTSAccumulator application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `indiTSAccumulator` unit tests.
/** \ingroup indiTSAccumulator_unit_test
 */
namespace indiTSAccumulatorTest
{

/// Verify the placeholder indiTSAccumulator test harness instantiates the app cleanly.
/**
 * \ingroup indiTSAccumulator_unit_test
 */
TEST_CASE( "indiTSAccumulator placeholder harness instantiates the app", "[indiTSAccumulator]" )
{
    // clang-format off
    #ifdef INDITSACCUMULATOR_TEST_DOXYGEN_REF
    indiTSAccumulator();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        indiTSAccumulator app;

        REQUIRE( true );
    }
}

} // namespace indiTSAccumulatorTest

} // namespace libXWCTest
