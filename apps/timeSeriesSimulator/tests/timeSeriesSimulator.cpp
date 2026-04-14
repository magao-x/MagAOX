/** \file timeSeriesSimulator.cpp
 * \brief Catch2 tests for the timeSeriesSimulator app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup timeSeriesSimulator_files
 */

#include "../../../tests/testXWC.hpp"

#include "../timeSeriesSimulator.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup timeSeriesSimulator_unit_test timeSeriesSimulator Unit Tests
 * \brief Unit tests for the timeSeriesSimulator application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `timeSeriesSimulator` unit tests.
/** \ingroup timeSeriesSimulator_unit_test
 */
namespace timeSeriesSimulatorTest
{

/// Verify the placeholder timeSeriesSimulator test harness instantiates the app cleanly.
/**
 * \ingroup timeSeriesSimulator_unit_test
 */
TEST_CASE( "timeSeriesSimulator placeholder harness instantiates the app", "[timeSeriesSimulator]" )
{
    // clang-format off
    #ifdef TIMESERIESSIMULATOR_TEST_DOXYGEN_REF
    timeSeriesSimulator();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        timeSeriesSimulator app;

        REQUIRE( true );
    }
}

} // namespace timeSeriesSimulatorTest

} // namespace libXWCTest
