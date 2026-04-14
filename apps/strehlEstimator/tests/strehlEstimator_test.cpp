/** \file strehlEstimator_test.cpp
 * \brief Catch2 tests for the strehlEstimator app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup strehlEstimator_files
 */

#include "../../../tests/testXWC.hpp"

#include "../strehlEstimator.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup strehlEstimator_unit_test strehlEstimator Unit Tests
 * \brief Unit tests for the strehlEstimator application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `strehlEstimator` unit tests.
/** \ingroup strehlEstimator_unit_test
 */
namespace strehlEstimatorTest
{

/// Verify the placeholder strehlEstimator test harness instantiates the app cleanly.
/**
 * \ingroup strehlEstimator_unit_test
 */
TEST_CASE( "strehlEstimator placeholder harness instantiates the app", "[strehlEstimator]" )
{
    // clang-format off
    #ifdef STREHLESTIMATOR_TEST_DOXYGEN_REF
    strehlEstimator();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        strehlEstimator app;

        REQUIRE( true );
    }
}

} // namespace strehlEstimatorTest

} // namespace libXWCTest
