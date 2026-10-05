/** \file modalFilter_test.cpp
 * \brief Catch2 tests for the modalFilter app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup modalFilter_files
 */

#include "../../../tests/testXWC.hpp"

#include "../modalFilter.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup modalFilter_unit_test modalFilter Unit Tests
 * \brief Unit tests for the modalFilter application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `modalFilter` unit tests.
/** \ingroup modalFilter_unit_test
 */
namespace modalFilterTest
{

/// Verify the placeholder modalFilter test harness instantiates the app cleanly.
/**
 * \ingroup modalFilter_unit_test
 */
TEST_CASE( "modalFilter placeholder harness instantiates the app", "[modalFilter]" )
{
    // clang-format off
    #ifdef MODALFILTER_TEST_DOXYGEN_REF
    modalFilter();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        modalFilter app;

        REQUIRE( true );
    }
}

} // namespace modalFilterTest

} // namespace libXWCTest
