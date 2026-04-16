/** \file refRMS_test.cpp
 * \brief Catch2 tests for the refRMS app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup refRMS_files
 */

#include "../../../tests/testXWC.hpp"

#include "../refRMS.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup refRMS_unit_test refRMS Unit Tests
 * \brief Unit tests for the refRMS application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `refRMS` unit tests.
/** \ingroup refRMS_unit_test
 */
namespace refRMSTest
{

/// Verify the placeholder refRMS test harness instantiates the app cleanly.
/**
 * \ingroup refRMS_unit_test
 */
TEST_CASE( "refRMS placeholder harness instantiates the app", "[refRMS]" )
{
    // clang-format off
    #ifdef REFRMS_TEST_DOXYGEN_REF
    refRMS();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        refRMS app;

        REQUIRE( true );
    }
}

} // namespace refRMSTest

} // namespace libXWCTest
