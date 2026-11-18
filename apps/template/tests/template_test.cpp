/** \file template_test.cpp
 * \brief Catch2 tests for the template app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup template_files
 */

#include "../../../tests/testXWC.hpp"

#include "../template.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup template_unit_test template Unit Tests
 * \brief Unit tests for the template application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `template` unit tests.
/** \ingroup template_unit_test
 */
namespace templateTest
{

/// Verify the placeholder template test harness instantiates the app cleanly.
/**
 * \ingroup template_unit_test
 */
TEST_CASE( "template placeholder harness instantiates the app", "[template]" )
{
    // clang-format off
    #ifdef TEMPLATE_TEST_DOXYGEN_REF
    templateMagAOX();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        templateMagAOX app;

        REQUIRE( true );
    }
}

} // namespace templateTest

} // namespace libXWCTest
