/** \file xt1121Ctrl_test.cpp
 * \brief Catch2 tests for the xt1121Ctrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xt1121Ctrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../xt1121Ctrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup xt1121Ctrl_unit_test xt1121Ctrl Unit Tests
 * \brief Unit tests for the xt1121Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `xt1121Ctrl` unit tests.
/** \ingroup xt1121Ctrl_unit_test
 */
namespace xt1121CtrlTest
{

/// Verify the placeholder xt1121Ctrl test harness instantiates the app cleanly.
/**
 * \ingroup xt1121Ctrl_unit_test
 */
TEST_CASE( "xt1121Ctrl placeholder harness instantiates the app", "[xt1121Ctrl]" )
{
    // clang-format off
    #ifdef XT1121CTRL_TEST_DOXYGEN_REF
    xt1121Ctrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        xt1121Ctrl app;

        REQUIRE( true );
    }
}

} // namespace xt1121CtrlTest

} // namespace libXWCTest
