/** \file pi335Ctrl_test.cpp
 * \brief Catch2 tests for the pi335Ctrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup pi335Ctrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../pi335Ctrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup pi335Ctrl_unit_test pi335Ctrl Unit Tests
 * \brief Unit tests for the pi335Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `pi335Ctrl` unit tests.
/** \ingroup pi335Ctrl_unit_test
 */
namespace pi335CtrlTest
{

/// Verify the placeholder pi335Ctrl test harness instantiates the app cleanly.
/**
 * \ingroup pi335Ctrl_unit_test
 */
TEST_CASE( "pi335Ctrl placeholder harness instantiates the app", "[pi335Ctrl]" )
{
    // clang-format off
    #ifdef PI335CTRL_TEST_DOXYGEN_REF
    pi335Ctrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        pi335Ctrl app;

        REQUIRE( true );
    }
}

} // namespace pi335CtrlTest

} // namespace libXWCTest
