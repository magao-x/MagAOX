/** \file flipperCtrl_test.cpp
 * \brief Catch2 tests for the flipperCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup flipperCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../flipperCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup flipperCtrl_unit_test flipperCtrl Unit Tests
 * \brief Unit tests for the flipperCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `flipperCtrl` unit tests.
/** \ingroup flipperCtrl_unit_test
 */
namespace flipperCtrlTest
{

/// Verify the placeholder flipperCtrl test harness instantiates the app cleanly.
/**
 * \ingroup flipperCtrl_unit_test
 */
TEST_CASE( "flipperCtrl placeholder harness instantiates the app", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        flipperCtrl app;

        REQUIRE( true );
    }
}

} // namespace flipperCtrlTest

} // namespace libXWCTest
