/** \file bmcCtrl_test.cpp
 * \brief Catch2 tests for the bmcCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup bmcCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../bmcCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup bmcCtrl_unit_test bmcCtrl Unit Tests
 * \brief Unit tests for the bmcCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `bmcCtrl` unit tests.
/** \ingroup bmcCtrl_unit_test
 */
namespace bmcCtrlTest
{

/// Verify the placeholder bmcCtrl test harness instantiates the app cleanly.
/**
 * \ingroup bmcCtrl_unit_test
 */
TEST_CASE( "bmcCtrl placeholder harness instantiates the app", "[bmcCtrl]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    bmcCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        bmcCtrl app;

        REQUIRE( true );
    }
}

} // namespace bmcCtrlTest

} // namespace libXWCTest
