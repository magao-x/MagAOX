/** \file filterWheelCtrl_test.cpp
 * \brief Catch2 tests for the filterWheelCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup filterWheelCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../filterWheelCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup filterWheelCtrl_unit_test filterWheelCtrl Unit Tests
 * \brief Unit tests for the filterWheelCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `filterWheelCtrl` unit tests.
/** \ingroup filterWheelCtrl_unit_test
 */
namespace filterWheelCtrlTest
{

/// Verify the placeholder filterWheelCtrl test harness instantiates the app cleanly.
/**
 * \ingroup filterWheelCtrl_unit_test
 */
TEST_CASE( "filterWheelCtrl placeholder harness instantiates the app", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        filterWheelCtrl app;

        REQUIRE( true );
    }
}

} // namespace filterWheelCtrlTest

} // namespace libXWCTest
