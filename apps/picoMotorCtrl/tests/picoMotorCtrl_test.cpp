/** \file picoMotorCtrl_test.cpp
 * \brief Catch2 tests for the picoMotorCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup picoMotorCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../picoMotorCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup picoMotorCtrl_unit_test picoMotorCtrl Unit Tests
 * \brief Unit tests for the picoMotorCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `picoMotorCtrl` unit tests.
/** \ingroup picoMotorCtrl_unit_test
 */
namespace picoMotorCtrlTest
{

/// Verify the placeholder picoMotorCtrl test harness instantiates the app cleanly.
/**
 * \ingroup picoMotorCtrl_unit_test
 */
TEST_CASE( "picoMotorCtrl placeholder harness instantiates the app", "[picoMotorCtrl]" )
{
    // clang-format off
    #ifdef PICOMOTORCTRL_TEST_DOXYGEN_REF
    picoMotorCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        picoMotorCtrl app;

        REQUIRE( true );
    }
}

} // namespace picoMotorCtrlTest

} // namespace libXWCTest
