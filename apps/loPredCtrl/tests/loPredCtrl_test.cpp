/** \file loPredCtrl_test.cpp
 * \brief Catch2 tests for the loPredCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup loPredCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../loPredCtrl.hpp"
#include "../testPredCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup loPredCtrl_unit_test loPredCtrl Unit Tests
 * \brief Unit tests for the loPredCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `loPredCtrl` unit tests.
/** \ingroup loPredCtrl_unit_test
 */
namespace loPredCtrlTest
{

/// Verify the placeholder loPredCtrl test harness instantiates the app cleanly.
/**
 * \ingroup loPredCtrl_unit_test
 */
TEST_CASE( "loPredCtrl placeholder harness instantiates the app", "[loPredCtrl]" )
{
    // clang-format off
    #ifdef LOPREDCTRL_TEST_DOXYGEN_REF
    loPredCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        loPredCtrl app;

        REQUIRE( true );
    }
}

} // namespace loPredCtrlTest

} // namespace libXWCTest
