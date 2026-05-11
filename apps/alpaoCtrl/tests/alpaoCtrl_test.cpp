/** \file alpaoCtrl_test.cpp
 * \brief Catch2 tests for the alpaoCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup alpaoCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../alpaoCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup alpaoCtrl_unit_test alpaoCtrl Unit Tests
 * \brief Unit tests for the alpaoCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `alpaoCtrl` unit tests.
/** \ingroup alpaoCtrl_unit_test
 */
namespace alpaoCtrlTest
{

/// Verify the placeholder alpaoCtrl test harness instantiates the app cleanly.
/**
 * \ingroup alpaoCtrl_unit_test
 */
TEST_CASE( "alpaoCtrl placeholder harness instantiates the app", "[alpaoCtrl]" )
{
    // clang-format off
    #ifdef ALPAOCTRL_TEST_DOXYGEN_REF
    alpaoCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        alpaoCtrl app;

        REQUIRE( true );
    }
}

} // namespace alpaoCtrlTest

} // namespace libXWCTest
