/** \file koolanceCtrl_test.cpp
 * \brief Catch2 tests for the koolanceCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup koolanceCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../koolanceCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup koolanceCtrl_unit_test koolanceCtrl Unit Tests
 * \brief Unit tests for the koolanceCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `koolanceCtrl` unit tests.
/** \ingroup koolanceCtrl_unit_test
 */
namespace koolanceCtrlTest
{

/// Verify the placeholder koolanceCtrl test harness instantiates the app cleanly.
/**
 * \ingroup koolanceCtrl_unit_test
 */
TEST_CASE( "koolanceCtrl placeholder harness instantiates the app", "[koolanceCtrl]" )
{
    // clang-format off
    #ifdef KOOLANCECTRL_TEST_DOXYGEN_REF
    koolanceCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        koolanceCtrl app;

        REQUIRE( true );
    }
}

} // namespace koolanceCtrlTest

} // namespace libXWCTest
