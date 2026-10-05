/** \file irisaoCtrl_test.cpp
 * \brief Catch2 tests for the irisaoCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup irisaoCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../irisaoCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup irisaoCtrl_unit_test irisaoCtrl Unit Tests
 * \brief Unit tests for the irisaoCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `irisaoCtrl` unit tests.
/** \ingroup irisaoCtrl_unit_test
 */
namespace irisaoCtrlTest
{

/// Verify the placeholder irisaoCtrl test harness instantiates the app cleanly.
/**
 * \ingroup irisaoCtrl_unit_test
 */
TEST_CASE( "irisaoCtrl placeholder harness instantiates the app", "[irisaoCtrl]" )
{
    // clang-format off
    #ifdef IRISAOCTRL_TEST_DOXYGEN_REF
    irisaoCtrl();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        irisaoCtrl app;

        REQUIRE( true );
    }
}

} // namespace irisaoCtrlTest

} // namespace libXWCTest
