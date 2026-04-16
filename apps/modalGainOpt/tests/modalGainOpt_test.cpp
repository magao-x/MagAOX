/** \file modalGainOpt_test.cpp
 * \brief Catch2 tests for the modalGainOpt app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup modalGainOpt_files
 */

#include "../../../tests/testXWC.hpp"

#include "../modalGainOpt.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup modalGainOpt_unit_test modalGainOpt Unit Tests
 * \brief Unit tests for the modalGainOpt application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `modalGainOpt` unit tests.
/** \ingroup modalGainOpt_unit_test
 */
namespace modalGainOptTest
{

/// Verify the placeholder modalGainOpt test harness instantiates the app cleanly.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt placeholder harness instantiates the app", "[modalGainOpt]" )
{
    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        modalGainOpt app;

        REQUIRE( true );
    }
}

} // namespace modalGainOptTest

} // namespace libXWCTest
