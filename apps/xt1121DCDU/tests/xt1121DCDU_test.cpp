/** \file xt1121DCDU_test.cpp
 * \brief Catch2 tests for the xt1121DCDU app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xt1121DCDU_files
 */

#include "../../../tests/testXWC.hpp"

#include "../xt1121DCDU.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup xt1121DCDU_unit_test xt1121DCDU Unit Tests
 * \brief Unit tests for the xt1121DCDU application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `xt1121DCDU` unit tests.
/** \ingroup xt1121DCDU_unit_test
 */
namespace xt1121DCDUTest
{

/// Verify the placeholder xt1121DCDU test harness instantiates the app cleanly.
/**
 * \ingroup xt1121DCDU_unit_test
 */
TEST_CASE( "xt1121DCDU placeholder harness instantiates the app", "[xt1121DCDU]" )
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        xt1121DCDU app;

        REQUIRE( true );
    }
}

} // namespace xt1121DCDUTest

} // namespace libXWCTest
