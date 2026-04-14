/** \file alignLoop_test.cpp
 * \brief Catch2 tests for the alignLoop app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup alignLoop_files
 */

#include "../../../tests/testXWC.hpp"

#include "../alignLoop.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup alignLoop_unit_test alignLoop Unit Tests
 * \brief Unit tests for the alignLoop application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `alignLoop` unit tests.
/** \ingroup alignLoop_unit_test
 */
namespace alignLoopTest
{

/// Verify the placeholder alignLoop test harness instantiates the app cleanly.
/**
 * \ingroup alignLoop_unit_test
 */
TEST_CASE( "alignLoop placeholder harness instantiates the app", "[alignLoop]" )
{
    // clang-format off
    #ifdef ALIGNLOOP_TEST_DOXYGEN_REF
    alignLoop();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        alignLoop app;

        REQUIRE( true );
    }
}

} // namespace alignLoopTest

} // namespace libXWCTest
