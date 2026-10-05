/** \file mzmqServer_test.cpp
 * \brief Catch2 tests for the mzmqServer app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup mzmqServer_files
 */

#include "../../../tests/testXWC.hpp"

#include "../mzmqServer.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup mzmqServer_unit_test mzmqServer Unit Tests
 * \brief Unit tests for the mzmqServer application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `mzmqServer` unit tests.
/** \ingroup mzmqServer_unit_test
 */
namespace mzmqServerTest
{

/// Verify the placeholder mzmqServer test harness instantiates the app cleanly.
/**
 * \ingroup mzmqServer_unit_test
 */
TEST_CASE( "mzmqServer placeholder harness instantiates the app", "[mzmqServer]" )
{
    // clang-format off
    #ifdef MZMQSERVER_TEST_DOXYGEN_REF
    mzmqServer();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        mzmqServer app;

        REQUIRE( true );
    }
}

} // namespace mzmqServerTest

} // namespace libXWCTest
