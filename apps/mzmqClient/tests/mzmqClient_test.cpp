/** \file mzmqClient_test.cpp
 * \brief Catch2 tests for the mzmqClient app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup mzmqClient_files
 */

#include "../../../tests/testXWC.hpp"

#include "../mzmqClient.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup mzmqClient_unit_test mzmqClient Unit Tests
 * \brief Unit tests for the mzmqClient application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `mzmqClient` unit tests.
/** \ingroup mzmqClient_unit_test
 */
namespace mzmqClientTest
{

/// Verify the placeholder mzmqClient test harness instantiates the app cleanly.
/**
 * \ingroup mzmqClient_unit_test
 */
TEST_CASE( "mzmqClient placeholder harness instantiates the app", "[mzmqClient]" )
{
    // clang-format off
    #ifdef MZMQCLIENT_TEST_DOXYGEN_REF
    mzmqClient();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        mzmqClient app;

        REQUIRE( true );
    }
}

} // namespace mzmqClientTest

} // namespace libXWCTest
