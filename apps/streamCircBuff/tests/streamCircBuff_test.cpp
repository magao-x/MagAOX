/** \file streamCircBuff_test.cpp
 * \brief Catch2 tests for the streamCircBuff app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup streamCircBuff_files
 */

#include "../../../tests/testXWC.hpp"

#include "../streamCircBuff.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup streamCircBuff_unit_test streamCircBuff Unit Tests
 * \brief Unit tests for the streamCircBuff application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `streamCircBuff` unit tests.
/** \ingroup streamCircBuff_unit_test
 */
namespace streamCircBuffTest
{

/// Verify the placeholder streamCircBuff test harness instantiates the app cleanly.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff placeholder harness instantiates the app", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    streamCircBuff();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        streamCircBuff app;

        REQUIRE( true );
    }
}

} // namespace streamCircBuffTest

} // namespace libXWCTest
