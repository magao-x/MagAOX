/** \file w2tcsOffloader_test.cpp
 * \brief Catch2 tests for the w2tcsOffloader app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup w2tcsOffloader_files
 */

#include "../../../tests/testXWC.hpp"

#include "../w2tcsOffloader.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup w2tcsOffloader_unit_test w2tcsOffloader Unit Tests
 * \brief Unit tests for the w2tcsOffloader application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `w2tcsOffloader` unit tests.
/** \ingroup w2tcsOffloader_unit_test
 */
namespace w2tcsOffloaderTest
{

/// Verify the placeholder w2tcsOffloader test harness instantiates the app cleanly.
/**
 * \ingroup w2tcsOffloader_unit_test
 */
TEST_CASE( "w2tcsOffloader placeholder harness instantiates the app", "[w2tcsOffloader]" )
{
    // clang-format off
    #ifdef W2TCSOFFLOADER_TEST_DOXYGEN_REF
    w2tcsOffloader();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        w2tcsOffloader app;

        REQUIRE( true );
    }
}

} // namespace w2tcsOffloaderTest

} // namespace libXWCTest
