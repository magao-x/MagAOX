/** \file t2wOffloader_test.cpp
 * \brief Catch2 tests for the t2wOffloader app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup t2wOffloader_files
 */

#include "../../../tests/testXWC.hpp"

#include "../t2wOffloader.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup t2wOffloader_unit_test t2wOffloader Unit Tests
 * \brief Unit tests for the t2wOffloader application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `t2wOffloader` unit tests.
/** \ingroup t2wOffloader_unit_test
 */
namespace t2wOffloaderTest
{

/// Verify the placeholder t2wOffloader test harness instantiates the app cleanly.
/**
 * \ingroup t2wOffloader_unit_test
 */
TEST_CASE( "t2wOffloader placeholder harness instantiates the app", "[t2wOffloader]" )
{
    // clang-format off
    #ifdef T2WOFFLOADER_TEST_DOXYGEN_REF
    t2wOffloader();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        t2wOffloader app;

        REQUIRE( true );
    }
}

} // namespace t2wOffloaderTest

} // namespace libXWCTest
