/** \file dmPokeCenter_test.cpp
 * \brief Catch2 tests for the dmPokeCenter app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup dmPokeCenter_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmPokeCenter.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup dmPokeCenter_unit_test dmPokeCenter Unit Tests
 * \brief Unit tests for the dmPokeCenter application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `dmPokeCenter` unit tests.
/** \ingroup dmPokeCenter_unit_test
 */
namespace dmPokeCenterTest
{

/// Verify the placeholder dmPokeCenter test harness instantiates the app cleanly.
/**
 * \ingroup dmPokeCenter_unit_test
 */
TEST_CASE( "dmPokeCenter placeholder harness instantiates the app", "[dmPokeCenter]" )
{
    // clang-format off
    #ifdef DMPOKECENTER_TEST_DOXYGEN_REF
    dmPokeCenter();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        dmPokeCenter app;

        REQUIRE( true );
    }
}

} // namespace dmPokeCenterTest

} // namespace libXWCTest
