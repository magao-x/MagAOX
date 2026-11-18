/** \file siglentSDG_parsers_test.cpp
 * \brief Catch2 tests for the siglentSDG parser helpers.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup siglentSDG_files
 */

#include "../../../tests/testXWC.hpp"

#include "../siglentSDG_parsers.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup siglentSDG_unit_test siglentSDG Unit Tests
 * \brief Unit tests for the siglentSDG application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `siglentSDG` unit tests.
/** \ingroup siglentSDG_unit_test
 */
namespace siglentSDGTest
{

/// Verify the placeholder siglentSDG parser test harness instantiates the helper paths cleanly.
/**
 * \ingroup siglentSDG_unit_test
 */
TEST_CASE( "siglentSDG parser placeholder harness compiles", "[siglentSDG]" )
{
    // clang-format off
    #ifdef SIGLENTSDG_TEST_DOXYGEN_REF
    siglentSDGVector();
    #endif
    // clang-format on

    SECTION( "default helper construction succeeds" )
    {
        siglentSDGVector values;

        REQUIRE( values.empty() );
    }
}

} // namespace siglentSDGTest

} // namespace libXWCTest
