/** \file trippLitePDU_test.cpp
 * \brief Catch2 tests for the trippLitePDU app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup trippLitePDU_files
 */

#include "../../../tests/testXWC.hpp"

#include "../trippLitePDU.hpp"
#include "../trippLitePDU_simulator.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup trippLitePDU_unit_test trippLitePDU Unit Tests
 * \brief Unit tests for the trippLitePDU application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `trippLitePDU` unit tests.
/** \ingroup trippLitePDU_unit_test
 */
namespace trippLitePDUTest
{

/// Verify the placeholder trippLitePDU test harness instantiates the app cleanly.
/**
 * \ingroup trippLitePDU_unit_test
 */
TEST_CASE( "trippLitePDU placeholder harness instantiates the app", "[trippLitePDU]" )
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        trippLitePDU app;

        REQUIRE( true );
    }
}

} // namespace trippLitePDUTest

} // namespace libXWCTest
