/** \file zaberUtils_test.cpp
 * \brief Catch2 tests for the zaberUtils in the zaberLowLevel app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevel_files
 */

#include "../../../tests/testXWC.hpp"

#include "../zaberUtils.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \addtogroup zaberLowLevel_unit_test
 * \brief Additional utility tests for the zaberLowLevel application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `zaberLowLevel` utility unit tests.
/** \ingroup zaberLowLevel_unit_test
 */
namespace zaberLowLevelTest
{

/// Verify `parseSystemSerial()` extracts axis addresses and serial numbers from a valid response.
/**
 * \ingroup zaberLowLevel_unit_test
 */
TEST_CASE( "zaberLowLevel utility helpers parse system.serial responses", "[zaberUtils]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    parseSystemSerial( *(std::vector<int> *)nullptr, *(std::vector<std::string> *)nullptr, "" );
    #endif
    // clang-format on

    std::string              tstr = "@01 0 OK IDLE WR 49822@02 0 OK IDLE WR 49820@03 0 OK IDLE WR 49821\n";
    std::vector<int>         address;
    std::vector<std::string> serial;
    int                      rv = parseSystemSerial( address, serial, tstr );

    REQUIRE( rv == 0 );
    REQUIRE( address[0] == 1 );
    REQUIRE( address[1] == 2 );
    REQUIRE( address[2] == 3 );

    REQUIRE( serial[0] == "49822" );
    REQUIRE( serial[1] == "49820" );
    REQUIRE( serial[2] == "49821" );
}

} // namespace zaberLowLevelTest

} // namespace libXWCTest
