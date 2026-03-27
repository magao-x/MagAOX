/** \file cred2Utils_test.cpp
 * \brief Catch2 tests for the C-RED 2 utility helpers.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 */

#include "../../../tests/catch2/catch.hpp"

#include "../cred2Utils.hpp"

using namespace MagAOX::app;

namespace cred2Utils_test
{

SCENARIO( "Cleaning CLI responses", "[cred2Utils]" )
{
    GIVEN( "A raw C-RED 2 response without a prompt" )
    {
        WHEN( "The response is cleaned" )
        {
            std::string clean = cred2CleanResponse( "400\r\n" );

            REQUIRE( clean == "400" );
        }
    }

    GIVEN( "A raw C-RED 2 response with a prompt" )
    {
        WHEN( "The response is cleaned" )
        {
            std::string clean = cred2CleanResponse( "400\r\nfli-cli>" );

            REQUIRE( clean == "400" );
        }
    }
}

SCENARIO( "Parsing raw float responses", "[cred2Utils]" )
{
    GIVEN( "A valid float response" )
    {
        float value = 0;

        WHEN( "The prompt suffix is present" )
        {
            int rv = cred2ParseFloat( value, "-15.5\r\nfli-cli>" );

            REQUIRE( rv == 0 );
            REQUIRE( value == Approx( -15.5f ) );
        }
    }

    GIVEN( "An invalid float response" )
    {
        float value = 0;

        WHEN( "The response is not numeric" )
        {
            int rv = cred2ParseFloat( value, "Result: OK\r\nfli-cli>" );

            REQUIRE( rv == -1 );
        }
    }
}

SCENARIO( "Parsing raw range responses", "[cred2Utils]" )
{
    GIVEN( "A valid range response" )
    {
        int firstValue  = 0;
        int secondValue = 0;

        WHEN( "The response contains start and end limits" )
        {
            int rv = cred2ParseRange( firstValue, secondValue, "0-639\r\nfli-cli>" );

            REQUIRE( rv == 0 );
            REQUIRE( firstValue == 0 );
            REQUIRE( secondValue == 639 );
        }
    }
}

SCENARIO( "Parsing raw boolean responses", "[cred2Utils]" )
{
    GIVEN( "A valid boolean response" )
    {
        bool value = false;

        WHEN( "The response is on" )
        {
            int rv = cred2ParseBool( value, "on\r\nfli-cli>" );

            REQUIRE( rv == 0 );
            REQUIRE( value == true );
        }
    }
}

SCENARIO( "Formatting C-RED 2 ROI commands", "[cred2Utils]" )
{
    GIVEN( "A valid MagAO-X full-frame ROI" )
    {
        cred2Roi roi;

        WHEN( "The ROI is converted to C-RED 2 corners" )
        {
            int rv = cred2RoiFromCenter( roi, 319.5f, 255.5f, 640, 512, 640, 512 );

            REQUIRE( rv == 0 );
            REQUIRE( roi.startColumn == 0 );
            REQUIRE( roi.endColumn == 639 );
            REQUIRE( roi.startRow == 0 );
            REQUIRE( roi.endRow == 511 );
            REQUIRE( roi.fullFrame == true );
        }
    }

    GIVEN( "A valid MagAO-X subframe ROI" )
    {
        cred2Roi roi;

        WHEN( "The ROI is converted to column and row command strings" )
        {
            int rv = cred2RoiFromCenter( roi, 127.5f, 63.5f, 256, 128, 640, 512 );

            REQUIRE( rv == 0 );
            REQUIRE( cred2ColumnsSpec( roi ) == "0-255" );
            REQUIRE( cred2RowsSpec( roi ) == "0-127" );
            REQUIRE( roi.fullFrame == false );
        }
    }
}

} // namespace cred2Utils_test
