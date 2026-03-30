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

SCENARIO( "Parsing raw float-vector responses", "[cred2Utils]" )
{
    GIVEN( "A valid delimited float response" )
    {
        std::vector<float> values;

        WHEN( "The response contains the bundled temperature values" )
        {
            int rv = cred2ParseFloatVector( values, "40.50:37.00:40.25:-14.92:2.29:27.50\r\n", 6 );

            REQUIRE( rv == 0 );
            REQUIRE( values.size() == 6 );
            REQUIRE( values[0] == Approx( 40.50f ) );
            REQUIRE( values[1] == Approx( 37.00f ) );
            REQUIRE( values[2] == Approx( 40.25f ) );
            REQUIRE( values[3] == Approx( -14.92f ) );
            REQUIRE( values[4] == Approx( 2.29f ) );
            REQUIRE( values[5] == Approx( 27.50f ) );
        }
    }

    GIVEN( "A malformed delimited float response" )
    {
        std::vector<float> values;

        WHEN( "The value count does not match" )
        {
            int rv = cred2ParseFloatVector( values, "40.50:37.00:40.25\r\n", 6 );

            REQUIRE( rv == -1 );
            REQUIRE( values.empty() );
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

SCENARIO( "Parsing raw cropping responses", "[cred2Utils]" )
{
    GIVEN( "A bundled cropping status response" )
    {
        bool enabled     = false;
        int  startColumn = 0;
        int  endColumn   = 0;
        int  startRow    = 0;
        int  endRow      = 0;

        WHEN( "The response contains the enabled flag and row/column limits" )
        {
            int rv = cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "on:192-447:128-383\r\n" );

            REQUIRE( rv == 0 );
            REQUIRE( enabled == true );
            REQUIRE( startColumn == 192 );
            REQUIRE( endColumn == 447 );
            REQUIRE( startRow == 128 );
            REQUIRE( endRow == 383 );
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

    GIVEN( "A valid C-RED 2 subframe ROI" )
    {
        cred2Roi roi;
        roi.startColumn = 192;
        roi.endColumn   = 447;
        roi.startRow    = 128;
        roi.endRow      = 383;
        roi.fullFrame   = false;

        WHEN( "The ROI is converted back to a MagAO-X center and size" )
        {
            float centerX = 0;
            float centerY = 0;
            int   width   = 0;
            int   height  = 0;

            int rv = cred2RoiToCenter( centerX, centerY, width, height, roi, 640, 512 );

            REQUIRE( rv == 0 );
            REQUIRE( centerX == Approx( 319.5f ) );
            REQUIRE( centerY == Approx( 255.5f ) );
            REQUIRE( width == 256 );
            REQUIRE( height == 256 );
        }
    }
}

} // namespace cred2Utils_test
