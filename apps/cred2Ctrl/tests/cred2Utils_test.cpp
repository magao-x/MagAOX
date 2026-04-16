/** \file cred2Utils_test.cpp
 * \brief Catch2 tests for the C-RED 2 utility helpers.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cred2Ctrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../cred2Utils.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \addtogroup cred2Ctrl_unit_test
 * \brief Additional utility tests for the cred2Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `cred2Ctrl` utility unit tests.
/** \ingroup cred2Ctrl_unit_test
 */
namespace cred2CtrlTest
{

/// Verify `cred2CleanResponse()` strips prompts and trailing line endings.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers clean CLI responses", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2CleanResponse( "" );
    #endif
    // clang-format on

    SECTION( "responses without prompts are trimmed" )
    {
        std::string clean = cred2CleanResponse( "400\r\n" );

        REQUIRE( clean == "400" );
    }

    SECTION( "responses with prompts are trimmed" )
    {
        std::string clean = cred2CleanResponse( "400\r\nfli-cli>" );

        REQUIRE( clean == "400" );
    }
}

/// Verify `cred2ParseFloat()` accepts valid values and rejects non-numeric responses.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers parse float responses", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2ParseFloat( *(float *)nullptr, "" );
    #endif
    // clang-format on

    SECTION( "valid float responses are parsed" )
    {
        float value = 0;
        int   rv    = cred2ParseFloat( value, "-15.5\r\nfli-cli>" );

        REQUIRE( rv == 0 );
        REQUIRE( value == Approx( -15.5f ) );
    }

    SECTION( "non-numeric responses are rejected" )
    {
        float value = 0;
        int   rv    = cred2ParseFloat( value, "Result: OK\r\nfli-cli>" );

        REQUIRE( rv == -1 );
    }
}

/// Verify `cred2ParseFloatVector()` enforces both parsing and element-count expectations.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers parse float-vector responses", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2ParseFloatVector( *(std::vector<float> *)nullptr, "", 0 );
    #endif
    // clang-format on

    SECTION( "valid delimited float responses are parsed" )
    {
        std::vector<float> values;
        int                rv = cred2ParseFloatVector( values, "40.50:37.00:40.25:-14.92:2.29:27.50\r\n", 6 );

        REQUIRE( rv == 0 );
        REQUIRE( values.size() == 6 );
        REQUIRE( values[0] == Approx( 40.50f ) );
        REQUIRE( values[1] == Approx( 37.00f ) );
        REQUIRE( values[2] == Approx( 40.25f ) );
        REQUIRE( values[3] == Approx( -14.92f ) );
        REQUIRE( values[4] == Approx( 2.29f ) );
        REQUIRE( values[5] == Approx( 27.50f ) );
    }

    SECTION( "mismatched element counts are rejected" )
    {
        std::vector<float> values;
        int                rv = cred2ParseFloatVector( values, "40.50:37.00:40.25\r\n", 6 );

        REQUIRE( rv == -1 );
        REQUIRE( values.empty() );
    }
}

/// Verify `cred2ParseRange()` extracts the encoded bounds from valid responses.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers parse range responses", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2ParseRange( *(int *)nullptr, *(int *)nullptr, "" );
    #endif
    // clang-format on

    int firstValue  = 0;
    int secondValue = 0;
    int rv          = cred2ParseRange( firstValue, secondValue, "0-639\r\nfli-cli>" );

    REQUIRE( rv == 0 );
    REQUIRE( firstValue == 0 );
    REQUIRE( secondValue == 639 );
}

/// Verify `cred2ParseBool()` translates textual on/off responses into booleans.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers parse boolean responses", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2ParseBool( *(bool *)nullptr, "" );
    #endif
    // clang-format on

    bool value = false;
    int  rv    = cred2ParseBool( value, "on\r\nfli-cli>" );

    REQUIRE( rv == 0 );
    REQUIRE( value == true );
}

/// Verify `cred2ParseCropState()` unpacks the enabled flag and ROI bounds.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers parse crop responses", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2ParseCropState( *(bool *)nullptr, *(int *)nullptr, *(int *)nullptr, *(int *)nullptr, *(int *)nullptr, "" );
    #endif
    // clang-format on

    bool enabled     = false;
    int  startColumn = 0;
    int  endColumn   = 0;
    int  startRow    = 0;
    int  endRow      = 0;
    int  rv = cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "on:192-447:128-383\r\n" );

    REQUIRE( rv == 0 );
    REQUIRE( enabled == true );
    REQUIRE( startColumn == 192 );
    REQUIRE( endColumn == 447 );
    REQUIRE( startRow == 128 );
    REQUIRE( endRow == 383 );
}

/// Verify the C-RED 2 ROI conversion helpers translate between center/size and corner commands.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl utility helpers format ROI commands", "[cred2Utils]" )
{
    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    cred2RoiFromCenter( *(cred2Roi *)nullptr, 0, 0, 0, 0, 0, 0 );
    cred2ColumnsSpec( cred2Roi() );
    cred2RowsSpec( cred2Roi() );
    cred2RoiToCenter( *(float *)nullptr, *(float *)nullptr, *(int *)nullptr, *(int *)nullptr, cred2Roi(), 0, 0 );
    #endif
    // clang-format on

    SECTION( "full-frame centers expand to the detector corners" )
    {
        cred2Roi roi;
        int      rv = cred2RoiFromCenter( roi, 319.5f, 255.5f, 640, 512, 640, 512 );

        REQUIRE( rv == 0 );
        REQUIRE( roi.startColumn == 0 );
        REQUIRE( roi.endColumn == 639 );
        REQUIRE( roi.startRow == 0 );
        REQUIRE( roi.endRow == 511 );
        REQUIRE( roi.fullFrame == true );
    }

    SECTION( "subframe centers convert to C-RED 2 row and column specifications" )
    {
        cred2Roi roi;
        int      rv = cred2RoiFromCenter( roi, 127.5f, 63.5f, 256, 128, 640, 512 );

        REQUIRE( rv == 0 );
        REQUIRE( cred2ColumnsSpec( roi ) == "0-255" );
        REQUIRE( cred2RowsSpec( roi ) == "0-127" );
        REQUIRE( roi.fullFrame == false );
    }

    SECTION( "corner-defined subframes convert back to a MagAO-X center and size" )
    {
        cred2Roi roi;
        roi.startColumn = 192;
        roi.endColumn   = 447;
        roi.startRow    = 128;
        roi.endRow      = 383;
        roi.fullFrame   = false;

        float centerX = 0;
        float centerY = 0;
        int   width   = 0;
        int   height  = 0;
        int   rv      = cred2RoiToCenter( centerX, centerY, width, height, roi, 640, 512 );

        REQUIRE( rv == 0 );
        REQUIRE( centerX == Approx( 319.5f ) );
        REQUIRE( centerY == Approx( 255.5f ) );
        REQUIRE( width == 256 );
        REQUIRE( height == 256 );
    }
}

} // namespace cred2CtrlTest

} // namespace libXWCTest
