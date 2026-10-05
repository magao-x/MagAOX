/** \file ocamUtils_test.cpp
 * \brief Catch2 tests for the ocamUtils in the ocam2KCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup ocam2KCtrl_files
 */

#include "../../../tests/testXWC.hpp"

#include "../ocamUtils.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \addtogroup ocam2KCtrl_unit_test
 * \brief Additional utility tests for the ocam2KCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `ocam2KCtrl` utility unit tests.
/** \ingroup ocam2KCtrl_unit_test
 */
namespace ocam2KCtrlTest
{

/// Verify `parseTemps()` accepts a valid OCAM temperature response and rejects truncated data.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl utility helpers parse temperature responses", "[ocamUtils]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    parseTemps( *(ocamTemps *)nullptr, "" );
    #endif
    // clang-format on

    SECTION( "valid temperature responses are parsed" )
    {
        std::string tstr = "Temperatures : CCD[26.3] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                           "SET[200]\nCooling Power [102]mW.\n\n";
        ocamTemps   temps;
        int         rv = parseTemps( temps, tstr );

        REQUIRE( rv == 0 );
        REQUIRE( temps.CCD == (float)26.3 );
        REQUIRE( temps.CPU == (float)41 );
        REQUIRE( temps.POWER == (float)34 );
        REQUIRE( temps.BIAS == (float)47 );
        REQUIRE( temps.WATER == (float)24.2 );
        REQUIRE( temps.LEFT == (float)33 );
        REQUIRE( temps.RIGHT == (float)38 );
        REQUIRE( temps.SET == (float)20.0 );
        REQUIRE( temps.COOLING_POWER == (float)102 );
    }

    SECTION( "truncated temperature responses are rejected" )
    {
        std::string tstr = "Temperatures : CCD[26.3] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                           "SET[200]\nCooling Power";
        ocamTemps   temps;
        int         rv = parseTemps( temps, tstr );

        REQUIRE( rv == -1 );
    }
}

/// Verify `parseEMGain()` accepts valid OCAM gain responses and rejects malformed strings.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl utility helpers parse gain responses", "[ocamUtils]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    parseEMGain( *(unsigned *)nullptr, "" );
    #endif
    // clang-format on

    SECTION( "valid gain responses are parsed" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to 2 \n\n" );

        REQUIRE( rv == 0 );
        REQUIRE( emgain == 2 );
    }

    SECTION( "maximum supported gains are parsed" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to 512 \n\n" );

        REQUIRE( rv == 0 );
        REQUIRE( emgain == 512 );
    }

    SECTION( "gain responses without the trailing delimiter are rejected" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to 512\n\n" );

        REQUIRE( rv == -1 );
        REQUIRE( emgain == 0 );
    }

    SECTION( "gain responses without a numeric value are rejected" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to \n\n" );

        REQUIRE( rv == -1 );
        REQUIRE( emgain == 0 );
    }

    SECTION( "gain responses with trailing junk are rejected" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to 512 rubbish added\n\n" );

        REQUIRE( rv == -1 );
        REQUIRE( emgain == 0 );
    }

    SECTION( "gain responses below the supported range are rejected" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to 0 \n\n" );

        REQUIRE( rv == -1 );
        REQUIRE( emgain == 0 );
    }

    SECTION( "gain responses above the supported range are rejected" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to 601 \n\n" );

        REQUIRE( rv == -1 );
        REQUIRE( emgain == 0 );
    }

    SECTION( "gain responses with invalid tokens are rejected" )
    {
        unsigned emgain = 1;
        int      rv     = parseEMGain( emgain, "Gain set to x \n\n" );

        REQUIRE( rv == -1 );
        REQUIRE( emgain == 0 );
    }
}

} // namespace ocam2KCtrlTest

} // namespace libXWCTest
