/** \file ocamUtils_test.cpp
 * \brief Catch2 tests for the ocamUtils in the ocam2KCtrl app.
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

/// Reject invalid temperature fields and framing without partially updating the output.
/** \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl utility helpers validate every temperature field", "[ocamUtils]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    parseTemps( *(ocamTemps *)nullptr, "" );
    #endif
    // clang-format on

    const std::string valid = "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                              "SET[-450]\nCooling Power [102]mW.\n\n";

    SECTION( "each numeric field requires a complete finite value" )
    {
        for( const std::string field :
             { "CCD[", "CPU[", "POWER[", "BIAS[", "WATER[", "LEFT[", "RIGHT[", "SET[", "Cooling Power [" } )
        {
            for( const std::string invalid :
                 { "", "rubbish", "3.0junk", "nan", "inf", "-inf", "1e100", "1e-100", "0x1p2", "+-1" } )
            {
                DYNAMIC_SECTION( field << invalid )
                {
                    std::string response = valid;
                    size_t      first    = response.find( field ) + field.size();
                    response.replace( first, response.find( ']', first ) - first, invalid );

                    ocamTemps temps;
                    temps.setInvalid();

                    REQUIRE( parseTemps( temps, response ) == -1 );
                    REQUIRE( temps.CCD == -999 );
                    REQUIRE( temps.CPU == -999 );
                    REQUIRE( temps.COOLING_POWER == -999 );
                }
            }
        }
    }

    SECTION( "malformed labels delimiters and physical values are rejected" )
    {
        for( const std::string response :
             { "Temperatures : CCD[-999] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
               "SET[-450]\nCooling Power [102]mW.\n\n",
               "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
               "SET[-9999]\nCooling Power [102]mW.\n\n",
               "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
               "SET[-450]\nCooling Power [-1]mW.\n\n",
               "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
               "SET[-450]\nCooling Power [102mW.\n\n",
               "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
               "SET[-450]\nCooling Power [102]mW.[1]\n\n",
               "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] WRONG[38] "
               "SET[-450]\nCooling Power [102]mW.\n\n",
               "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
               "SET[-450]\nCooling Power [102]mW. rubbish\n\n" } )
        {
            CAPTURE( response );
            ocamTemps temps;
            temps.setInvalid();

            REQUIRE( parseTemps( temps, response ) == -1 );
            REQUIRE( temps.SET == -999 );
            REQUIRE( temps.CPU == -999 );
        }
    }

    SECTION( "surrounding whitespace and decimal exponent values are accepted" )
    {
        std::string response = valid;
        response.replace( response.find( "CCD[-45]" ), 8, "CCD[ -4.5e1 ]" );
        response.replace( response.find( "CPU[41]" ), 7, "CPU[ +4.1e1 ]" );
        ocamTemps temps;

        REQUIRE( parseTemps( temps, "\r\n" + response ) == 0 );
        REQUIRE( temps.CCD == -45 );
        REQUIRE( temps.CPU == 41 );
        REQUIRE( temps.SET == -45 );
    }
}

/// Accept complete positive FPS replies and reject invalid numbers or framing.
/** \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl utility helpers parse FPS responses", "[ocamUtils]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    parseFPS( *(float *)nullptr, "" );
    #endif
    // clang-format on

    SECTION( "valid rates retain fractional precision" )
    {
        for( const std::string response : { "fps [150.5] Hz\n", "\r\nfps [ +1.505e2 ] Hz\r\n" } )
        {
            float fps = 0;

            REQUIRE( parseFPS( fps, response ) == 0 );
            REQUIRE( fps == 150.5F );
        }
    }

    SECTION( "malformed nonpositive nonfinite and out of range rates are rejected" )
    {
        for( const std::string response : { "",
                                            "fps [0] Hz\n",
                                            "fps [-0] Hz\n",
                                            "fps [-1] Hz\n",
                                            "fps [] Hz\n",
                                            "fps [x] Hz\n",
                                            "fps [150.5x] Hz\n",
                                            "fps [nan] Hz\n",
                                            "fps [inf] Hz\n",
                                            "fps [-inf] Hz\n",
                                            "fps [1e100] Hz\n",
                                            "fps [1e-100] Hz\n",
                                            "fps [0x1p2] Hz\n",
                                            "fps [+-1] Hz\n",
                                            "fps [150.5 Hz\n",
                                            "wrong [150.5] Hz\n",
                                            "fps [150.5] wrong\n",
                                            "fps [150.5] Hz[2]\n" } )
        {
            CAPTURE( response );
            float fps = 12.5;

            REQUIRE( parseFPS( fps, response ) == -1 );
            REQUIRE( fps == 12.5F );
        }
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

    SECTION( "gain responses require matching labels and a complete integer" )
    {
        for( const std::string response : { "Other set to 42 \n\n",
                                            "Gain was to 42 \n\n",
                                            "Gain set xx 42 \n\n",
                                            "Gain set to 42 rubbish",
                                            "Gain set to 42x \n\n",
                                            "Gain set to 42.5 \n\n",
                                            "Gain set to 4e1 \n\n",
                                            "Gain set to -1 \n\n",
                                            "Gain set to +1 \n\n",
                                            "Gain set to nan \n\n",
                                            "Gain set to inf \n\n",
                                            "Gain set to 99999999999999999999999 \n\n" } )
        {
            CAPTURE( response );
            unsigned emgain = 42;

            REQUIRE( parseEMGain( emgain, response ) == -1 );
            REQUIRE( emgain == 0 );
        }
    }

    SECTION( "the gain range includes both endpoints" )
    {
        unsigned emgain = 0;

        REQUIRE( parseEMGain( emgain, "Gain set to 1 \r\n" ) == 0 );
        REQUIRE( emgain == 1 );
        REQUIRE( parseEMGain( emgain, "Gain set to 600 \r\n" ) == 0 );
        REQUIRE( emgain == 600 );
    }
}

} // namespace ocam2KCtrlTest

} // namespace libXWCTest
