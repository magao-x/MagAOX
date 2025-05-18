/** \file Types_tests.cpp
 * \brief Catch2 tests for Types.hpp.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include <iostream>

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../Types.hpp"

/** Scenario: Converting signed integers
 *
 * Verify to and from conversions of signed integers
 *
 * \anchor tests_xindi_Types_conversions_signed_int
 */
SCENARIO( "Converting signed integers", "[xindi::Types::conversions]" )
{
    GIVEN( "A small number" )
    {
        WHEN( "A positive small number" )
        {
            int i0 = 2;

            std::string str = xindi::value2string( i0 );

            int i1 = xindi::string2value<int>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A negative small number" )
        {
            int i0 = -2;

            std::string str = xindi::value2string( i0 );

            int i1 = xindi::string2value<int>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A positive large number" )
        {
            int64_t i0 = 9223372036854775807;

            std::string str = xindi::value2string( i0 );

            int64_t i1 = xindi::string2value<int64_t>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A negative large number" )
        {
            int64_t i0 = -9223372036854775807;

            std::string str = xindi::value2string( i0 );

            int64_t i1 = xindi::string2value<int64_t>( str );

            REQUIRE( i1 == i0 );
        }
    }
}

/** Scenario: Converting unsigned integers
 *
 * Verify to and from conversions of unsigned integers
 *
 * \anchor tests_xindi_Types_conversions_unsigned_int
 */
SCENARIO( "Converting unsigned integers", "[xindi::Types::conversions]" )
{
    GIVEN( "A small number" )
    {
        WHEN( "A small number" )
        {
            unsigned int i0 = 2;

            std::string str = xindi::value2string( i0 );

            unsigned int i1 = xindi::string2value<unsigned int>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A large number number" )
        {
            uint64_t i0 = 18446744073709551615ul;

            std::string str = xindi::value2string( i0 );

            uint64_t i1 = xindi::string2value<uint64_t>( str );

            REQUIRE( i1 == i0 );
        }
    }
}

/** Scenario: Converting bools
 *
 * Verify to and from conversions of bools
 *
 * \anchor tests_xindi_Types_conversions_bool
 */
SCENARIO( "Converting bools", "[xindi::Types::conversions]" )
{
    GIVEN( "A bool" )
    {
        WHEN( "true" )
        {
            bool b0 = true;

            std::string str = xindi::value2string( b0 );

            bool b1 = xindi::string2value<bool>( str );

            REQUIRE( b1 == b0 );
        }
        WHEN( "false" )
        {
            bool b0 = false;

            std::string str = xindi::value2string( b0 );

            bool b1 = xindi::string2value<bool>( str );

            REQUIRE( b1 == b0 );
        }
    }
}

/** Scenario: Converting real floating point numbers
 *
 * Verify to and from conversions of real floating point numbers
 *
 * \anchor tests_xindi_Types_conversions_real_floats
 */
SCENARIO( "Converting real floating point numbers", "[xindi::Types::conversions]" )
{
    GIVEN( "A single precision number" )
    {
        WHEN( "A positive number with precision of 1" )
        {
            float f0 = 1.2;

            std::string str = xindi::value2string( f0 );

            float f1 = xindi::string2value<float>( str );

            REQUIRE( f1 == f0 );
        }
        WHEN( "A positive number with precision of 7" )
        {
            float f0 = 1.01234567;

            std::string str = xindi::value2string( f0 );

            float f1 = xindi::string2value<float>( str );

            REQUIRE( f1 == f0 );
        }
    }

    GIVEN( "A double precision number" )
    {
        WHEN( "A positive number with precision of 1" )
        {
            double d0 = 1.2;

            std::string str = xindi::value2string( d0 );

            double d1 = xindi::string2value<double>( str );

            REQUIRE( d1 == d0 );
        }

        WHEN( "A positive number with precision of 16" )
        {
            double d0 = 1.0123456789123456;

            std::string str = xindi::value2string( d0 );

            double d1 = xindi::string2value<double>( str );

            REQUIRE( d1 == d0 );
        }
    }
}

/** Scenario: Converting complex floating point numbers
 *
 * Verify to and from conversions of complex floating point numbers
 *
 * \anchor tests_xindi_Types_conversions_complex_floats
 */
SCENARIO( "Converting complex floating point numbers", "[xindi::Types::conversions]" )
{
    GIVEN( "A single precision complex number" )
    {
        WHEN( "Both positive with precision of 1" )
        {
            std::complex<float> cf0( 1.2, 2.4 );
            std::string         str = xindi::value2string( cf0 );

            std::complex<float> cf1 = xindi::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }
        WHEN( "Both negative with precision of 1" )
        {
            std::complex<float> cf0( -3.7, -5.1 );
            std::string         str = xindi::value2string( cf0 );

            std::complex<float> cf1 = xindi::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }

        WHEN( "Real positive, imaginary negative with precision of 1" )
        {
            std::complex<float> cf0( 9.3, -8.5 );
            std::string         str = xindi::value2string( cf0 );

            std::complex<float> cf1 = xindi::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }

        WHEN( "Real negative, imaginary positive with precision of 1" )
        {
            std::complex<float> cf0( -2.5, 7.6 );
            std::string         str = xindi::value2string( cf0 );

            std::complex<float> cf1 = xindi::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }
    }
}

/** Scenario: Converting Switches
 *
 * Verify to and from conversions of type Switches
 *
 * \anchor tests_xindi_Types_conversions_switch
 */
SCENARIO( "Converting Switches", "[xindi::Types::conversions]" )
{
    GIVEN( "A Switch" )
    {
        WHEN( "On" )
        {
            xindi::Switch sw0 = xindi::Switch::On;

            std::string str = xindi::value2string( sw0 );

            xindi::Switch sw1 = xindi::string2value<xindi::Switch>( str );

            REQUIRE( sw1 == sw0 );
        }
        WHEN( "Off" )
        {
            xindi::Switch sw0 = xindi::Switch::Off;

            std::string str = xindi::value2string( sw0 );

            xindi::Switch sw1 = xindi::string2value<xindi::Switch>( str );

            REQUIRE( sw1 == sw0 );
        }
        WHEN( "Unknown" )
        {
            xindi::Switch sw0 = xindi::Switch::Unknown;

            std::string str = xindi::value2string( sw0 );

            xindi::Switch sw1 = xindi::string2value<xindi::Switch>( str );

            REQUIRE( sw1 == sw0 );
        }
        WHEN( "Arb String" )
        {
            xindi::Switch sw0 = xindi::Switch::Unknown;

            xindi::Switch sw1 = xindi::string2value<xindi::Switch>( "xysn1" );

            REQUIRE( sw1 == sw0 );
        }
    }
}

/** Scenario: Converting Lights
 *
 * Verify to and from conversions of type Lights
 *
 * \anchor tests_xindi_Types_conversions_light
 */
SCENARIO( "Converting Lights", "[xindi::Types::conversions]" )
{
    GIVEN( "A Light" )
    {
        WHEN( "Idle" )
        {
            xindi::Light l0 = xindi::Light::Idle;

            std::string str = xindi::value2string( l0 );

            xindi::Light l1 = xindi::string2value<xindi::Light>( str );

            REQUIRE( l1 == l0 );
        }
        WHEN( "Ok" )
        {
            xindi::Light l0 = xindi::Light::Ok;

            std::string str = xindi::value2string( l0 );

            xindi::Light l1 = xindi::string2value<xindi::Light>( str );

            REQUIRE( l1 == l0 );
        }
        WHEN( "Busy" )
        {
            xindi::Light l0 = xindi::Light::Busy;

            std::string str = xindi::value2string( l0 );

            xindi::Light l1 = xindi::string2value<xindi::Light>( str );

            REQUIRE( l1 == l0 );
        }
        WHEN( "Alert" )
        {
            xindi::Light l0 = xindi::Light::Alert;

            std::string str = xindi::value2string( l0 );

            xindi::Light l1 = xindi::string2value<xindi::Light>( str );

            REQUIRE( l1 == l0 );
        }
        WHEN( "Unknown" )
        {
            xindi::Light l0 = xindi::Light::Unknown;

            std::string str = xindi::value2string( l0 );

            xindi::Light l1 = xindi::string2value<xindi::Light>( str );

            REQUIRE( l1 == l0 );
        }
        WHEN( "Arb String" )
        {
            xindi::Light l0 = xindi::Light::Unknown;

            xindi::Light l1 = xindi::string2value<xindi::Light>( "qpmz56" );

            REQUIRE( l1 == l0 );
        }
    }
}
