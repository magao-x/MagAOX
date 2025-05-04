/** \file IndiElement_internal_tests.cpp
 * \brief Catch2 tests for the internal namespace in IndiElement.hpp.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include <iostream>

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../IndiElement.hpp"

SCENARIO( "Converting signed integers", "[pcf::internal::conversions]" )
{
    GIVEN( "A small number" )
    {
        WHEN( "A positive small number" )
        {
            int i0 = 2;

            std::string str = pcf::internal::value2string( i0 );

            int i1 = pcf::internal::string2value<int>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A negative small number" )
        {
            int i0 = -2;

            std::string str = pcf::internal::value2string( i0 );

            int i1 = pcf::internal::string2value<int>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A positive large number" )
        {
            int64_t i0 = 9223372036854775807;

            std::string str = pcf::internal::value2string( i0 );

            int64_t i1 = pcf::internal::string2value<int64_t>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A negative large number" )
        {
            int64_t i0 = -9223372036854775807;

            std::string str = pcf::internal::value2string( i0 );

            int64_t i1 = pcf::internal::string2value<int64_t>( str );

            REQUIRE( i1 == i0 );
        }
    }
}

SCENARIO( "Converting unsigned integers", "[pcf::internal::conversions]" )
{
    GIVEN( "A small number" )
    {
        WHEN( "A small number" )
        {
            unsigned int i0 = 2;

            std::string str = pcf::internal::value2string( i0 );

            unsigned int i1 = pcf::internal::string2value<unsigned int>( str );

            REQUIRE( i1 == i0 );
        }
        WHEN( "A large number number" )
        {
            uint64_t i0 = 18446744073709551615ul;

            std::string str = pcf::internal::value2string( i0 );

            uint64_t i1 = pcf::internal::string2value<uint64_t>( str );

            REQUIRE( i1 == i0 );
        }
    }
}

SCENARIO( "Converting bools", "[pcf::internal::conversions]" )
{
    GIVEN( "A bool" )
    {
        WHEN( "true" )
        {
            bool b0 = true;

            std::string str = pcf::internal::value2string( b0 );

            bool b1 = pcf::internal::string2value<bool>( str );

            REQUIRE( b1 == b0 );
        }
        WHEN( "false" )
        {
            bool b0 = false;

            std::string str = pcf::internal::value2string( b0 );

            bool b1 = pcf::internal::string2value<bool>( str );

            REQUIRE( b1 == b0 );
        }
    }
}
SCENARIO( "Converting real floating point numbers", "[pcf::internal::conversions]" )
{
    GIVEN( "A single precision number" )
    {
        WHEN( "A positive number with precision of 1" )
        {
            float f0 = 1.2;

            std::string str = pcf::internal::value2string( f0 );

            float f1 = pcf::internal::string2value<float>( str );

            REQUIRE( f1 == f0 );
        }
        WHEN( "A positive number with precision of 7" )
        {
            float f0 = 1.01234567;

            std::string str = pcf::internal::value2string( f0 );

            float f1 = pcf::internal::string2value<float>( str );

            REQUIRE( f1 == f0 );
        }
    }

    GIVEN( "A double precision number" )
    {
        WHEN( "A positive number with precision of 1" )
        {
            double d0 = 1.2;

            std::string str = pcf::internal::value2string( d0 );

            double d1 = pcf::internal::string2value<double>( str );

            REQUIRE( d1 == d0 );
        }

        WHEN( "A positive number with precision of 16" )
        {
            double d0 = 1.0123456789123456;

            std::string str = pcf::internal::value2string( d0 );

            double d1 = pcf::internal::string2value<double>( str );

            REQUIRE( d1 == d0 );
        }
    }
}

SCENARIO( "Converting complex floating point numbers", "[pcf::internal::conversions]" )
{
    GIVEN( "A single precision complex number" )
    {
        WHEN( "Both positive with precision of 1" )
        {
            std::complex<float> cf0( 1.2, 2.4 );
            std::string         str = pcf::internal::value2string( cf0 );

            std::complex<float> cf1 = pcf::internal::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }
        WHEN( "Both negative with precision of 1" )
        {
            std::complex<float> cf0( -3.7, -5.1 );
            std::string         str = pcf::internal::value2string( cf0 );

            std::complex<float> cf1 = pcf::internal::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }

        WHEN( "Real positive, imaginary negative with precision of 1" )
        {
            std::complex<float> cf0( 9.3, -8.5 );
            std::string         str = pcf::internal::value2string( cf0 );

            std::complex<float> cf1 = pcf::internal::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }

        WHEN( "Real negative, imaginary positive with precision of 1" )
        {
            std::complex<float> cf0( -2.5, 7.6 );
            std::string         str = pcf::internal::value2string( cf0 );

            std::complex<float> cf1 = pcf::internal::string2value<std::complex<float>>( str );

            REQUIRE( cf1 == cf0 );
        }
    }
}
