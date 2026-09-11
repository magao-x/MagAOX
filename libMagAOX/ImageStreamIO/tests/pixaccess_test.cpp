/** \file pixaccess_test.cpp
 * \brief Catch2 tests for the pixel access helpers in libMagAOX/ImageStreamIO/pixaccess.hpp.
 *
 * getPix() reads one pixel from a raw buffer and converts it to the requested type.
 * getPixPointer() returns the getPix() instantiation for an ImageStreamIO data type code.
 * The tests call these directly on small local arrays. No shared memory stream is needed.
 *
 * History:
 */
#include "../../../tests/catch2/catch.hpp"

#include "../pixaccess.hpp"

#include <cstdint>
#include <limits>

namespace libXWCTest
{
namespace ImageStreamIOTest
{

/// Namespace for XWC::pixaccess tests
/** \ingroup ImageStreamIO_unit_test
 *
 */
namespace pixaccessTest
{

/// Direct calls to getPix<returnT,dataT> for every supported data type.
/** Each section reads the minimum, zero, and maximum of one type and checks the value
 * survives the cast. The last section checks a return type other than double.
 * \ingroup ImageStreamIO_unit_test
 */
TEST_CASE( "getPix casts image data to the return type", "[libMagAOX::ImageStreamIO::pixaccess]" )
{
    SECTION( "uint8_t data" )
    {
        uint8_t data[3] = { 0, 1, std::numeric_limits<uint8_t>::max() };

        REQUIRE( getPix<double, uint8_t>( data, 0 ) == 0.0 );
        REQUIRE( getPix<double, uint8_t>( data, 1 ) == 1.0 );
        REQUIRE( getPix<double, uint8_t>( data, 2 ) == static_cast<double>( std::numeric_limits<uint8_t>::max() ) );
    }

    SECTION( "int8_t data" )
    {
        int8_t data[3] = { std::numeric_limits<int8_t>::min(), 0, std::numeric_limits<int8_t>::max() };

        REQUIRE( getPix<double, int8_t>( data, 0 ) == static_cast<double>( std::numeric_limits<int8_t>::min() ) );
        REQUIRE( getPix<double, int8_t>( data, 1 ) == 0.0 );
        REQUIRE( getPix<double, int8_t>( data, 2 ) == static_cast<double>( std::numeric_limits<int8_t>::max() ) );
    }

    SECTION( "uint16_t data" )
    {
        uint16_t data[3] = { 0, 1, std::numeric_limits<uint16_t>::max() };

        REQUIRE( getPix<double, uint16_t>( data, 0 ) == 0.0 );
        REQUIRE( getPix<double, uint16_t>( data, 1 ) == 1.0 );
        REQUIRE( getPix<double, uint16_t>( data, 2 ) == static_cast<double>( std::numeric_limits<uint16_t>::max() ) );
    }

    SECTION( "int16_t data" )
    {
        int16_t data[3] = { std::numeric_limits<int16_t>::min(), 0, std::numeric_limits<int16_t>::max() };

        REQUIRE( getPix<double, int16_t>( data, 0 ) == static_cast<double>( std::numeric_limits<int16_t>::min() ) );
        REQUIRE( getPix<double, int16_t>( data, 1 ) == 0.0 );
        REQUIRE( getPix<double, int16_t>( data, 2 ) == static_cast<double>( std::numeric_limits<int16_t>::max() ) );
    }

    SECTION( "uint32_t data" )
    {
        uint32_t data[3] = { 0, 1, std::numeric_limits<uint32_t>::max() };

        REQUIRE( getPix<double, uint32_t>( data, 0 ) == 0.0 );
        REQUIRE( getPix<double, uint32_t>( data, 1 ) == 1.0 );
        REQUIRE( getPix<double, uint32_t>( data, 2 ) == static_cast<double>( std::numeric_limits<uint32_t>::max() ) );
    }

    SECTION( "int32_t data" )
    {
        int32_t data[3] = { std::numeric_limits<int32_t>::min(), 0, std::numeric_limits<int32_t>::max() };

        REQUIRE( getPix<double, int32_t>( data, 0 ) == static_cast<double>( std::numeric_limits<int32_t>::min() ) );
        REQUIRE( getPix<double, int32_t>( data, 1 ) == 0.0 );
        REQUIRE( getPix<double, int32_t>( data, 2 ) == static_cast<double>( std::numeric_limits<int32_t>::max() ) );
    }

    SECTION( "uint64_t data" )
    {
        uint64_t data[3] = { 0, 1, std::numeric_limits<uint64_t>::max() };

        REQUIRE( getPix<double, uint64_t>( data, 0 ) == 0.0 );
        REQUIRE( getPix<double, uint64_t>( data, 1 ) == 1.0 );
        REQUIRE( getPix<double, uint64_t>( data, 2 ) == static_cast<double>( std::numeric_limits<uint64_t>::max() ) );
    }

    SECTION( "int64_t data" )
    {
        int64_t data[3] = { std::numeric_limits<int64_t>::min(), 0, std::numeric_limits<int64_t>::max() };

        REQUIRE( getPix<double, int64_t>( data, 0 ) == static_cast<double>( std::numeric_limits<int64_t>::min() ) );
        REQUIRE( getPix<double, int64_t>( data, 1 ) == 0.0 );
        REQUIRE( getPix<double, int64_t>( data, 2 ) == static_cast<double>( std::numeric_limits<int64_t>::max() ) );
    }

    SECTION( "float data" )
    {
        float data[3] = { -1.5f, 0.0f, 3.5f };

        REQUIRE( getPix<double, float>( data, 0 ) == -1.5 );
        REQUIRE( getPix<double, float>( data, 1 ) == 0.0 );
        REQUIRE( getPix<double, float>( data, 2 ) == 3.5 );
    }

    SECTION( "double data" )
    {
        double data[3] = { -1.5, 0.0, 3.5 };

        REQUIRE( getPix<double, double>( data, 0 ) == -1.5 );
        REQUIRE( getPix<double, double>( data, 1 ) == 0.0 );
        REQUIRE( getPix<double, double>( data, 2 ) == 3.5 );
    }

    SECTION( "return type differs from data type" )
    {
        uint16_t data[1] = { 42 };

        REQUIRE( getPix<int, uint16_t>( data, 0 ) == 42 );
    }
}

/// Direct calls to the compile time getPixPointer<returnT,imageStructDataT>().
/** For each data type code the returned function pointer must not be null and must read the
 * pixel correctly when called.
 * \ingroup ImageStreamIO_unit_test
 */
TEST_CASE( "getPixPointer<returnT,imageStructDataT> returns the getPix instantiation for that type",
           "[libMagAOX::ImageStreamIO::pixaccess]" )
{
    SECTION( "IMAGESTRUCT_UINT8" )
    {
        uint8_t data[1] = { 5 };

        auto fp = getPixPointer<double, IMAGESTRUCT_UINT8>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 5.0 );
    }

    SECTION( "IMAGESTRUCT_INT8" )
    {
        int8_t data[1] = { -5 };

        auto fp = getPixPointer<double, IMAGESTRUCT_INT8>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -5.0 );
    }

    SECTION( "IMAGESTRUCT_UINT16" )
    {
        uint16_t data[1] = { 500 };

        auto fp = getPixPointer<double, IMAGESTRUCT_UINT16>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 500.0 );
    }

    SECTION( "IMAGESTRUCT_INT16" )
    {
        int16_t data[1] = { -500 };

        auto fp = getPixPointer<double, IMAGESTRUCT_INT16>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -500.0 );
    }

    SECTION( "IMAGESTRUCT_UINT32" )
    {
        uint32_t data[1] = { 70000 };

        auto fp = getPixPointer<double, IMAGESTRUCT_UINT32>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 70000.0 );
    }

    SECTION( "IMAGESTRUCT_INT32" )
    {
        int32_t data[1] = { -70000 };

        auto fp = getPixPointer<double, IMAGESTRUCT_INT32>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -70000.0 );
    }

    SECTION( "IMAGESTRUCT_UINT64" )
    {
        uint64_t data[1] = { 5000000000ULL }; // outside the 32 bit range

        auto fp = getPixPointer<double, IMAGESTRUCT_UINT64>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 5000000000.0 );
    }

    SECTION( "IMAGESTRUCT_INT64" )
    {
        int64_t data[1] = { -5000000000LL }; // outside the 32 bit range

        auto fp = getPixPointer<double, IMAGESTRUCT_INT64>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -5000000000.0 );
    }

    SECTION( "IMAGESTRUCT_FLOAT" )
    {
        float data[1] = { 2.25f };

        auto fp = getPixPointer<double, IMAGESTRUCT_FLOAT>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 2.25 );
    }

    SECTION( "IMAGESTRUCT_DOUBLE" )
    {
        double data[1] = { 2.25 };

        auto fp = getPixPointer<double, IMAGESTRUCT_DOUBLE>();

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 2.25 );
    }
}

/// Runtime dispatch through getPixPointer<returnT>(int).
/** The data type code is passed as a runtime integer. Each supported code must return a
 * working function pointer. Complex, event, and invalid codes must return nullptr.
 * \ingroup ImageStreamIO_unit_test
 */
TEST_CASE( "getPixPointer<returnT>(int) dispatches on a runtime image data type code",
           "[libMagAOX::ImageStreamIO::pixaccess]" )
{
    SECTION( "IMAGESTRUCT_UINT8" )
    {
        uint8_t data[1] = { 7 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_UINT8 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 7.0 );
    }

    SECTION( "IMAGESTRUCT_INT8" )
    {
        int8_t data[1] = { -7 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_INT8 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -7.0 );
    }

    SECTION( "IMAGESTRUCT_UINT16" )
    {
        uint16_t data[1] = { 700 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_UINT16 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 700.0 );
    }

    SECTION( "IMAGESTRUCT_INT16" )
    {
        int16_t data[1] = { -700 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_INT16 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -700.0 );
    }

    SECTION( "IMAGESTRUCT_UINT32" )
    {
        uint32_t data[1] = { 80000 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_UINT32 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 80000.0 );
    }

    SECTION( "IMAGESTRUCT_INT32" )
    {
        int32_t data[1] = { -80000 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_INT32 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -80000.0 );
    }

    SECTION( "IMAGESTRUCT_UINT64" )
    {
        uint64_t data[1] = { 6000000000ULL }; // outside the 32 bit range

        auto fp = getPixPointer<double>( IMAGESTRUCT_UINT64 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 6000000000.0 );
    }

    SECTION( "IMAGESTRUCT_INT64" )
    {
        int64_t data[1] = { -6000000000LL }; // outside the 32 bit range

        auto fp = getPixPointer<double>( IMAGESTRUCT_INT64 );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == -6000000000.0 );
    }

    SECTION( "IMAGESTRUCT_FLOAT" )
    {
        float data[1] = { 4.5f };

        auto fp = getPixPointer<double>( IMAGESTRUCT_FLOAT );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 4.5 );
    }

    SECTION( "IMAGESTRUCT_DOUBLE" )
    {
        double data[1] = { 4.5 };

        auto fp = getPixPointer<double>( IMAGESTRUCT_DOUBLE );

        REQUIRE( fp != nullptr );
        REQUIRE( fp( data, 0 ) == 4.5 );
    }

    SECTION( "unknown data type codes return nullptr" )
    {
        REQUIRE( getPixPointer<double>( IMAGESTRUCT_COMPLEX_FLOAT ) == nullptr );
        REQUIRE( getPixPointer<double>( IMAGESTRUCT_COMPLEX_DOUBLE ) == nullptr );
        REQUIRE( getPixPointer<double>( IMAGESTRUCT_EVENT_UI8_UI8_UI16_UI8 ) == nullptr );
        REQUIRE( getPixPointer<double>( 0 ) == nullptr );
        REQUIRE( getPixPointer<double>( -1 ) == nullptr );
    }
}

} // namespace pixaccessTest
} // namespace ImageStreamIOTest
} // namespace libXWCTest
