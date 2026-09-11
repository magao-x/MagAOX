/** \file ImageStreamIO_test.cpp
 * \brief Catch2 tests for the imageStructDataType traits in libMagAOX/ImageStreamIO/ImageStruct.hpp.
 *
 * These are compile time and in-memory tests. The type mapping is checked with
 * STATIC_REQUIRE. The pointer setting is checked on a stack IMAGE structure. No shared
 * memory stream is created.
 *
 * History:
 */
#include "../../../tests/catch2/catch.hpp"

#include "../ImageStruct.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace libXWCTest
{
namespace ImageStreamIOTest
{

/** \defgroup ImageStreamIO_unit_test libXWC::ImageStreamIO Unit Tests
 * \ingroup unit_test
 */

/// Namespace for XWC::imageStructDataType tests
/** \ingroup ImageStreamIO_unit_test
 *
 */
namespace imageStructDataTypeTest
{

/// Each imageStructDataType<code> specialization maps to the right C++ type and sets the right IMAGE array member.
/** One section per data type code checks the type, the size, the maximum value, and that
 * setPointer() stores the pointer in the matching member of the IMAGE array union.
 *
 * \ingroup ImageStreamIO_unit_test
 */
TEST_CASE( "imageStructDataType maps a data type code to a C++ type and sets the IMAGE array pointer",
           "[libMagAOX::ImageStreamIO::imageStructDataType]" )
{
    SECTION( "IMAGESTRUCT_UINT8" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_UINT8>::type, uint8_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT8>::size == sizeof( uint8_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT8>::max == std::numeric_limits<uint8_t>::max() );

        IMAGE   im{};
        uint8_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_UINT8>::setPointer( im, data );

        REQUIRE( im.array.UI8 == data );
    }

    SECTION( "IMAGESTRUCT_INT8" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_INT8>::type, int8_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT8>::size == sizeof( int8_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT8>::max == std::numeric_limits<int8_t>::max() );

        IMAGE  im{};
        int8_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_INT8>::setPointer( im, data );

        REQUIRE( im.array.SI8 == data );
    }

    SECTION( "IMAGESTRUCT_UINT16" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_UINT16>::type, uint16_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT16>::size == sizeof( uint16_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT16>::max == std::numeric_limits<uint16_t>::max() );

        IMAGE    im{};
        uint16_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_UINT16>::setPointer( im, data );

        REQUIRE( im.array.UI16 == data );
    }

    SECTION( "IMAGESTRUCT_INT16" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_INT16>::type, int16_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT16>::size == sizeof( int16_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT16>::max == std::numeric_limits<int16_t>::max() );

        IMAGE   im{};
        int16_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_INT16>::setPointer( im, data );

        REQUIRE( im.array.SI16 == data );
    }

    SECTION( "IMAGESTRUCT_UINT32" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_UINT32>::type, uint32_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT32>::size == sizeof( uint32_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT32>::max == std::numeric_limits<uint32_t>::max() );

        IMAGE    im{};
        uint32_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_UINT32>::setPointer( im, data );

        REQUIRE( im.array.UI32 == data );
    }

    SECTION( "IMAGESTRUCT_INT32" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_INT32>::type, int32_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT32>::size == sizeof( int32_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT32>::max == std::numeric_limits<int32_t>::max() );

        IMAGE   im{};
        int32_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_INT32>::setPointer( im, data );

        REQUIRE( im.array.SI32 == data );
    }

    SECTION( "IMAGESTRUCT_UINT64" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_UINT64>::type, uint64_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT64>::size == sizeof( uint64_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_UINT64>::max == std::numeric_limits<uint64_t>::max() );

        IMAGE    im{};
        uint64_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_UINT64>::setPointer( im, data );

        REQUIRE( im.array.UI64 == data );
    }

    SECTION( "IMAGESTRUCT_INT64" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_INT64>::type, int64_t>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT64>::size == sizeof( int64_t ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_INT64>::max == std::numeric_limits<int64_t>::max() );

        IMAGE   im{};
        int64_t data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_INT64>::setPointer( im, data );

        REQUIRE( im.array.SI64 == data );
    }

    SECTION( "IMAGESTRUCT_FLOAT" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_FLOAT>::type, float>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_FLOAT>::size == sizeof( float ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_FLOAT>::max == std::numeric_limits<float>::max() );

        IMAGE im{};
        float data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_FLOAT>::setPointer( im, data );

        REQUIRE( im.array.F == data );
    }

    SECTION( "IMAGESTRUCT_DOUBLE" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_DOUBLE>::type, double>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_DOUBLE>::size == sizeof( double ) );
        REQUIRE( imageStructDataType<IMAGESTRUCT_DOUBLE>::max == std::numeric_limits<double>::max() );

        IMAGE  im{};
        double data[1] = { 0 };

        imageStructDataType<IMAGESTRUCT_DOUBLE>::setPointer( im, data );

        REQUIRE( im.array.D == data );
    }

    SECTION( "IMAGESTRUCT_COMPLEX_FLOAT" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_COMPLEX_FLOAT>::type, complex_float>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_COMPLEX_FLOAT>::size == sizeof( complex_float ) );

        IMAGE         im{};
        complex_float data[1] = { { 0.0f, 0.0f } };

        imageStructDataType<IMAGESTRUCT_COMPLEX_FLOAT>::setPointer( im, data );

        REQUIRE( im.array.CF == data );
    }

    SECTION( "IMAGESTRUCT_COMPLEX_DOUBLE" )
    {
        STATIC_REQUIRE( std::is_same<imageStructDataType<IMAGESTRUCT_COMPLEX_DOUBLE>::type, complex_double>::value );
        REQUIRE( imageStructDataType<IMAGESTRUCT_COMPLEX_DOUBLE>::size == sizeof( complex_double ) );

        IMAGE          im{};
        complex_double data[1] = { { 0.0, 0.0 } };

        imageStructDataType<IMAGESTRUCT_COMPLEX_DOUBLE>::setPointer( im, data );

        REQUIRE( im.array.CD == data );
    }
}

} // namespace imageStructDataTypeTest
} // namespace ImageStreamIOTest
} // namespace libXWCTest
