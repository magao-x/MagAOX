/** \file ImageStreamIO_test.cpp
 * \brief Catch2 tests for the ImageStreamIO pixel access helpers.
 */

#include "../../../tests/testXWC.hpp"

#include "../ImageStruct.hpp"
#include "../pixaccess.hpp"

#include <vector>

namespace libXWCTest
{
namespace pixaccessTest
{

/// \cond PIXACCESS_TEST_HELPERS
/// Check that the frame converters for one type match getPix for every pixel, when setting and adding.
template <int imageStructDataT>
void checkFrame()
{
    typedef typename imageStructDataType<imageStructDataT>::type dataT;

    std::vector<dataT> src( 37 ); // not a multiple of a vector width
    for( size_t n = 0; n < src.size(); ++n )
    {
        src[n] = static_cast<dataT>( ( n * 7 ) % 100 );
    }

    auto setFrame = getPixFramePointer<float, false>( imageStructDataT );
    auto addFrame = getPixFramePointer<float, true>( imageStructDataT );
    auto pixget   = getPixPointer<float>( imageStructDataT );

    REQUIRE( setFrame != nullptr );
    REQUIRE( addFrame != nullptr );

    std::vector<float> dest( src.size() + 1, -1 );
    setFrame( dest.data(), src.data(), src.size() );

    for( size_t n = 0; n < src.size(); ++n )
    {
        REQUIRE( dest[n] == pixget( src.data(), n ) );
    }

    REQUIRE( dest.back() == -1 );

    addFrame( dest.data(), src.data(), src.size() );

    for( size_t n = 0; n < src.size(); ++n )
    {
        REQUIRE( dest[n] == 2 * pixget( src.data(), n ) );
    }

    REQUIRE( dest.back() == -1 );
}
/// \endcond

/// The frame converters match the per-pixel converter for every supported type, without writing past the frame.
/** \ingroup pixaccess_unit_test
 */
TEST_CASE( "getPixFrame converts and accumulates whole frames", "[pixaccess]" )
{
    // clang-format off
    #ifdef PIXACCESS_TEST_DOXYGEN_REF
    getPixFrame;
    getPixFramePointer;
    #endif
    // clang-format on

    checkFrame<IMAGESTRUCT_UINT8>();
    checkFrame<IMAGESTRUCT_INT8>();
    checkFrame<IMAGESTRUCT_UINT16>();
    checkFrame<IMAGESTRUCT_INT16>();
    checkFrame<IMAGESTRUCT_UINT32>();
    checkFrame<IMAGESTRUCT_INT32>();
    checkFrame<IMAGESTRUCT_UINT64>();
    checkFrame<IMAGESTRUCT_INT64>();
    checkFrame<IMAGESTRUCT_FLOAT>();
    checkFrame<IMAGESTRUCT_DOUBLE>();
}

/// An unsupported type code gives no frame converter.
/** \ingroup pixaccess_unit_test
 */
TEST_CASE( "getPixFramePointer rejects unknown types", "[pixaccess]" )
{
    REQUIRE( getPixFramePointer<float, false>( -1 ) == nullptr );
    REQUIRE( getPixFramePointer<float, true>( -1 ) == nullptr );
}

} // namespace pixaccessTest
} // namespace libXWCTest
