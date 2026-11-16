/** \file streamCopy_test.cpp
 * \brief Catch2 tests for the non-temporal memory copy.
 */

#include "../../../tests/testXWC.hpp"

#include "../streamCopy.hpp"

#include <vector>

namespace libXWCTest
{
namespace streamCopyTest
{

/// The streaming copy matches memcpy for any destination alignment and length, without writing outside the copy.
/**
 * \ingroup common_unit_test
 */
TEST_CASE( "streamCopy copies exactly", "[streamCopy]" )
{
    // clang-format off
    #ifdef STREAMCOPY_TEST_DOXYGEN_REF
    MagAOX::streamCopy;
    #endif
    // clang-format on

    std::vector<char> src( 300 );
    for( size_t n = 0; n < src.size(); ++n )
    {
        src[n] = static_cast<char>( n * 7 + 1 );
    }

    for( size_t offset = 0; offset < 16; ++offset )
    {
        for( size_t nbytes : { 0, 1, 15, 16, 17, 31, 64, 255 } )
        {
            alignas( 16 ) char dest[320];
            memset( dest, 0, sizeof( dest ) );

            MagAOX::streamCopy( dest + offset, src.data() + 3, nbytes );

            REQUIRE( memcmp( dest + offset, src.data() + 3, nbytes ) == 0 );

            for( size_t n = 0; n < sizeof( dest ); ++n )
            {
                if( n < offset || n >= offset + nbytes )
                {
                    REQUIRE( dest[n] == 0 );
                }
            }
        }
    }
}

} // namespace streamCopyTest
} // namespace libXWCTest
