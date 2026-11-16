/** \file ocamDescrambler_test.cpp
 * \brief Hardware-free pixel equivalence tests against the real vendored OCAM SDK.
 * \ingroup ocam2KCtrl_descramble_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include "../ocamDescrambler.hpp"
#include "ocamDescrambleFixture.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>

namespace libXWCTest
{
namespace ocam2KCtrlTest
{

using MagAOX::app::ocamDescrambler;

/// \cond OCAM2KCTRL_TEST_SUPPORT

/// Image extents for one application-supported SDK mode.
struct descrambleMode
{
    ocam2_mode mode;        ///< SDK mode initialized by this case.
    size_t     rawPixels;   ///< Complete SDK input extent, including non-image words.
    size_t     imagePixels; ///< Number of output words to compare.
};

constexpr std::array<descrambleMode, 4> descrambleModes{
    { { OCAM2_NORMAL, OCAM2_PIXELS_RAW_NORMAL, OCAM2_PIXELS_IMAGE_NORMAL },
      { OCAM2_BINNING2x2, OCAM2_PIXELS_RAW_BINNING2x2, OCAM2_PIXELS_IMAGE_BINNING2x2 },
      { OCAM2_BINNING1x3, OCAM2_PIXELS_RAW_BINNING1x3, OCAM2_PIXELS_IMAGE_BINNING1x3 },
      { OCAM2_BINNING1x4, OCAM2_PIXELS_RAW_BINNING1x4, OCAM2_PIXELS_IMAGE_BINNING1x4 } } };

/// Own an isolated temporary table and its single SDK handle.
class descrambleFixture
{
    std::string m_path;     ///< Temporary CSV removed after the SDK table is released.
    ocam2_id    m_id{ -1 }; ///< Valid SDK handle, including zero; negative means closed.

  public:
    /// Write a complete valid table with a terminator inside the SDK's file allocation.
    explicit descrambleFixture( const std::vector<int> &table /**< [in] Normal-mode mapping. */ );

    /// Close the active handle and remove the temporary table.
    ~descrambleFixture();

    /// Initialize one SDK mode after releasing any previous handle.
    ocam2_id open( ocam2_mode  mode,          /**< [in] SDK mode to initialize. */
                   const char *path = nullptr /**< [in] Optional external valid table; null uses the fixture. */
    );

    /// Release the handle and process-global SDK mapping before another fixture runs.
    void close();
};

descrambleFixture::descrambleFixture( const std::vector<int> &table )
{
    char path[] = "/tmp/ocam-descramble-test-XXXXXX";
    int  fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    m_path = path;
    ::close( fd );

    std::ofstream file( m_path, std::ios::binary );
    REQUIRE( file.good() );

    for( int index : table )
    {
        file << index << ',';
    }

    // The SDK allocates exactly the file size, without appending a C string terminator.
    file.put( '\0' );
    file.close();
    REQUIRE( !file.fail() );
}

descrambleFixture::~descrambleFixture()
{
    close();
    std::remove( m_path.c_str() );
}

ocam2_id descrambleFixture::open( ocam2_mode mode, const char *path )
{
    close();
    REQUIRE( ocam2_init( mode, path == nullptr ? m_path.c_str() : path, &m_id ) == OCAM2_OK );
    return m_id;
}

void descrambleFixture::close()
{
    if( m_id >= 0 )
    {
        ocam2_exit( m_id );
        m_id = -1;
    }
}

/// Fill every raw word with a discriminating 16-bit pattern, preserving signed bit representations.
void fillDescrambleRaw( short       *raw,    /**< [out] Complete input allocation. */
                        size_t       count,  /**< [in] Input word count. */
                        unsigned int pattern /**< [in] Index, inverse, sign, alternating, zero, or random selection. */
);

/// Compare both optimized kernels with the SDK while checking input immutability and output guards.
void verifyDescrambleFrame( ocam2_id               id,          /**< [in] Initialized reference SDK handle. */
                            const ocamDescrambler &descrambler, /**< [in] Prepared candidate. */
                            unsigned int           pattern,     /**< [in] Raw input pattern. */
                            size_t                 offset       /**< [in] Guard words preceding each destination. */
);

void fillDescrambleRaw( short *raw, size_t count, unsigned int pattern )
{
    std::mt19937 random( 0x2478u + pattern );

    for( size_t index = 0; index < count; ++index )
    {
        uint16_t value = static_cast<uint16_t>( index );

        if( pattern == 1 )
        {
            value = static_cast<uint16_t>( ~value );
        }
        else if( pattern == 2 )
        {
            value = index % 2 == 0 ? 0x8000 : 0x7fff;
        }
        else if( pattern == 3 )
        {
            value = index % 2 == 0 ? 0xaaaa : 0x5555;
        }
        else if( pattern == 4 )
        {
            value = 0;
        }
        else if( pattern == 5 )
        {
            value = static_cast<uint16_t>( random() );
        }

        std::memcpy( raw + index, &value, sizeof( value ) );
    }

    const unsigned int number = 0xfedc0000u + pattern;
    std::memcpy( reinterpret_cast<char *>( raw ) + OCAM2_IMAGE_NB_OFFSET, &number, sizeof( number ) );
}

void verifyDescrambleFrame( ocam2_id id, const ocamDescrambler &descrambler, unsigned int pattern, size_t offset )
{
    const short        guard = 0x5a5a;
    std::vector<short> raw( descrambler.rawPixels() + 16, guard );
    fillDescrambleRaw( raw.data() + 8, descrambler.rawPixels(), pattern );
    const auto before = raw;

    std::vector<short> expected( descrambler.imagePixels() + offset + 8, guard );
    auto               mapped    = expected;
    auto               optimized = expected;
    unsigned int       number    = 0;
    ocam2_descramble( id, &number, expected.data() + offset, raw.data() + 8 );

    REQUIRE( number == 0xfedc0000u + pattern );

    descrambler.mapped( mapped.data() + offset, raw.data() + 8 );
    descrambler.descramble( optimized.data() + offset, raw.data() + 8 );

    REQUIRE( mapped == expected );
    REQUIRE( optimized == expected );
    REQUIRE( raw == before );
    REQUIRE( std::all_of( expected.begin(),
                          expected.begin() + offset,
                          [guard]( short value )
                          {
                              return value == guard;
                          } ) );
    REQUIRE( std::all_of( expected.end() - 8,
                          expected.end(),
                          [guard]( short value )
                          {
                              return value == guard;
                          } ) );
}

/// \endcond

/// Compare every app mode with the real SDK for structured and arbitrary valid maps and 16-bit inputs.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocamDescrambler reproduces the real SDK in every app mode", "[ocamDescrambler]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocamDescrambler::prepare( 0 ); ocamDescrambler::ready(); ocamDescrambler::structured();
    ocamDescrambler::rawPixels(); ocamDescrambler::imagePixels();
    ocamDescrambler::mapped( nullptr, nullptr ); ocamDescrambler::descramble( nullptr, nullptr );
    #endif
    // clang-format on

    for( bool regular : { true, false } )
    {
        auto table = ocamDescrambleTable();

        if( !regular )
        {
            // Deliberately non-affine, duplicate indices and metadata words are valid mappings.
            for( size_t index = 0; index < table.size(); ++index )
            {
                table[index] = ( index * 37 + index / 7 ) % OCAM2_PIXELS_RAW_BINNING1x4;
            }
        }

        descrambleFixture fixture( table );
        ocamDescrambler   descrambler;

        for( const auto &mode : descrambleModes )
        {
            CAPTURE( regular, mode.mode );
            const auto id = fixture.open( mode.mode );

            REQUIRE( id == 0 );
            REQUIRE( descrambler.prepare( id ) == 0 );
            REQUIRE( descrambler.ready() );
            REQUIRE( descrambler.structured() == ( regular && mode.mode == OCAM2_BINNING2x2 ) );
            REQUIRE( descrambler.rawPixels() == mode.rawPixels );
            REQUIRE( descrambler.imagePixels() == mode.imagePixels );

            for( unsigned int pattern = 0; pattern < 6; ++pattern )
            {
                for( size_t offset : { 1u, 8u } )
                {
                    CAPTURE( pattern, offset );
                    verifyDescrambleFrame( id, descrambler, pattern, offset );
                }
            }

            REQUIRE( descrambler.prepare( id ) == 0 );
            verifyDescrambleFrame( id, descrambler, 5, 8 );
        }
    }
}

/// Require complete validation before choosing the structured bin2 traversal.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocamDescrambler falls back when any structured run is different", "[ocamDescrambler]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocamDescrambler::prepare( 0 ); ocamDescrambler::structured(); ocamDescrambler::descramble( nullptr, nullptr );
    #endif
    // clang-format on

    for( size_t changed : { 0u, 58u, 60u, 43200u, 57598u } )
    {
        CAPTURE( changed );
        auto table = ocamDescrambleTable();
        ++table[changed];
        descrambleFixture fixture( table );
        const auto        id = fixture.open( OCAM2_BINNING2x2 );
        ocamDescrambler   descrambler;

        REQUIRE( descrambler.prepare( id ) == 0 );
        REQUIRE( descrambler.ready() );
        REQUIRE( !descrambler.structured() );
        verifyDescrambleFrame( id, descrambler, 0, 1 );
        verifyDescrambleFrame( id, descrambler, 5, 8 );
    }
}

/// Reset stale state on failure and retain owned mapping data after the SDK releases its global table.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocamDescrambler owns prepared maps and clears failed preparations", "[ocamDescrambler]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocamDescrambler::prepare( 0 ); ocamDescrambler::clear(); ocamDescrambler::ready();
    ocamDescrambler::structured(); ocamDescrambler::rawPixels(); ocamDescrambler::imagePixels();
    ocamDescrambler::descramble( nullptr, nullptr );
    #endif
    // clang-format on

    descrambleFixture fixture( ocamDescrambleTable() );
    ocamDescrambler   descrambler;
    REQUIRE( !descrambler.ready() );
    REQUIRE( !descrambler.structured() );
    REQUIRE( descrambler.rawPixels() == 0 );
    REQUIRE( descrambler.imagePixels() == 0 );

    for( const auto &mode : descrambleModes )
    {
        const auto id = fixture.open( mode.mode );
        REQUIRE( descrambler.prepare( id ) == 0 );

        std::vector<short> raw( mode.rawPixels );
        fillDescrambleRaw( raw.data(), raw.size(), 5 );
        std::vector<short> expected( mode.imagePixels );
        std::vector<short> actual( mode.imagePixels );
        unsigned int       number = 0;
        ocam2_descramble( id, &number, expected.data(), raw.data() );
        fixture.close();

        for( int repeat = 0; repeat < 4; ++repeat )
        {
            descrambler.descramble( actual.data(), raw.data() );
            REQUIRE( actual == expected );
        }

        descrambler.clear();
        REQUIRE( !descrambler.ready() );
        REQUIRE( !descrambler.structured() );
        REQUIRE( descrambler.rawPixels() == 0 );
        REQUIRE( descrambler.imagePixels() == 0 );
    }

    for( const auto invalid : { -1, OCAM2_SDK_MAX_CAMERA } )
    {
        REQUIRE( descrambler.prepare( fixture.open( OCAM2_BINNING2x2 ) ) == 0 );
        REQUIRE( descrambler.prepare( invalid ) == -1 );
        REQUIRE( !descrambler.ready() );
        REQUIRE( !descrambler.structured() );
        REQUIRE( descrambler.rawPixels() == 0 );
        REQUIRE( descrambler.imagePixels() == 0 );
    }

    REQUIRE( descrambler.prepare( fixture.open( OCAM2_NORMAL ) ) == 0 );
    REQUIRE( descrambler.prepare( fixture.open( OCAM2_CROPPING240x120 ) ) == -1 );
    REQUIRE( !descrambler.ready() );
    REQUIRE( descrambler.rawPixels() == 0 );
    REQUIRE( descrambler.imagePixels() == 0 );
}

/// Permit operator-supplied valid calibration tables without requiring instrument configuration in CI.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocamDescrambler agrees with an optional supplied calibration table", "[ocamDescrambler]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocamDescrambler::prepare( 0 ); ocamDescrambler::mapped( nullptr, nullptr );
    ocamDescrambler::descramble( nullptr, nullptr );
    #endif
    // clang-format on

    const char *path = std::getenv( "MAGAOX_OCAM_DESCRAMBLE_TABLE" );

    if( path == nullptr )
    {
        SUCCEED( "Set MAGAOX_OCAM_DESCRAMBLE_TABLE to also test a valid external SDK table." );
        return;
    }

    descrambleFixture fixture( ocamDescrambleTable() );
    ocamDescrambler   descrambler;

    for( const auto &mode : descrambleModes )
    {
        CAPTURE( path, mode.mode );
        const auto id = fixture.open( mode.mode, path );
        REQUIRE( descrambler.prepare( id ) == 0 );
        REQUIRE( descrambler.rawPixels() == mode.rawPixels );
        REQUIRE( descrambler.imagePixels() == mode.imagePixels );

        for( unsigned int pattern = 0; pattern < 6; ++pattern )
        {
            verifyDescrambleFrame( id, descrambler, pattern, 1 );
        }
    }
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
