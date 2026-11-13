/** \file orcaUtils_test.cpp
 * \brief Catch2 tests for the ORCA-Quest2 utility helpers.
 *
 * \ingroup orcaCtrl_files
 */

/** \defgroup orcaCtrl_unit_test orcaCtrl Unit Tests
 * \brief Unit tests for the orcaCtrl application.
 *
 * \ingroup application_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include "../orcaUtils.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/// Namespace for `orcaCtrl` utility unit tests.
/** \ingroup orcaCtrl_unit_test
 */
namespace orcaCtrlTest
{

/// Verify `orcaSelectFrame()` publishes frames in order and maps them to the right ring index.
/** Covers the no-backlog and backlog cases, ring wrap-around, and a capture that didn't start at
 * ring index 0.
 *
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl frame selection publishes frames in order", "[orcaUtils]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaFrameSelection docSel;
    orcaSelectFrame( docSel, 0, 1, 0, 10, 2 );
    #endif
    // clang-format on

    orcaFrameSelection sel;

    SECTION( "the newest frame is selected when there is no backlog" )
    {
        // 5 frames transferred into a 10-frame ring starting at index 0; frame 4 is at index 4
        REQUIRE( orcaSelectFrame( sel, 4, 5, 4, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 4 );
        REQUIRE( sel.ringIndex == 4 );
        REQUIRE( sel.dropped == 0 );
    }

    SECTION( "the oldest unpublished frame is selected from a backlog" )
    {
        // 7 frames transferred, frame 4 not yet published: publish 4, then 5, then 6
        REQUIRE( orcaSelectFrame( sel, 4, 7, 6, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 4 );
        REQUIRE( sel.ringIndex == 4 );
        REQUIRE( sel.dropped == 0 );

        REQUIRE( orcaSelectFrame( sel, 5, 7, 6, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 5 );
        REQUIRE( sel.ringIndex == 5 );
    }

    SECTION( "ring indices wrap around" )
    {
        // 12 frames transferred into a 10-frame ring: frame 11 is at index 1, frame 9 at index 9
        REQUIRE( orcaSelectFrame( sel, 9, 12, 1, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 9 );
        REQUIRE( sel.ringIndex == 9 );
        REQUIRE( sel.dropped == 0 );

        REQUIRE( orcaSelectFrame( sel, 10, 12, 1, 10, 2 ) == 0 );
        REQUIRE( sel.ringIndex == 0 );
    }

    SECTION( "ring indices are relative to the newest frame, not to index 0" )
    {
        // Capture started at ring index 3: 4 frames transferred, newest (frame 3) at index 6
        REQUIRE( orcaSelectFrame( sel, 0, 4, 6, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 0 );
        REQUIRE( sel.ringIndex == 3 );
    }

    SECTION( "a backlog that exactly fills the safe part of the ring is not dropped" )
    {
        // ring 10, margin 2: up to 8 unpublished frames are safe
        REQUIRE( orcaSelectFrame( sel, 2, 10, 9, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 2 );
        REQUIRE( sel.ringIndex == 2 );
        REQUIRE( sel.dropped == 0 );
    }
}

/// Verify `orcaSelectFrame()` skips overwritten frames, counts them, and rejects invalid input.
/**
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl frame selection handles ring-buffer overruns", "[orcaUtils]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaFrameSelection docSel;
    orcaSelectFrame( docSel, 0, 1, 0, 10, 2 );
    #endif
    // clang-format on

    orcaFrameSelection sel;

    SECTION( "an overrun of one frame skips one frame" )
    {
        // 9 unpublished frames (2..10) but only 8 are safe: skip frame 2, publish frame 3
        REQUIRE( orcaSelectFrame( sel, 2, 11, 0, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 3 );
        REQUIRE( sel.dropped == 1 );
        REQUIRE( sel.ringIndex == 3 );
    }

    SECTION( "a large overrun skips to the oldest safe frame" )
    {
        // 1000 frames transferred, nothing published since frame 0
        REQUIRE( orcaSelectFrame( sel, 0, 1000, 999 % 10, 10, 2 ) == 0 );
        REQUIRE( sel.frame == 992 );
        REQUIRE( sel.dropped == 992 );
        REQUIRE( sel.ringIndex == 992 % 10 );
    }

    SECTION( "invalid inputs are rejected" )
    {
        REQUIRE( orcaSelectFrame( sel, 5, 5, 4, 10, 2 ) == -1 );  // nothing unpublished
        REQUIRE( orcaSelectFrame( sel, -1, 5, 4, 10, 2 ) == -1 ); // negative frame
        REQUIRE( orcaSelectFrame( sel, 0, 5, 4, 0, 0 ) == -1 );   // empty ring
        REQUIRE( orcaSelectFrame( sel, 0, 5, 4, 10, 10 ) == -1 ); // margin covers the whole ring
        REQUIRE( orcaSelectFrame( sel, 0, 5, 10, 10, 2 ) == -1 ); // newest index outside the ring
        REQUIRE( orcaSelectFrame( sel, 0, 5, -1, 10, 2 ) == -1 ); // no newest frame yet
    }
}

/// Verify `orcaBufferFrames()` sizes the DCAM ring buffer by time and memory.
/** Uses ORCA-Quest2 frame sizes: full frame 4096×2304, and 256×256 and 128×128 ROIs, at 16 bits.
 *
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl buffer sizing covers 0.1 s within the memory limit", "[orcaUtils]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaBufferFrames( 0, 0, 0, 0, 0, 0 );
    #endif
    // clang-format on

    constexpr uint64_t maxBytes = 536870912; // 0.5 GB, the camera.circBuffMaxBytes default

    SECTION( "128x128 at 2013 Hz holds 0.1 s of frames" )
    {
        REQUIRE( orcaBufferFrames( 128 * 128 * 2, 2012.88, 0.1, maxBytes, 10, 10000 ) == 202 );
    }

    SECTION( "256x256 at 1044 Hz holds 0.1 s of frames" )
    {
        REQUIRE( orcaBufferFrames( 256 * 256 * 2, 1044.0, 0.1, maxBytes, 10, 10000 ) == 105 );
    }

    SECTION( "full frame at 120 Hz holds 0.1 s of frames, within the memory limit" )
    {
        const int32_t n = orcaBufferFrames( 4096ULL * 2304 * 2, 120.0, 0.1, maxBytes, 10, 10000 );

        REQUIRE( n == 12 );
        REQUIRE( static_cast<uint64_t>( n ) * 4096 * 2304 * 2 <= maxBytes );
    }

    SECTION( "the memory limit wins over the time span" )
    {
        // 18.9 MB frames with a 100 MB limit: 5 frames
        REQUIRE( orcaBufferFrames( 4096ULL * 2304 * 2, 120.0, 1.0, 100000000, 10, 10000 ) == 5 );
    }

    SECTION( "the minimum and maximum frame counts are applied" )
    {
        REQUIRE( orcaBufferFrames( 1024, 1.0, 0.1, maxBytes, 10, 10000 ) == 10 );         // slow: minimum
        REQUIRE( orcaBufferFrames( 1024, 1.0e6, 0.1, maxBytes, 10, 10000 ) == 10000 );    // fast: maximum
        REQUIRE( orcaBufferFrames( 1024, 0.0, 0.1, maxBytes, 10, 10000 ) == 10 );         // unknown rate
        REQUIRE( orcaBufferFrames( 4096ULL * 2304 * 2, 120.0, 0.1, 1, 10, 10000 ) == 3 ); // never below 3
    }
}

/// Verify `orcaFramestampAdvanceOk()` accepts expected advances, including 16-bit wraps.
/**
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl framestamp check allows for a 16-bit counter", "[orcaUtils]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaFramestampAdvanceOk( 0, 0, 0 );
    #endif
    // clang-format on

    SECTION( "a normal advance is accepted" )
    {
        REQUIRE( orcaFramestampAdvanceOk( 100, 2113, 2013 ) );
    }

    SECTION( "an advance across a 16-bit wrap is accepted" )
    {
        // 65530 -> 65535 -> 0 -> 2006 is 2012 frames
        REQUIRE( orcaFramestampAdvanceOk( 65530, 2006, 2012 ) );
    }

    SECTION( "an advance across a 32-bit wrap is accepted" )
    {
        REQUIRE( orcaFramestampAdvanceOk( 2147483647, -2147483648, 1 ) );
    }

    SECTION( "a wrong advance is rejected" )
    {
        REQUIRE_FALSE( orcaFramestampAdvanceOk( 100, 2114, 2013 ) );
        REQUIRE_FALSE( orcaFramestampAdvanceOk( 65530, 2007, 2012 ) );
    }
}

/// Verify `orcaClampROI()` rounds ROIs to 4-pixel steps and keeps them on the 4096×2304 sensor.
/** Includes the full-frame request with an off-center center that caused an ERROR loop in hardware
 * test round 1 (agents/plans/2026-10/orcaCtrl-latency-new.md).
 *
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl ROI clamping keeps the ROI valid on the sensor", "[orcaUtils]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    float docX = 2047, docY = 1151;
    int   docW = 128, docH = 128;
    orcaClampROI( docX, docY, docW, docH, 4096, 2304, 4 );
    #endif
    // clang-format on

    constexpr int32_t W = 4096;
    constexpr int32_t H = 2304;

    SECTION( "a valid ROI is unchanged" )
    {
        float x = 2047, y = 1151;
        int   w = 128, h = 128;

        REQUIRE( orcaClampROI( x, y, w, h, W, H, 4 ) == false );
        REQUIRE( x == 2047 );
        REQUIRE( y == 1151 );
        REQUIRE( w == 128 );
        REQUIRE( h == 128 );
    }

    SECTION( "a full-frame request with an off-center center is moved onto the sensor" )
    {
        float x = 511, y = 511;
        int   w = 4096, h = 2304;

        REQUIRE( orcaClampROI( x, y, w, h, W, H, 4 ) == true );
        REQUIRE( w == 4096 );
        REQUIRE( h == 2304 );
        REQUIRE( x == 2047 ); // corner 0
        REQUIRE( y == 1151 ); // corner 0
    }

    SECTION( "sizes are rounded down to 4-pixel steps and limited to the sensor" )
    {
        float x = 2047, y = 1151;
        int   w = 130, h = 5000;

        REQUIRE( orcaClampROI( x, y, w, h, W, H, 4 ) == true );
        REQUIRE( w == 128 );
        REQUIRE( h == 2304 );

        w = 0;
        h = 2;
        orcaClampROI( x, y, w, h, W, H, 4 );
        REQUIRE( w == 4 );
        REQUIRE( h == 4 );
    }

    SECTION( "corners are rounded to the nearest 4-pixel step" )
    {
        float x = 1000, y = 1151;
        int   w = 128, h = 128;

        REQUIRE( orcaClampROI( x, y, w, h, W, H, 4 ) == true );
        REQUIRE( x == 999 ); // corner 937 -> 936
    }

    SECTION( "an ROI off the right and bottom edges is clamped inside" )
    {
        float x = 4090, y = 2300;
        int   w = 128, h = 128;

        REQUIRE( orcaClampROI( x, y, w, h, W, H, 4 ) == true );
        REQUIRE( x == 4031 ); // corner 3968
        REQUIRE( y == 2239 ); // corner 2176
    }

    SECTION( "a half-pixel center is truncated the same way setDcamRoi() does" )
    {
        float x = 511.5, y = 511.5;
        int   w = 1024, h = 1024;

        orcaClampROI( x, y, w, h, W, H, 4 );
        REQUIRE( x == 511 ); // corner 0
        REQUIRE( y == 511 );
    }
}

} // namespace orcaCtrlTest
} // namespace libXWCTest
