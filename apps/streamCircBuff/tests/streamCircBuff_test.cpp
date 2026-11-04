/** \file streamCircBuff_test.cpp
 * \brief Catch2 tests for the streamCircBuff app.
 *
 * \ingroup streamCircBuff_files
 */

#include "../../../tests/testXWC.hpp"

#include "../streamCircBuff.hpp"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace MagAOX
{
namespace app
{
/// Expose the real callbacks with a test-owned source image, without shared memory or INDI.
class streamCircBuff_test : public streamCircBuff
{
  public:
    using streamCircBuff::acquireAndCheckValid;

    using streamCircBuff::frameGrabberRelayTiming;

    using streamCircBuff::loadImageIntoStream;

    using streamCircBuff::m_currImageTimestamp;

    using streamCircBuff::m_rewritten;

    using streamCircBuff::m_smSemaphore;

    using streamCircBuff::m_stagedFresh;

    using streamCircBuff::processImage;

    static constexpr uint32_t c_width  = 2; ///< Source frame width.
    static constexpr uint32_t c_height = 2; ///< Source frame height.

    IMAGE_METADATA m_md{}; ///< Source metadata.

    std::vector<int16_t> m_pixels; ///< Source frames.

    std::vector<timespec> m_atimes; ///< Source per-frame acquisition times.

    std::vector<timespec> m_wtimes; ///< Source per-frame write times.

    std::vector<uint64_t> m_cnts; ///< Source per-frame counters.

    /// Initialize the app semaphore normally created at startup.
    streamCircBuff_test();

    /// Release the app semaphore.
    ~streamCircBuff_test();

    /// Install an int16 source with `depth` frames, values 1, 2, ... in frame order, and allocate.
    int source( uint32_t depth /**< [in] number of source frames */ );

    /// Pointer to the start of a source frame.
    void *frame( size_t slot /**< [in] source frame index */ );
};

streamCircBuff_test::streamCircBuff_test()
{
    sem_init( &m_smSemaphore, 0, 0 );
}

streamCircBuff_test::~streamCircBuff_test()
{
    sem_destroy( &m_smSemaphore );
}

int streamCircBuff_test::source( uint32_t depth )
{
    size_t npix = c_width * c_height;
    m_pixels.resize( npix * depth );
    for( size_t n = 0; n < m_pixels.size(); ++n )
    {
        m_pixels[n] = n + 1;
    }
    m_atimes.assign( depth, timespec{ 0, 0 } );
    m_wtimes.assign( depth, timespec{ 0, 0 } );
    m_cnts.assign( depth, 0 );

    IMAGE &src = shmimMonitorT::m_imageStream;
    src        = IMAGE{};
    src.md     = &m_md;
    src.array.raw = m_pixels.data();
    if( depth > 1 )
    {
        src.atimearray     = m_atimes.data();
        src.writetimearray = m_wtimes.data();
        src.cntarray       = m_cnts.data();
    }

    shmimMonitorT::m_width    = c_width;
    shmimMonitorT::m_height   = c_height;
    shmimMonitorT::m_depth    = depth;
    shmimMonitorT::m_dataType = _DATATYPE_INT16;
    shmimMonitorT::m_typeSize = sizeof( int16_t );

    int rv = allocate( dev::shmimT() );
    if( rv < 0 || ( rv = configureAcquisition() ) < 0 )
    {
        return rv;
    }
    frameGrabberT::m_typeSize = ImageStreamIO_typesize( frameGrabberT::m_dataType ); // as the worker does
    return 0;
}

void *streamCircBuff_test::frame( size_t slot )
{
    return m_pixels.data() + slot * c_width * c_height;
}
} // namespace app
} // namespace MagAOX
/// \endcond

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup streamCircBuff_unit_test streamCircBuff Unit Tests
 * \brief Unit tests for the streamCircBuff application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `streamCircBuff` unit tests.
/** \ingroup streamCircBuff_unit_test
 */
namespace streamCircBuffTest
{

/// True if two timespecs are equal.
bool same( const timespec &a /**< [in] first time */, const timespec &b /**< [in] second time */ )
{
    return a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec;
}

/// A relayed frame keeps the source pixels and data type, acquisition time, write time and counter.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff relays the source frame and its timing", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::processImage;
    MagAOX::app::streamCircBuff::acquireAndCheckValid;
    MagAOX::app::streamCircBuff::loadImageIntoStream;
    MagAOX::app::streamCircBuff::frameGrabberRelayTiming;
    #endif
    // clang-format on

    streamCircBuff_test app;
    REQUIRE( app.source( 1 ) == 0 );
    app.m_md.atime     = { 5, 6 };
    app.m_md.writetime = { 7, 8 };
    app.m_md.cnt0      = 41;

    REQUIRE( app.processImage( app.frame( 0 ), dev::shmimT() ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    std::vector<int16_t> out( 4 );
    REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
    REQUIRE( out == std::vector<int16_t>{ 1, 2, 3, 4 } );
    REQUIRE( same( app.m_currImageTimestamp, { 5, 6 } ) );

    timespec wt{ 0, 0 };
    uint64_t cnt0 = 1;
    REQUIRE( app.frameGrabberRelayTiming( wt, cnt0 ) == 0 );
    REQUIRE( same( wt, { 7, 8 } ) );
    REQUIRE( cnt0 == 41 );
}

/// A single-frame source written during the copy is dropped and counted.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff drops a frame rewritten during the copy", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::processImage;
    #endif
    // clang-format on

    streamCircBuff_test app;
    REQUIRE( app.source( 1 ) == 0 );
    app.m_md.write = 1;

    REQUIRE( app.processImage( app.frame( 0 ), dev::shmimT() ) == 0 );
    REQUIRE( app.m_rewritten == 1 );
    REQUIRE_FALSE( app.m_stagedFresh );

    int posts = -1;
    sem_getvalue( &app.m_smSemaphore, &posts );
    REQUIRE( posts == 0 );
}

/// Extra wakeups do not publish the same staged frame twice.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff publishes each staged frame once", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::acquireAndCheckValid;
    #endif
    // clang-format on

    streamCircBuff_test app;
    REQUIRE( app.source( 1 ) == 0 );

    REQUIRE( app.processImage( app.frame( 0 ), dev::shmimT() ) == 0 );
    sem_post( &app.m_smSemaphore );

    std::vector<int16_t> out( 4 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 1 );
}

/// The published counter keeps increasing after the source counter restarts, and keeps source gaps.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff keeps the published counter increasing", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::frameGrabberRelayTiming;
    #endif
    // clang-format on

    streamCircBuff_test app;
    REQUIRE( app.source( 1 ) == 0 );

    std::vector<int16_t> out( 4 );
    uint64_t           published = 0;
    std::vector<uint64_t> got;
    for( uint64_t srcCnt0 : { 41, 42, 3, 4, 10 } )
    {
        app.m_md.cnt0 = srcCnt0;
        REQUIRE( app.processImage( app.frame( 0 ), dev::shmimT() ) == 0 );
        REQUIRE( app.acquireAndCheckValid() == 0 );
        REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );

        timespec wt;
        uint64_t cnt0 = published + 1;
        REQUIRE( app.frameGrabberRelayTiming( wt, cnt0 ) == 0 );
        published = cnt0;
        got.push_back( cnt0 );
    }

    REQUIRE( got == std::vector<uint64_t>{ 41, 42, 43, 44, 50 } );
}

/// A multi-frame source is copied from the current frame with that frame's timing, while another frame is written.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff relays a multi-frame source using its per-frame timing", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::processImage;
    #endif
    // clang-format on

    streamCircBuff_test app;
    REQUIRE( app.source( 3 ) == 0 );
    app.m_atimes[1] = { 11, 12 };
    app.m_wtimes[1] = { 13, 14 };
    app.m_cnts[1]   = 77;
    app.m_md.cnt0   = 78;
    app.m_md.write  = 1;

    REQUIRE( app.processImage( app.frame( 1 ), dev::shmimT() ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    std::vector<int16_t> out( 4 );
    REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
    REQUIRE( out == std::vector<int16_t>{ 5, 6, 7, 8 } );
    REQUIRE( same( app.m_currImageTimestamp, { 11, 12 } ) );

    timespec wt;
    uint64_t cnt0 = 1;
    REQUIRE( app.frameGrabberRelayTiming( wt, cnt0 ) == 0 );
    REQUIRE( same( wt, { 13, 14 } ) );
    REQUIRE( cnt0 == 77 );
}

} // namespace streamCircBuffTest

} // namespace libXWCTest
