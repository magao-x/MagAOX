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

    using streamCircBuff::fps;

    using streamCircBuff::frameGrabberRelayTiming;

    using streamCircBuff::loadImageIntoStream;

    using streamCircBuff::m_currImageTimestamp;

    using streamCircBuff::m_fpsDevice;

    using streamCircBuff::m_fpsTol;

    using streamCircBuff::m_indiP_fpsSource;

    using streamCircBuff::m_rewritten;

    using streamCircBuff::m_smSemaphore;

    using streamCircBuff::m_stagedFresh;

    using streamCircBuff::m_toFloat;

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

    /// Stage single-frame source frames at a fixed acquisition interval.
    void feed( uint64_t  count /**< [in] number of frames */,
               uint64_t  step /**< [in] counter increment per staged frame, more than 1 for missed frames */,
               double    interval /**< [in] acquisition interval between staged frames, in seconds */,
               uint64_t &cnt0 /**< [in/out] counter of the next frame */,
               double   &t /**< [in/out] acquisition time of the next frame, in seconds */ );
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

    IMAGE &src    = shmimMonitorT::m_imageStream;
    src           = IMAGE{};
    src.md        = &m_md;
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

void streamCircBuff_test::feed( uint64_t count, uint64_t step, double interval, uint64_t &cnt0, double &t )
{
    for( uint64_t n = 0; n < count; ++n, cnt0 += step )
    {
        m_md.cnt0  = cnt0;
        m_md.atime = { static_cast<time_t>( t ), static_cast<long>( ( t - static_cast<time_t>( t ) ) * 1e9 ) };
        processImage( frame( 0 ), dev::shmimT() );
        t += interval;
    }
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

    std::vector<int16_t>  out( 4 );
    uint64_t              published = 0;
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

/// With toFloat the output is float and the source pixels are converted.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff converts to float when configured", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::configureAcquisition;
    MagAOX::app::streamCircBuff::loadImageIntoStream;
    #endif
    // clang-format on

    streamCircBuff_test app;
    app.m_toFloat = true;
    REQUIRE( app.source( 1 ) == 0 );
    REQUIRE( app.processImage( app.frame( 0 ), dev::shmimT() ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    std::vector<float> out( 4 );
    REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
    REQUIRE( out == std::vector<float>{ 1, 2, 3, 4 } );
}

/// The measured source rate uses counters, so dropped frames do not bias it, and changes beyond the tolerance.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff measures the source frame rate", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::measureFPS;
    MagAOX::app::streamCircBuff::fps;
    #endif
    // clang-format on

    streamCircBuff_test app;
    REQUIRE( app.source( 1 ) == 0 );
    double   t    = 100;
    uint64_t cnt0 = 1;

    // nothing until a full second has been seen
    app.feed( 1000, 1, 5e-4, cnt0, t );
    REQUIRE( app.fps() == 0 );

    // a 2 kHz source with every other frame missed
    app.feed( 1100, 2, 1e-3, cnt0, t );
    REQUIRE( app.fps() == Approx( 2000 ).epsilon( 1e-3 ) );

    // a change within the tolerance keeps the reported rate
    app.feed( 2100, 1, 1. / 2010, cnt0, t );
    REQUIRE( app.fps() == Approx( 2000 ).epsilon( 1e-3 ) );

    // a change beyond the tolerance updates it, once a full second is at the new rate
    app.feed( 2200, 1, 1e-3, cnt0, t );
    REQUIRE( app.fps() == Approx( 1000 ).epsilon( 1e-3 ) );

    // a counter restart restarts the measurement
    cnt0 = 1;
    app.feed( 700, 1, 2e-3, cnt0, t );
    REQUIRE( app.fps() == Approx( 500 ).epsilon( 1e-3 ) );
}

/// With an fps device the frame rate follows its property within the tolerance, and is not measured.
/**
 * \ingroup streamCircBuff_unit_test
 */
TEST_CASE( "streamCircBuff reports the frame rate of an fps device", "[streamCircBuff]" )
{
    // clang-format off
    #ifdef STREAMCIRCBUFF_TEST_DOXYGEN_REF
    MagAOX::app::streamCircBuff::setCallBack_m_indiP_fpsSource;
    MagAOX::app::streamCircBuff::processImage;
    #endif
    // clang-format on

    streamCircBuff_test app;
    app.m_fpsDevice = "camwfs";
    app.m_indiP_fpsSource.setDevice( "camwfs" );
    app.m_indiP_fpsSource.setName( "fps" );
    REQUIRE( app.source( 1 ) == 0 );

    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( "camwfs" );
    ip.setName( "fps" );
    ip.add( pcf::IndiElement( "current" ) );

    ip["current"] = 2000;
    REQUIRE( app.setCallBack_m_indiP_fpsSource( ip ) == 0 );
    REQUIRE( app.fps() == 2000 );

    ip["current"] = 2010;
    REQUIRE( app.setCallBack_m_indiP_fpsSource( ip ) == 0 );
    REQUIRE( app.fps() == 2000 );

    ip["current"] = 1000;
    REQUIRE( app.setCallBack_m_indiP_fpsSource( ip ) == 0 );
    REQUIRE( app.fps() == 1000 );

    // frames at another rate are not measured
    double   t    = 100;
    uint64_t cnt0 = 1;
    app.feed( 1500, 1, 2e-3, cnt0, t );
    REQUIRE( app.fps() == 1000 );

    pcf::IndiProperty wrong( ip );
    wrong.setName( "other" );
    REQUIRE( app.setCallBack_m_indiP_fpsSource( wrong ) == -1 );
}

} // namespace streamCircBuffTest

} // namespace libXWCTest
