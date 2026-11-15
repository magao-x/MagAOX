/** \file loopTimingStream_test.cpp
 * \brief Offline metadata polling, continuity and lifecycle tests for loop timing.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingStream.hpp"

#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

using Stream    = MagAOX::app::loopTiming::Stream;
using Publisher = MagAOX::app::dev::dmSdkTimingStream;

/// \cond LOOP_TIMING_STREAM_FIXTURES
namespace
{

/// Keep all image operations inside one process-private workstation directory.
class Directory
{
  public:
    std::string m_path; ///< Temporary directory retained until every test mapping is closed.

    std::string m_previous; ///< Original environment value, when present.

    bool m_hadPrevious{ false }; ///< Whether teardown must restore an environment value.

    /// Select the private directory before ImageStreamIO's first cached lookup.
    Directory();

    /// Restore the environment and remove only private test files.
    ~Directory();
};

Directory::Directory()
{
    char       pattern[] = "/tmp/loop-timing-stream-XXXXXX";
    const auto path      = mkdtemp( pattern );

    if( !path )
    {
        throw std::runtime_error( "cannot create private loop timing stream directory" );
    }

    m_path              = path;
    const auto previous = getenv( "MILK_SHM_DIR" );
    m_hadPrevious       = previous != nullptr;

    if( previous )
    {
        m_previous = previous;
    }

    REQUIRE( setenv( "MILK_SHM_DIR", m_path.c_str(), 1 ) == 0 );
}

Directory::~Directory()
{
    if( m_hadPrevious )
    {
        setenv( "MILK_SHM_DIR", m_previous.c_str(), 1 );
    }
    else
    {
        unsetenv( "MILK_SHM_DIR" );
    }

    std::filesystem::remove_all( m_path );
}

/// Return the fixture directory retained until all test objects are destroyed.
Directory &directory();

Directory &directory()
{
    static Directory value;
    return value;
}

/// Create unique names without touching a default or instrument stream directory.
std::string uniqueName();

std::string uniqueName()
{
    static_cast<void>( directory() );
    static unsigned sequence = 0;
    return "loopTiming" + std::to_string( getpid() ) + "_" + std::to_string( ++sequence );
}

/// Own a synthetic camera and an independent metadata-only reader.
class Camera
{
  public:
    IMAGE m_image{}; ///< Test-owned source image with no reader semaphores.

    std::string m_name; ///< Unique private source name.

    Stream m_reader; ///< Reader closes before the source mapping is released.

    /// Create a source with one already published frame.
    Camera();

    /// Close mappings without invoking semaphore destruction.
    ~Camera();

    /// Publish deterministic metadata through atomic writes, without notification helpers.
    void publish( uint64_t counter /**< [in] Source publication identity. */,
                  int64_t  atimeNs /**< [in] Acquisition timestamp; publication is 100 ns later. */ );
};

Camera::Camera() : m_name( uniqueName() )
{
    uint32_t size[]{ 1, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu( &m_image, m_name.c_str(), 2, size, _DATATYPE_UINT8, -1, 1, 0, 0, 0, 0 ) == 0 );
    publish( 10, 10000000000 );
}

Camera::~Camera()
{
    m_reader.close();
    ImageStreamIO_closeIm( &m_image );
}

void Camera::publish( uint64_t counter, int64_t atimeNs )
{
    __atomic_store_n( &m_image.md->write, uint8_t( 1 ), __ATOMIC_SEQ_CST );
    __atomic_store_n( &m_image.md->atime.tv_sec, time_t( atimeNs / 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_image.md->atime.tv_nsec, long( atimeNs % 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_image.md->writetime.tv_sec, time_t( ( atimeNs + 100 ) / 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_image.md->writetime.tv_nsec, long( ( atimeNs + 100 ) % 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_image.md->cnt0, counter, __ATOMIC_RELEASE );
    __atomic_store_n( &m_image.md->write, uint8_t( 0 ), __ATOMIC_RELEASE );
}

} // namespace
/// \endcond

/** \brief Baseline old frames and distinguish progression, loss, reset and ordinary unchanged polls.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing stream reports publication continuity", "[loopTimingMonitor][stream]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Stream::open( "", false );
    MagAOX::app::loopTiming::Stream::close();
    MagAOX::app::loopTiming::Stream::isOpen();
    MagAOX::app::loopTiming::Stream::checkIdentity();
    MagAOX::app::loopTiming::Stream::baseline();
    MagAOX::app::loopTiming::Stream::poll( {} );
    #endif
    // clang-format on
    Camera           camera;
    Stream::Snapshot value;
    REQUIRE( camera.m_reader.open( camera.m_name ) == 0 );
    REQUIRE( camera.m_reader.isOpen() );
    REQUIRE( camera.m_reader.checkIdentity() );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::unchanged );
    REQUIRE( camera.m_image.md->sem == 0 );

    camera.publish( 11, 10000500000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::updated );
    REQUIRE( value.counter == 11 );
    REQUIRE( value.atimeNs == 10000500000 );
    REQUIRE( value.writetimeNs == value.atimeNs + 100 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::unchanged );

    camera.publish( 15, 10002500000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::gap );
    REQUIRE( value.missed == 3 );
    camera.publish( 1, 10003000000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::reset );
    camera.publish( 2, 10003500000 );
    camera.m_reader.baseline();
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::unchanged );
    camera.publish( 3, 10004000000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::updated );

    camera.m_reader.close();
    REQUIRE_FALSE( camera.m_reader.isOpen() );
    REQUIRE_FALSE( camera.m_reader.checkIdentity() );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::invalid );
    REQUIRE( camera.m_image.md->cnt0 == 3 );
}

/** \brief Reject active writers, malformed timestamps and changed metadata without a new counter.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing stream rejects unusable timestamps", "[loopTimingMonitor][stream]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Stream::baseline();
    MagAOX::app::loopTiming::Stream::poll( {} );
    #endif
    // clang-format on
    Camera           camera;
    Stream::Snapshot value;
    REQUIRE( camera.m_reader.open( camera.m_name ) == 0 );
    camera.m_image.md->write = 1;
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::busy );
    camera.m_image.md->write         = 0;
    camera.m_image.md->atime.tv_nsec = 1000000000;
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::invalid );

    camera.publish( 10, 10000500000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::invalid );
    camera.m_reader.baseline();
    camera.publish( 11, 10000000000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::invalid );

    camera.publish( 0, 0 );
    camera.m_reader.baseline();
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::busy );
    camera.publish( 1, 11000000000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::unchanged );
    camera.publish( 2, 11000500000 );
    REQUIRE( camera.m_reader.poll( value ) == Stream::Result::updated );
}

/** \brief Enforce the SDK clock schema and detect a same-name publisher restart by inode.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing stream recognizes dedicated SDK publishers", "[loopTimingMonitor][stream]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Stream::open( "", true );
    MagAOX::app::loopTiming::Stream::checkIdentity();
    MagAOX::app::loopTiming::Stream::poll( {} );
    #endif
    // clang-format on
    Camera camera;
    REQUIRE( camera.m_reader.open( camera.m_name, true ) < 0 );
    REQUIRE_FALSE( camera.m_reader.isOpen() );

    const auto       name = uniqueName();
    Publisher        publisher;
    Stream           reader;
    Stream::Snapshot value;
    REQUIRE( reader.open( name, true ) < 0 );
    REQUIRE( publisher.open( name ) == 0 );
    REQUIRE( publisher.publish( 1000 ) );
    REQUIRE( reader.open( name, true ) == 0 );
    REQUIRE( reader.poll( value ) == Stream::Result::unchanged );
    const auto inode = value.inode;
    REQUIRE( publisher.publish( 2000 ) );
    REQUIRE( reader.poll( value ) == Stream::Result::updated );
    REQUIRE( value.atimeNs == 2000 );
    REQUIRE( value.writetimeNs >= value.atimeNs );

    publisher.close();
    REQUIRE( publisher.open( name ) == 0 );
    REQUIRE_FALSE( reader.checkIdentity() );
    REQUIRE( publisher.publish( 3000 ) );
    REQUIRE( reader.open( name, true ) == 0 );
    REQUIRE( reader.poll( value ) == Stream::Result::unchanged );
    REQUIRE( value.inode != inode );
    REQUIRE( reader.checkIdentity() );

    REQUIRE( reader.open( "../wrong" ) < 0 );
    REQUIRE( reader.open( "" ) < 0 );
}

/** \brief Failed library attachment cannot leave an invalid mapping for cleanup.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing stream tolerates incompatible image versions", "[loopTimingMonitor][stream]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Stream::open( "", false );
    MagAOX::app::loopTiming::Stream::close();
    #endif
    // clang-format on
    Camera     camera;
    const auto first              = camera.m_image.md->version[0];
    camera.m_image.md->version[0] = '!';
    REQUIRE( camera.m_reader.open( camera.m_name ) < 0 );
    REQUIRE_FALSE( camera.m_reader.isOpen() );
    camera.m_reader.close();
    camera.m_image.md->version[0] = first;
    REQUIRE( camera.m_reader.open( camera.m_name ) == 0 );
}

/** \brief Concurrent metadata publication never yields a torn accepted timestamp pair.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing stream qualifies concurrent metadata", "[loopTimingMonitor][stream]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Stream::poll( {} );
    #endif
    // clang-format on
    Camera camera;
    camera.publish( 1, 10000010000 );
    REQUIRE( camera.m_reader.open( camera.m_name ) == 0 );
    std::atomic<bool> done{ false };
    std::thread       writer(
        [&]()
        {
            for( uint64_t counter = 2; counter <= 10000; ++counter )
            {
                camera.publish( counter, 10000000000 + int64_t( counter ) * 10000 );
            }

            done.store( true, std::memory_order_release );
        } );

    bool coherent = true;

    do
    {
        Stream::Snapshot value;
        const auto       result = camera.m_reader.poll( value );

        if( result == Stream::Result::updated || result == Stream::Result::gap )
        {
            coherent = coherent && value.atimeNs == 10000000000 + int64_t( value.counter ) * 10000 &&
                       value.writetimeNs == value.atimeNs + 100;
        }
    } while( !done.load( std::memory_order_acquire ) );

    writer.join();
    REQUIRE( coherent );
    Stream::Snapshot last;
    const auto       result = camera.m_reader.poll( last );
    REQUIRE(
        ( result == Stream::Result::updated || result == Stream::Result::gap || result == Stream::Result::unchanged ) );
    REQUIRE( last.counter == 10000 );
    REQUIRE( last.atimeNs == 10100000000 );
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
