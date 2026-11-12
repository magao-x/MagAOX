/** \file dmTimingEventStream_test.cpp
 * \brief Offline event protocol, bounded history, publication coherence and lifecycle tests.
 * \ingroup dmTimingEventStream_test
 */
#include "../../../../tests/testXWC.hpp"

#include <atomic>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include <ImageStreamIO/ImageStreamIO.h>

/** \defgroup dmTimingEventStream_test DM timing event transport tests
 * \ingroup app_unit_test
 */

/// \cond DM_TIMING_EVENT_TEST_HELPERS
static std::atomic<unsigned> eventClockCalls{ 0 }; ///< Counts only clock reads made by the header under test.

/// Count a transport clock read while preserving the real clock implementation.
static int eventClockGettime( clockid_t clock /**< [in] Selected local clock. */,
                              timespec *stamp /**< [out] Real clock value. */ )
{
    ++eventClockCalls;
    return clock_gettime( clock, stamp );
}
/// \endcond

// ImageStreamIO is already declared; notification calls in the transport fail to compile.
#define clock_gettime eventClockGettime
#define ImageStreamIO_UpdateIm timing_event_notifications_are_forbidden
#define ImageStreamIO_sempost timing_event_notifications_are_forbidden
#define ImageStreamIO_semwait timing_event_waits_are_forbidden
#include "../dmTimingEventStream.hpp"
#undef ImageStreamIO_semwait
#undef ImageStreamIO_sempost
#undef ImageStreamIO_UpdateIm
#undef clock_gettime

namespace libXWCTest
{
namespace dmTimingEventStreamTest
{
using Stream = MagAOX::app::dev::dmTimingEventStream;
using Result = Stream::Result;
using namespace MagAOX::dmTimingEvents;

/// \cond DM_TIMING_EVENT_TEST_HELPERS
namespace
{
/// Own a process-private ImageStreamIO directory before its first cached lookup.
struct Directory
{
    std::string m_path;                 ///< Temporary workstation directory containing only these fixtures.
    std::string m_previous;             ///< Previous MILK_SHM_DIR value, restored at teardown.
    bool        m_hadPrevious{ false }; ///< Whether teardown restores or removes the environment variable.

    /// Install the private stream directory.
    Directory();

    /// Remove private test files after every mapping has been closed.
    ~Directory();
};

Directory::Directory()
{
    char       pattern[] = "/tmp/dm-timing-events-XXXXXX";
    const auto path      = mkdtemp( pattern );

    if( !path )
    {
        throw std::runtime_error( "cannot create private timing-event directory" );
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

/// Return a new private name while retaining the shared test directory through process exit.
std::string name();

std::string name()
{
    static Directory directory;
    static unsigned  count = 0;
    return "dmEvents" + std::to_string( getpid() ) + "_" + std::to_string( ++count );
}

/// Construct fields whose independent values reveal torn or mismatched transport records.
Record sdk( uint64_t sequence /**< [in] Producer logical sequence. */ );

Record sdk( uint64_t sequence )
{
    Record record;
    record.kind     = Kind::sdk;
    record.key      = { 1234, sequence * 3, 2000000000, int64_t( sequence ) };
    record.sequence = sequence;
    record.startNs  = 1000000000 + int64_t( sequence );
    return record;
}

/// Independently attach metadata for deliberate isolated protocol fault injection.
struct Mapping
{
    IMAGE m_image{}; ///< Independent mapping closed without destroying shared resources.

    /// Open one already created private stream.
    explicit Mapping( const std::string &stream /**< [in] Existing private fixture name. */ );

    /// Close this independent mapping.
    ~Mapping();
};

Mapping::Mapping( const std::string &stream )
{
    REQUIRE( ImageStreamIO_openIm( &m_image, stream.c_str() ) == IMAGESTREAMIO_SUCCESS );
}

Mapping::~Mapping()
{
    ImageStreamIO_closeIm( &m_image );
}
} // namespace
/// \endcond

/** \brief Validate exact identity fields and preserve off-mode main classifications through wire encoding. */
TEST_CASE( "DM timing events preserve all endpoint and classification fields", "[dmTimingEvents]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::dmTimingEvents::valid;
    MagAOX::dmTimingEvents::equal;
    MagAOX::dmTimingEvents::encode;
    MagAOX::dmTimingEvents::decode;
#endif
    // clang-format on

    auto   record = sdk( 7 );
    Record decoded;
    REQUIRE( decode( encode( record ), decoded ) );
    CHECK( equal( decoded.key, record.key ) );
    CHECK( decoded.sequence == 7 );
    CHECK( decoded.startNs == record.startNs );
    CHECK_FALSE( valid( Key{} ) );
    CHECK_FALSE( valid( Key{ 1, 1, 2, 1000000000 } ) );
    CHECK_FALSE( valid( Key{ 1, 1, std::numeric_limits<int64_t>::max(), 0 } ) );
    CHECK_FALSE(
        equal( record.key, Key{ record.key.inode, record.key.counter, record.key.sec, record.key.nsec + 1 } ) );

    record.kind           = Kind::main;
    record.generation     = 19;
    record.primaryInode   = 5432;
    record.primaryCounter = 55;
    record.steps          = 4;
    record.startNs        = 0;
    record.enabled        = GENERATE( false, true );
    REQUIRE( decode( encode( record ), decoded ) );
    CHECK( decoded.kind == Kind::main );
    CHECK( decoded.generation == 19 );
    CHECK( decoded.primaryInode == 5432 );
    CHECK( decoded.primaryCounter == 55 );
    CHECK( decoded.steps == 4 );
    CHECK( decoded.enabled == record.enabled );

    record.kind = Kind::mode;
    record.key  = {};
    REQUIRE( decode( encode( record ), decoded ) );
    CHECK( decoded.kind == Kind::mode );
    CHECK( equal( decoded.key, Key{} ) );
}

/** \brief Reject unrecognized kinds, malformed flags and absent required fields without modifying output. */
TEST_CASE( "DM timing events reject malformed wire records", "[dmTimingEvents]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::dmTimingEvents::decode;
#endif
    // clang-format on

    const auto record  = sdk( 9 );
    auto       wire    = encode( record );
    Record     decoded = sdk( 5 );

    SECTION( "unknown kind" )
    {
        wire[0] = 4;
    }

    SECTION( "invalid boolean" )
    {
        wire[10] = 2;
    }

    SECTION( "missing sequence" )
    {
        wire[5] = 0;
    }

    SECTION( "missing SDK endpoint" )
    {
        wire[11] = 0;
    }

    SECTION( "main without source identity" )
    {
        wire[0]  = uint64_t( Kind::main );
        wire[11] = 0;
    }

    REQUIRE_FALSE( decode( wire, decoded ) );
    CHECK( decoded.sequence == 5 );
    CHECK( equal( decoded.key, sdk( 5 ).key ) );
}

/** \brief Disabled publication succeeds while invalid enabled names and capacity bounds fail safely. */
TEST_CASE( "DM timing event stream validates optional startup", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::open;
    MagAOX::app::dev::dmTimingEventStream::publish;
    MagAOX::app::dev::dmTimingEventStream::enabled;
    MagAOX::app::dev::dmTimingEventStream::Reader::open;
    MagAOX::app::dev::dmTimingEventStream::Reader::poll;
#endif
    // clang-format on

    Stream publisher;
    REQUIRE( publisher.open( "" ) == 0 );
    CHECK_FALSE( publisher.enabled() );
    CHECK( publisher.publish( Record{} ) );
    const auto stream = name();
    CHECK( publisher.open( stream, Stream::minCapacity - 1 ) < 0 );
    CHECK( publisher.open( stream, Stream::maxCapacity + 1 ) < 0 );
    CHECK( publisher.open( "../not-local", 4 ) < 0 );
    CHECK( publisher.open( ".", 4 ) < 0 );
    Stream::Reader reader;
    CHECK( reader.open( stream ) < 0 );
    Record record;
    CHECK( reader.poll( record ) == Result::invalid );
    CHECK( reader.inode() == 0 );
}

/** \brief Drain a whole burst in order without per-publication clock reads or semaphore notifications. */
TEST_CASE( "DM timing event stream retains ordered bursts without clock reads", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::publish;
    MagAOX::app::dev::dmTimingEventStream::compatible;
    MagAOX::app::dev::dmTimingEventStream::Reader::poll;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 16 ) == 0 );
    REQUIRE( reader.open( stream ) == 0 );
    Mapping mapping( stream );
    CHECK( Stream::compatible( &mapping.m_image ) );
    CHECK( mapping.m_image.md->sem == 0 );
    const auto calls = eventClockCalls.load();

    for( uint64_t index = 1; index <= 12; ++index )
    {
        REQUIRE( publisher.publish( sdk( index ) ) );
    }

    CHECK( eventClockCalls.load() == calls );
    Record record;

    for( uint64_t index = 1; index <= 12; ++index )
    {
        REQUIRE( reader.poll( record ) == Result::updated );
        CHECK( record.sequence == index );
        CHECK( equal( record.key, sdk( index ).key ) );
        CHECK( record.startNs == sdk( index ).startNs );
    }

    CHECK( reader.poll( record ) == Result::unchanged );
    CHECK( reader.missed() == 0 );
}

/** \brief Distinguish overwritten ring slots from producer-sequence gaps predating publication. */
TEST_CASE( "DM timing event stream counts ring overwrite without rewriting producer identity", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::publish;
    MagAOX::app::dev::dmTimingEventStream::Reader::poll;
    MagAOX::app::dev::dmTimingEventStream::Reader::missed;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 4 ) == 0 );
    REQUIRE( reader.open( stream ) == 0 );

    for( uint64_t index = 1; index <= 10; ++index )
    {
        REQUIRE( publisher.publish( sdk( index ) ) );
    }

    Record record;
    REQUIRE( reader.poll( record ) == Result::gap );
    CHECK( reader.missed() == 6 );
    CHECK( record.sequence == 7 );

    for( uint64_t index = 8; index <= 10; ++index )
    {
        REQUIRE( reader.poll( record ) == Result::updated );
        CHECK( reader.missed() == 0 );
        CHECK( record.sequence == index );
    }

    REQUIRE( publisher.publish( sdk( 14 ) ) );
    REQUIRE( reader.poll( record ) == Result::updated );
    CHECK( reader.missed() == 0 );
    CHECK( record.sequence == 14 );
    CHECK_FALSE( publisher.publish( sdk( 14 ) ) );
    CHECK_FALSE( publisher.publish( sdk( 13 ) ) );
    CHECK( reader.poll( record ) == Result::unchanged );
}

/** \brief Attachment and explicit baselining skip historical records rather than reporting them as new. */
TEST_CASE( "DM timing event reader baselines existing history", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::Reader::open;
    MagAOX::app::dev::dmTimingEventStream::Reader::baseline;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 4 ) == 0 );
    REQUIRE( publisher.publish( sdk( 1 ) ) );
    REQUIRE( publisher.publish( sdk( 2 ) ) );
    REQUIRE( reader.open( stream ) == 0 );
    Record record;
    REQUIRE( reader.poll( record ) == Result::unchanged );
    REQUIRE( publisher.publish( sdk( 3 ) ) );
    REQUIRE( reader.poll( record ) == Result::updated );
    CHECK( record.sequence == 3 );
    REQUIRE( publisher.publish( sdk( 4 ) ) );
    reader.baseline();
    CHECK( reader.poll( record ) == Result::unchanged );
}

/** \brief Restart creates a new incarnation while independent old mappings remain readable and unchanged. */
TEST_CASE( "DM timing event publisher restart replaces its backing inode", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::open;
    MagAOX::app::dev::dmTimingEventStream::close;
    MagAOX::app::dev::dmTimingEventStream::Reader::checkIdentity;
    MagAOX::app::dev::dmTimingEventStream::Reader::inode;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 4 ) == 0 );
    REQUIRE( reader.open( stream ) == 0 );
    REQUIRE( publisher.publish( sdk( 1 ) ) );
    Mapping    old( stream );
    const auto original = reader.inode();
    publisher.close();
    REQUIRE( publisher.open( stream, 8 ) == 0 );
    CHECK_FALSE( reader.checkIdentity() );
    Record record;
    REQUIRE( reader.poll( record ) == Result::updated );
    CHECK( record.sequence == 1 );
    CHECK( old.m_image.md->cnt0 == 1 );
    REQUIRE( reader.open( stream ) == 0 );
    CHECK( reader.inode() != original );
    CHECK( reader.checkIdentity() );
    REQUIRE( publisher.publish( sdk( 1 ) ) );
    REQUIRE( reader.poll( record ) == Result::updated );
    CHECK( record.sequence == 1 );
}

/** \brief Competing writers and incompatible images cannot replace an existing stream. */
TEST_CASE( "DM timing event stream refuses ownership collisions", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::open;
    MagAOX::app::dev::dmTimingEventStream::Reader::open;
#endif
    // clang-format on

    const auto stream = name();
    Stream     first;
    Stream     second;
    REQUIRE( first.open( stream, 4 ) == 0 );
    CHECK( second.open( stream, 4 ) < 0 );
    CHECK( first.enabled() );
    CHECK_FALSE( second.enabled() );
    REQUIRE( first.publish( sdk( 1 ) ) );

    const auto other = name();
    IMAGE      incompatible{};
    uint32_t   size[]{ 1, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu( &incompatible, other.c_str(), 2, size, _DATATYPE_UINT8, -1, 1, 0, 0, 0, 0 ) ==
             0 );
    const auto inode = incompatible.md->inode;
    CHECK( second.open( other, 4 ) < 0 );
    CHECK( incompatible.md->inode == inode );
    Stream::Reader reader;
    CHECK( reader.open( other ) < 0 );
    ImageStreamIO_closeIm( &incompatible );
}

/** \brief In-progress and malformed slots never leak partial records or advance the reader cursor. */
TEST_CASE( "DM timing event reader rejects torn busy and malformed slots", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::Reader::poll;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 4 ) == 0 );
    REQUIRE( reader.open( stream ) == 0 );
    REQUIRE( publisher.publish( sdk( 1 ) ) );
    Mapping mapping( stream );
    auto   *slot = mapping.m_image.array.UI64;
    __atomic_store_n( slot, uint64_t( 0 ), __ATOMIC_SEQ_CST );
    Record record = sdk( 99 );
    CHECK( reader.poll( record ) == Result::busy );
    CHECK( record.sequence == 99 );
    __atomic_store_n( slot, uint64_t( 1 ), __ATOMIC_RELEASE );
    __atomic_store_n( slot + 11, uint64_t( 2 ), __ATOMIC_RELEASE );
    CHECK( reader.poll( record ) == Result::invalid );
    CHECK( record.sequence == 99 );
    __atomic_store_n( slot + 11, uint64_t( 0 ), __ATOMIC_RELEASE );
    REQUIRE( reader.poll( record ) == Result::updated );
    CHECK( record.sequence == 1 );
}

/** \brief Counter or incarnation regressions explicitly reset the reader baseline. */
TEST_CASE( "DM timing event reader reports metadata resets", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::Reader::poll;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 4 ) == 0 );
    REQUIRE( reader.open( stream ) == 0 );
    REQUIRE( publisher.publish( sdk( 1 ) ) );
    Record record;
    REQUIRE( reader.poll( record ) == Result::updated );
    Mapping mapping( stream );

    SECTION( "counter regression" )
    {
        __atomic_store_n( &mapping.m_image.md->cnt0, uint64_t( 0 ), __ATOMIC_RELEASE );
    }

    SECTION( "incarnation change" )
    {
        __atomic_store_n( &mapping.m_image.md->inode, mapping.m_image.md->inode + 1, __ATOMIC_RELEASE );
    }

    CHECK( reader.poll( record ) == Result::reset );
    CHECK( reader.poll( record ) == Result::unchanged );
}

/** \brief A concurrent wrapping publisher yields only internally consistent records with exact overwrite accounting. */
TEST_CASE( "DM timing event slots remain coherent while overwritten concurrently", "[dmTimingEventStream]" )
{
    // clang-format off
#ifdef DM_TIMING_EVENT_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTimingEventStream::publish;
    MagAOX::app::dev::dmTimingEventStream::Reader::poll;
    MagAOX::app::dev::dmTimingEventStream::Reader::missed;
#endif
    // clang-format on

    const auto     stream = name();
    Stream         publisher;
    Stream::Reader reader;
    REQUIRE( publisher.open( stream, 32 ) == 0 );
    REQUIRE( reader.open( stream ) == 0 );
    constexpr uint64_t total = 20000;
    std::atomic<bool>  failed{ false };
    std::thread        writer(
        [&]()
        {
            for( uint64_t index = 1; index <= total; ++index )
            {
                if( !publisher.publish( sdk( index ) ) )
                {
                    failed = true;
                }
            }
        } );

    uint64_t observed = 0;
    uint64_t missed   = 0;
    uint64_t last     = 0;
    bool     coherent = true;

    for( unsigned attempt = 0; attempt < 1000000 && last < total && !failed.load(); ++attempt )
    {
        Record     record;
        const auto result = reader.poll( record );

        if( result == Result::updated || result == Result::gap )
        {
            coherent = coherent && record.sequence > last && equal( record.key, sdk( record.sequence ).key ) &&
                       record.startNs == sdk( record.sequence ).startNs;
            last = record.sequence;
            ++observed;
            missed += reader.missed();
        }
        else if( result != Result::unchanged && result != Result::busy )
        {
            coherent = false;
            break;
        }
        else
        {
            std::this_thread::yield();
        }
    }

    writer.join();
    CHECK_FALSE( failed.load() );
    CHECK( coherent );
    CHECK( last == total );
    CHECK( observed + missed == total );
}

} // namespace dmTimingEventStreamTest
} // namespace libXWCTest
