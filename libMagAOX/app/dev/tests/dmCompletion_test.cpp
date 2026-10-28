/** \file dmCompletion_test.cpp
 * \brief Offline identity, lifecycle and semaphore tests of DM reader completion.
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmCompletion.hpp"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <stdexcept>

namespace libXWCTest
{
namespace dmCompletionTest
{

using Publisher = MagAOX::app::dev::dmCompletion;
using namespace MagAOX::dmCompletion;
using MagAOX::dmTiming::Outcome;

/// \cond DM_COMPLETION_TEST_HELPERS
namespace
{

/// Own one process-private directory for ImageStreamIO's cached directory lookup.
struct Directory
{
    std::string m_path; ///< Workstation-local directory, never an instrument stream directory.

    std::string m_previous; ///< Original shared-memory directory environment.

    bool m_hadPrevious{ false }; ///< Restore an original value rather than introducing one.

    /// Install the temporary directory before the first image operation.
    Directory();

    /// Remove test-owned files and restore the original environment.
    ~Directory();
};

Directory::Directory()
{
    char       pattern[] = "/tmp/dm-completion-test-XXXXXX";
    const auto path      = mkdtemp( pattern );

    if( path == nullptr )
    {
        throw std::runtime_error( "cannot create private DM completion test directory" );
    }

    m_path              = path;
    const auto previous = getenv( "MILK_SHM_DIR" );
    m_hadPrevious       = previous != nullptr;

    if( previous != nullptr )
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

/// Return the directory retained until all fixtures and stream mappings are destroyed.
Directory &directory();

Directory &directory()
{
    static Directory value;
    return value;
}

/// Own a source image and independently opened completion reader without worker threads.
struct Fixture
{
    IMAGE m_source{}; ///< Small private input whose metadata identifies genuine commands.

    IMAGE m_reader{}; ///< Independent stream attachment used for coherent reads and real semaphore waits.

    std::string m_name; ///< Unique completion stream name within the private directory.

    Publisher m_publisher; ///< Sole completion writer under test.

    /// Create a source and optionally open the completion writer and reader.
    explicit Fixture( bool opened = true /**< [in] Create the completion stream immediately. */ );

    /// Invalidate completion before closing the reader and destroying the source.
    ~Fixture();

    /// Open an independent observer and discard startup notifications.
    void attach();

    /// Advance the source identity exactly once without posting its command semaphore.
    void next();

    /// Consume every pending completion notification and return their count.
    unsigned drain();

    /// Read the current completion record through the observer's independent mapping.
    Record read();
};

Fixture::Fixture( bool opened )
{
    static_cast<void>( directory() );
    static unsigned sequence = 0;
    m_name                   = "done" + std::to_string( getpid() ) + "x" + std::to_string( ++sequence );
    uint32_t size[]{ 2, 2 };
    REQUIRE(
        ImageStreamIO_createIm_gpu(
            &m_source, ( m_name + "input" ).c_str(), 2, size, _DATATYPE_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) ==
        0 );
    std::fill_n( m_source.array.F, 4, 0.25F );
    m_source.md->write     = 0;
    m_source.md->cnt0      = 100;
    m_source.md->writetime = { 1800000000, 123456789 };

    if( opened )
    {
        REQUIRE( m_publisher.open( m_name ) == 0 );
        attach();
    }
}

Fixture::~Fixture()
{
    m_publisher.close();

    if( m_reader.md != nullptr )
    {
        ImageStreamIO_closeIm( &m_reader );
    }

    ImageStreamIO_destroyIm( &m_source );
}

void Fixture::attach()
{
    REQUIRE( ImageStreamIO_openIm( &m_reader, m_name.c_str() ) == 0 );
    REQUIRE( m_reader.md->sem > 0 );
    static_cast<void>( drain() );
}

void Fixture::next()
{
    ++m_source.md->cnt0;
    m_source.md->writetime.tv_nsec += 500000;
}

unsigned Fixture::drain()
{
    unsigned count = 0;

    while( sem_trywait( m_reader.semptr[0] ) == 0 )
    {
        ++count;
    }

    REQUIRE( errno == EAGAIN );
    return count;
}

Record Fixture::read()
{
    Record result;
    REQUIRE( Publisher::read( &m_reader, result ) );
    return result;
}

/// Construct a qualified record with intentionally independent raw and monotonic clocks.
Record completeRecord();

Record completeRecord()
{
    Record result;
    result.session         = 7;
    result.epoch           = 2;
    result.sequence        = 11;
    result.state           = State::complete;
    result.inputInode      = 1234;
    result.inputCounter    = 456;
    result.publicationSec  = 1800000000;
    result.publicationNsec = 123456789;
    result.startedNs       = 1000000;
    result.finishedNs      = 1125000;
    result.outcome         = Outcome::success;
    result.publisherPid    = getpid();
    return result;
}

} // namespace
/// \endcond

/** \defgroup dmCompletion_tests DM reader-completion tests
 * \ingroup app_dev_unit_tests
 */

/// Reject unrecognized transport formats without replacing the last coherent observation.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion transport preserves command identity and rejects malformed records", "[dev::dmCompletion]" )
{
    const auto original = completeRecord();
    const auto wire     = encode( original );
    Record     decoded;
    REQUIRE( decode( wire, decoded ) );
    CHECK( qualified( decoded ) );
    CHECK( encode( decoded ) == wire );
    CHECK( decoded.publicationSec == 1800000000 );
    CHECK( decoded.finishedNs - decoded.startedNs == 125000 );
    CHECK( std::string( stateName( decoded.state ) ) == "complete" );

    for( const size_t index : { size_t( 0 ), size_t( 1 ), size_t( 5 ), size_t( 12 ), size_t( 15 ) } )
    {
        auto damaged        = wire;
        damaged[index]      = UINT64_MAX;
        const auto previous = encode( decoded );
        CHECK_FALSE( decode( damaged, decoded ) );
        CHECK( encode( decoded ) == previous );
    }

    auto failure           = original;
    failure.state          = State::error;
    failure.outcome        = Outcome::failed;
    failure.callbackResult = -17;
    REQUIRE( decode( encode( failure ), decoded ) );
    CHECK( decoded.callbackResult == -17 );
    CHECK_FALSE( qualified( decoded ) );

    auto invalid            = original;
    invalid.publicationNsec = 1000000000;
    CHECK_FALSE( qualified( invalid ) );
    invalid            = original;
    invalid.finishedNs = invalid.startedNs - 1;
    CHECK_FALSE( qualified( invalid ) );
    invalid       = original;
    invalid.epoch = 0;
    CHECK_FALSE( qualified( invalid ) );
    invalid       = original;
    invalid.state = State::busy;
    CHECK_FALSE( qualified( invalid ) );
}

/// Publish busy without waking observers, then notify only after the genuine reader finishes.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion semaphore follows the coherent reader-finished record", "[dev::dmCompletion]" )
{
    // clang-format off
#ifdef DM_COMPLETION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmCompletion::begin( nullptr, true );
    MagAOX::app::dev::dmCompletion::finish( nullptr, 0, Outcome::success );
    MagAOX::app::dev::dmCompletion::read( nullptr, {} );
#endif
    // clang-format on

    Fixture fixture;
    CHECK( fixture.read().state == State::unavailable );
    REQUIRE( fixture.m_publisher.enabled() );
    fixture.m_publisher.begin( &fixture.m_source );
    auto record = fixture.read();
    CHECK( record.state == State::busy );
    CHECK( record.finishedNs == 0 );
    CHECK_FALSE( qualified( record ) );
    CHECK( fixture.drain() == 0 );

    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    REQUIRE( sem_trywait( fixture.m_reader.semptr[0] ) == 0 );
    record = fixture.read();
    CHECK( qualified( record ) );
    CHECK( record.inputInode == fixture.m_source.md->inode );
    CHECK( record.inputCounter == fixture.m_source.md->cnt0 );
    CHECK( record.publicationSec == fixture.m_source.md->writetime.tv_sec );
    CHECK( record.publicationNsec == fixture.m_source.md->writetime.tv_nsec );
    CHECK( record.finishedNs >= record.startedNs );
    CHECK( encode( record ) == encode( fixture.m_publisher.snapshot() ) );
    CHECK( fixture.drain() == 0 );
    CHECK( fixture.m_source.md->cnt0 == 100 );
    CHECK( fixture.m_source.md->write == 0 );
    CHECK( fixture.m_source.array.F[0] == 0.25F );
    int sourcePosts = -1;
    REQUIRE( sem_getvalue( fixture.m_source.semptr[0], &sourcePosts ) == 0 );
    CHECK( sourcePosts == 0 );
}

/// Preserve failure and qualification distinctions even when the callback itself returns success.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion requires a successful qualified backend outcome", "[dev::dmCompletion]" )
{
    Fixture fixture;

    for( const auto outcome :
         { Outcome::unknown, Outcome::failed, Outcome::partial, Outcome::noSend, Outcome::rejected } )
    {
        fixture.m_publisher.begin( &fixture.m_source );
        fixture.m_publisher.finish( &fixture.m_source, 0, outcome );
        const auto record = fixture.read();
        CHECK_FALSE( qualified( record ) );
        CHECK( record.state != State::complete );
        CHECK( record.outcome == outcome );
        CHECK( record.callbackResult == 0 );
        CHECK( fixture.drain() == 1 );
        fixture.next();
    }

    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, -5, Outcome::success );
    const auto record = fixture.read();
    CHECK_FALSE( qualified( record ) );
    CHECK( record.state == State::error );
    CHECK( record.outcome == Outcome::success );
    CHECK( record.callbackResult == -5 );
}

/// Invalidate synthetic, duplicate and unstable attempts instead of retaining an earlier success.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion does not qualify synthetic duplicate or changing command identities", "[dev::dmCompletion]" )
{
    Fixture fixture;
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto first = fixture.read();
    REQUIRE( qualified( first ) );
    static_cast<void>( fixture.drain() );

    SECTION( "Synthetic callbacks have no genuine semaphore receipt" )
    {
        fixture.next();
        fixture.m_publisher.begin( &fixture.m_source, false );
    }

    SECTION( "A duplicate notification has no new command identity" )
    {
        fixture.m_publisher.begin( &fixture.m_source );
    }

    SECTION( "A source being written at entry is not coherent" )
    {
        fixture.next();
        fixture.m_source.md->write = 1;
        fixture.m_publisher.begin( &fixture.m_source );
        fixture.m_source.md->write = 0;
    }

    SECTION( "An unpublished source has no qualified command counter" )
    {
        fixture.m_source.md->cnt0 = 0;
        fixture.m_publisher.begin( &fixture.m_source );
    }

    SECTION( "An invalid publication timestamp cannot identify a command" )
    {
        fixture.next();
        fixture.m_source.md->writetime.tv_nsec = 1000000000;
        fixture.m_publisher.begin( &fixture.m_source );
    }

    SECTION( "The source counter changes during the callback" )
    {
        fixture.next();
        fixture.m_publisher.begin( &fixture.m_source );
        ++fixture.m_source.md->cnt0;
    }

    SECTION( "The source publication stamp changes during the callback" )
    {
        fixture.next();
        fixture.m_publisher.begin( &fixture.m_source );
        ++fixture.m_source.md->writetime.tv_nsec;
    }

    SECTION( "The source remains write-active at completion" )
    {
        fixture.next();
        fixture.m_publisher.begin( &fixture.m_source );
        fixture.m_source.md->write = 1;
    }

    SECTION( "The source mapping is absent at completion" )
    {
        fixture.next();
        fixture.m_publisher.begin( &fixture.m_source );
        fixture.m_publisher.finish( nullptr, 0, Outcome::success );
        const auto record = fixture.read();
        CHECK_FALSE( qualified( record ) );
        CHECK( record.state == State::unqualified );
        CHECK( record.sequence > first.sequence );
        return;
    }

    SECTION( "A different source mapping is supplied at completion" )
    {
        Fixture replacement( false );
        fixture.next();
        fixture.m_publisher.begin( &fixture.m_source );
        fixture.m_publisher.finish( &replacement.m_source, 0, Outcome::success );
        const auto record = fixture.read();
        CHECK_FALSE( qualified( record ) );
        CHECK( record.state == State::unqualified );
        CHECK( record.sequence > first.sequence );
        CHECK( fixture.drain() == 1 );
        return;
    }

    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto record = fixture.read();
    CHECK_FALSE( qualified( record ) );
    CHECK( record.state == State::unqualified );
    CHECK( record.sequence > first.sequence );
    CHECK( fixture.drain() == 1 );
}

/// Distinguish counter resets and source replacement from commands in the previous attachment epoch.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion changes epoch when source identity restarts", "[dev::dmCompletion]" )
{
    Fixture fixture;
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto first = fixture.read();
    REQUIRE( qualified( first ) );

    fixture.m_source.md->cnt0 = 1;
    ++fixture.m_source.md->writetime.tv_sec;
    fixture.m_publisher.begin( &fixture.m_source );
    CHECK( fixture.read().state == State::busy );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto reset = fixture.read();
    CHECK( qualified( reset ) );
    CHECK( reset.epoch > first.epoch );
    CHECK( reset.inputCounter == 1 );
    CHECK( reset.session == first.session );

    Fixture replacement( false );
    fixture.m_publisher.begin( &replacement.m_source );
    fixture.m_publisher.finish( &replacement.m_source, 0, Outcome::success );
    const auto replaced = fixture.read();
    CHECK( qualified( replaced ) );
    CHECK( replaced.epoch > reset.epoch );
    CHECK( replaced.inputInode == replacement.m_source.md->inode );
    CHECK( replaced.inputInode != first.inputInode );
}

/// Keep a synthetic startup callback from consuming the first real command's progression baseline.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion qualifies the first genuine receipt after synthetic startup", "[dev::dmCompletion]" )
{
    Fixture fixture;
    fixture.m_publisher.begin( &fixture.m_source, false );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    CHECK_FALSE( qualified( fixture.read() ) );
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto record = fixture.read();
    CHECK( qualified( record ) );
    CHECK( record.inputCounter == 100 );
    CHECK( record.sequence == 2 );
}

/// Treat notifications as wakeups while the latest coherent sequence remains authoritative.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion reader observes the latest record after queued notifications", "[dev::dmCompletion]" )
{
    Fixture fixture;

    for( unsigned j = 0; j < 4; ++j )
    {
        fixture.m_publisher.begin( &fixture.m_source );
        fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
        fixture.next();
    }

    REQUIRE( sem_trywait( fixture.m_reader.semptr[0] ) == 0 );
    const auto latest = fixture.read();
    CHECK( qualified( latest ) );
    CHECK( latest.sequence == 4 );
    CHECK( latest.inputCounter == 103 );
    CHECK( fixture.drain() == 3 );

    fixture.m_publisher.begin( &fixture.m_source );
    CHECK( fixture.read().state == State::busy );
    CHECK_FALSE( qualified( fixture.read() ) );
    CHECK( fixture.drain() == 0 );
}

/// Publish invalidation on reset and shutdown and give a restarted writer a new session.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion reset shutdown and restart revoke old successful records", "[dev::dmCompletion]" )
{
    Fixture fixture;
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto first = fixture.read();
    REQUIRE( qualified( first ) );
    static_cast<void>( fixture.drain() );

    fixture.m_publisher.reset();
    auto record = fixture.read();
    CHECK( record.state == State::unavailable );
    CHECK_FALSE( qualified( record ) );
    CHECK( record.epoch > first.epoch );
    CHECK( fixture.drain() == 1 );

    fixture.next();
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    REQUIRE( qualified( fixture.read() ) );
    static_cast<void>( fixture.drain() );
    fixture.m_publisher.close();
    record = fixture.read();
    CHECK( record.state == State::unavailable );
    CHECK_FALSE( qualified( record ) );
    CHECK_FALSE( fixture.m_publisher.enabled() );
    CHECK( fixture.drain() == 1 );

    SECTION( "Ordinary restart reuses the existing completion mapping" )
    {
        CHECK( fixture.m_reader.md->write == 0 );
    }

    SECTION( "An interrupted writer can leave a recognized record write-active" )
    {
        fixture.m_reader.md->write = 1;
    }

    Publisher restarted;
    REQUIRE( restarted.open( fixture.m_name ) == 0 );
    CHECK( fixture.m_reader.md->write == 0 );
    record = fixture.read();
    CHECK( record.state == State::unavailable );
    CHECK( record.session != first.session );
    restarted.begin( &fixture.m_source );
    restarted.finish( &fixture.m_source, 0, Outcome::success );
    CHECK( qualified( fixture.read() ) );
    CHECK( fixture.read().session == record.session );
}

/// Refuse simultaneous writers and unrelated stream layouts without modifying their contents.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion owns its output exclusively and preserves unrelated images", "[dev::dmCompletion]" )
{
    Fixture fixture;
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    const auto original = encode( fixture.read() );
    const auto counter  = fixture.m_reader.md->cnt0;
    Publisher  competitor;
    CHECK( competitor.open( fixture.m_name ) != 0 );
    CHECK_FALSE( competitor.enabled() );
    CHECK( encode( fixture.read() ) == original );
    CHECK( fixture.m_reader.md->cnt0 == counter );

    CHECK( competitor.open( fixture.m_name + "input" ) != 0 );
    CHECK_FALSE( competitor.enabled() );
    CHECK( fixture.m_source.md->datatype == _DATATYPE_FLOAT );
    CHECK( fixture.m_source.md->cnt0 == 100 );
    CHECK( fixture.m_source.array.F[0] == 0.25F );

    IMAGE      unrelated{};
    const auto unrelatedName = fixture.m_name + "foreign";
    uint32_t   size          = words;
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &unrelated, unrelatedName.c_str(), 1, &size, _DATATYPE_UINT64, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) ==
             0 );
    std::fill_n( unrelated.array.UI64, words, uint64_t( 12345 ) );
    unrelated.md->cnt0 = 71;
    CHECK( competitor.open( unrelatedName ) != 0 );
    CHECK_FALSE( competitor.enabled() );
    CHECK( unrelated.md->cnt0 == 71 );
    CHECK( unrelated.array.UI64[0] == 12345 );
    CHECK( unrelated.array.UI64[words - 1] == 12345 );
    ImageStreamIO_destroyIm( &unrelated );
}

/// Reject a torn or malformed completion record instead of returning the previous success.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion reader requires coherent versioned stream contents", "[dev::dmCompletion]" )
{
    Fixture fixture;
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    Record     observed = fixture.read();
    const auto previous = encode( observed );

    fixture.m_reader.md->write = 1;
    CHECK_FALSE( Publisher::read( &fixture.m_reader, observed ) );
    CHECK( encode( observed ) == previous );
    fixture.m_reader.md->write     = 0;
    fixture.m_reader.array.UI64[0] = 0;
    CHECK_FALSE( Publisher::read( &fixture.m_reader, observed ) );
    CHECK( encode( observed ) == previous );
    fixture.m_reader.array.UI64[0] = magic;
    CHECK_FALSE( Publisher::read( nullptr, observed ) );
    CHECK_FALSE( Publisher::read( &fixture.m_source, observed ) );
    CHECK( encode( observed ) == previous );
}

/// Keep disabled publication inert and preserve the caller's errno across the callback hot path.
/** \ingroup dmCompletion_tests */
TEST_CASE( "DM completion disabled and hot path operations preserve caller state", "[dev::dmCompletion]" )
{
    Fixture fixture( false );
    CHECK_FALSE( fixture.m_publisher.enabled() );
    REQUIRE( fixture.m_publisher.open( "" ) == 0 );
    CHECK_FALSE( fixture.m_publisher.enabled() );
    errno = ENOTTY;
    fixture.m_publisher.begin( &fixture.m_source );
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    CHECK( errno == ENOTTY );
    CHECK_FALSE( qualified( fixture.m_publisher.snapshot() ) );
    CHECK( fixture.m_source.md->cnt0 == 100 );
    CHECK( fixture.m_source.md->write == 0 );
    CHECK( fixture.m_source.array.F[3] == 0.25F );

    REQUIRE( fixture.m_publisher.open( fixture.m_name ) == 0 );
    fixture.attach();
    errno = ENOTTY;
    fixture.m_publisher.begin( &fixture.m_source );
    CHECK( errno == ENOTTY );
    errno = EDOM;
    fixture.m_publisher.finish( &fixture.m_source, 0, Outcome::success );
    CHECK( errno == EDOM );
    CHECK( qualified( fixture.read() ) );
}

} // namespace dmCompletionTest
} // namespace libXWCTest
