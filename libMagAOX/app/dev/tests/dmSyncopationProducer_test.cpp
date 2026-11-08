/** \file dmSyncopationProducer_test.cpp
 * \brief Private same-host transport, deadline and restart tests for syncopation producers.
 * \ingroup dmSyncopationProducer_unit_test
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmSyncopationProducer.hpp"

#include <filesystem>
#include <stdexcept>

namespace libXWCTest
{
namespace dmSyncopationProducerTest
{
using namespace MagAOX::app;

/// \cond DM_SYNCOPATION_PRODUCER_TEST_HARNESS
namespace
{

/// Supply only the CRTP shutdown flag and expose narrow protocol configuration seams.
struct Producer : dev::dmSyncopationProducer<Producer>
{
    bool m_shutdown{ false }; ///< Offline test shutdown; no app or hardware threads exist.

    using dev::dmSyncopationProducer<Producer>::m_syncopationOutstanding;
    using dev::dmSyncopationProducer<Producer>::m_syncopationPending;
    using dev::dmSyncopationProducer<Producer>::m_syncopationTiming;
    using dev::dmSyncopationProducer<Producer>::m_writeLeadUs;
};

/// Hold all ImageStreamIO files under a test-private workstation directory.
struct Directory
{
    std::string m_path; ///< Directory retained until every stream fixture has closed.

    /// Set the private image directory before ImageStreamIO caches it.
    Directory();

    /// Remove only this test's private directory.
    ~Directory();
};

Directory::Directory()
{
    char       pattern[] = "/tmp/synco-producer-test-XXXXXX";
    const auto path      = mkdtemp( pattern );

    if( !path || setenv( "MILK_SHM_DIR", path, 1 ) != 0 )
    {
        throw std::runtime_error( "cannot create private producer-test stream directory" );
    }

    m_path = path;
}

Directory::~Directory()
{
    std::filesystem::remove_all( m_path );
}

/// Get the process-private directory before making the first image.
Directory &directory();

Directory &directory()
{
    static Directory value;
    return value;
}

/// Model an independent scheduler and producer with actual small shared-memory images.
struct Fixture
{
    IMAGE m_source{}; ///< Four-pixel immutable-until-ack producer payload.

    IMAGE m_schedule{}; ///< Offline test scheduler's timing image.

    IMAGE m_ack{}; ///< Offline test scheduler's transport acknowledgement image.

    IMAGE m_ready{}; ///< Read-only independent observation of producer commit publications.

    Producer m_producer; ///< CRTP helper under test, used synchronously without threads.

    synco::Record m_record{}; ///< Last schedule advertised by the test scheduler.

    std::string m_name; ///< Unique local image prefix, never a configured instrument name.

    /// Create a source, schedule and consumer acknowledgement for one test.
    Fixture();

    /// Close producer attachments before destroying owned streams.
    ~Fixture();

    /// Advertise an available step with enough time for deterministic local tests.
    void advertise( int64_t epoch      = 1 /**< [in] Restart identity. */,
                    int64_t generation = 1 /**< [in] Schedule generation. */,
                    int     step       = 1 /**< [in] Selected auxiliary or passive slot. */ );

    /// Return the current producer commit from an independent attachment.
    synco::CommitRecord commit();

    /// Echo one exact transport disposition without touching source data.
    void acknowledge( const synco::CommitRecord &command /**< [in] Exact outstanding commit. */,
                      int                        status = synco::copied /**< [in] Snapshot or discard disposition. */ );
};

Fixture::Fixture()
{
    static_cast<void>( directory() );
    static unsigned sequence = 0;
    m_name                   = "producer" + std::to_string( getpid() ) + "x" + std::to_string( ++sequence );
    uint32_t size[]{ 2, 2 };
    REQUIRE(
        ImageStreamIO_createIm_gpu(
            &m_source, ( m_name + "_aux" ).c_str(), 2, size, _DATATYPE_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) ==
        0 );
    REQUIRE( synco::openRecord( m_schedule, m_name + "_timing", synco::words, true ) );
    REQUIRE( synco::openRecord( m_ack, m_name + "_aux_ack", synco::ackWords, true ) );
    std::fill_n( m_source.array.F, 4, 0.25F );
    m_source.md->write = 1;
    REQUIRE( ImageStreamIO_UpdateIm( &m_source ) == 0 );
    m_producer.m_syncopationTiming = m_name + "_timing";
    // Publish immediately; tests do not depend on real-time scheduling precision.
    m_producer.m_writeLeadUs = 100000;
    advertise();
}

Fixture::~Fixture()
{
    m_producer.syncopationClose();

    if( m_ready.md )
    {
        ImageStreamIO_closeIm( &m_ready );
    }

    for( auto *image : { &m_source, &m_schedule, &m_ack } )
    {
        if( image->md )
        {
            ImageStreamIO_destroyIm( image );
        }
    }
}

void Fixture::advertise( int64_t epoch, int64_t generation, int step )
{
    const int64_t now               = synco::monotonicNs();
    m_record                        = {};
    m_record[synco::version]        = synco::productionVersion;
    m_record[synco::epoch]          = epoch;
    m_record[synco::generation]     = generation;
    m_record[synco::state]          = synco::armed;
    m_record[synco::steps]          = 4;
    m_record[synco::period]         = 100000000;
    m_record[synco::reference]      = now;
    m_record[synco::expires]        = now + 200000000;
    m_record[synco::cycle]          = 100;
    m_record[synco::cutoffs + step] = now + 50000000;
    m_record[synco::targets + step] = now + 60000000;
    REQUIRE( synco::writeRecord( m_schedule, m_record ) );
}

synco::CommitRecord Fixture::commit()
{
    if( !m_ready.md )
    {
        REQUIRE( synco::openRecord( m_ready, m_name + "_aux_ready", synco::commitWords, false ) );
    }

    synco::CommitRecord command{};
    REQUIRE( synco::readRecord( m_ready, command ) );
    return command;
}

void Fixture::acknowledge( const synco::CommitRecord &command, int status )
{
    synco::AckRecord ack{};
    std::copy( command.begin(), command.end(), ack.begin() );
    ack[synco::ackStatus] = status;
    REQUIRE( synco::writeRecord( m_ack, ack ) );
}

} // namespace
/// \endcond

/// Confirm new consumers reject legacy layout while its original validator remains compatible.
TEST_CASE( "syncopation schedule versions and exact target identities", "[dmSyncopationProducer]" )
{
    Fixture    fixture;
    const auto now = synco::monotonicNs();
    REQUIRE( synco::validProductionRecord( fixture.m_record, now ) );
    REQUIRE_FALSE( synco::validRecord( fixture.m_record, now ) );
    REQUIRE( synco::nextProductionStep( fixture.m_record, now ) == 1 );
    fixture.m_record[synco::version] = synco::layoutVersion;
    REQUIRE( synco::validRecord( fixture.m_record, now ) );
    REQUIRE_FALSE( synco::validProductionRecord( fixture.m_record, now ) );

    fixture.advertise( 2, 3, 0 );
    const auto slot = synco::request( fixture.m_record, 0 );
    REQUIRE( slot.cycle == 101 );
    REQUIRE( slot.step == 0 );
    REQUIRE( slot.epoch == 2 );
    REQUIRE( slot.generation == 3 );
}

/// An exact transport acknowledgement, including discard, releases payload ownership before physical DM completion.
TEST_CASE( "producer freezes each payload until exact snapshot acknowledgement", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPoll;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPublish;
    #endif
    // clang-format on
    Fixture        fixture;
    synco::Request slot;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    const float pixels[]{ 1, 2, 3, 4 };
    REQUIRE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
    const auto command = fixture.commit();
    REQUIRE( synco::validCommand( command, synco::monotonicNs() ) );
    REQUIRE( synco::commandMatches( command, slot ) );
    REQUIRE( command[synco::cmdPayloadCounter] == static_cast<int64_t>( fixture.m_source.md->cnt0 ) );
    REQUIRE( command[synco::cmdPayloadInode] == static_cast<int64_t>( fixture.m_source.md->inode ) );
    REQUIRE( std::equal( std::begin( pixels ), std::end( pixels ), fixture.m_source.array.F ) );

    fixture.advertise();
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 0 );
    auto wrong = command;
    ++wrong[synco::cmdSequence];
    fixture.acknowledge( wrong );
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 0 );

    SECTION( "private snapshot permits immediate buffer reuse" )
    {
        fixture.acknowledge( command );
    }
    SECTION( "expired command discard also permits buffer reuse" )
    {
        fixture.acknowledge( command, synco::discarded );
    }

    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    REQUIRE_FALSE( fixture.m_producer.m_syncopationOutstanding );
}

/// Generation resets discard pending ownership only through a new valid scheduler record.
TEST_CASE( "producer rejects obsolete plans and waits through invalid timing", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPoll;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPublish;
    #endif
    // clang-format on
    Fixture        fixture;
    synco::Request old;
    REQUIRE( fixture.m_producer.syncopationPoll( old, fixture.m_source ) == 1 );
    const float pixels[]{ 1, 1, 1, 1 };
    fixture.advertise( 1, 2 );
    REQUIRE_FALSE( fixture.m_producer.syncopationPublish( pixels, old, fixture.m_source ) );
    synco::Request current;
    REQUIRE( fixture.m_producer.syncopationPoll( current, fixture.m_source ) == 1 );
    REQUIRE( fixture.m_producer.syncopationPublish( pixels, current, fixture.m_source ) );

    fixture.m_record[synco::state] = synco::invalid;
    REQUIRE( synco::writeRecord( fixture.m_schedule, fixture.m_record ) );
    REQUIRE( fixture.m_producer.syncopationPoll( current, fixture.m_source ) == 0 );
    REQUIRE( fixture.m_producer.m_syncopationOutstanding );
    fixture.advertise( 2, 1 );
    REQUIRE( fixture.m_producer.syncopationPoll( current, fixture.m_source ) == 1 );
    REQUIRE_FALSE( fixture.m_producer.m_syncopationOutstanding );
}

/// A deadline or nonfinite-pixel failure leaves the payload uncommitted, and duplicate requests are not replayed.
TEST_CASE( "producer rejects late incomplete and duplicate work", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPublish;
    #endif
    // clang-format on
    Fixture        fixture;
    synco::Request slot;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    auto duplicate = slot;
    REQUIRE( fixture.m_producer.syncopationPoll( duplicate, fixture.m_source ) == 0 );
    const auto counter = fixture.m_source.md->cnt0;

    SECTION( "a cutoff must still be in the future" )
    {
        const auto wake = synco::toTimespec( slot.cutoffNs + 1 );
        REQUIRE( clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr ) == 0 );
        const float pixels[]{ 1, 1, 1, 1 };
        REQUIRE_FALSE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
    }
    SECTION( "a complete contribution must contain finite pixels" )
    {
        for( const auto invalid : { std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity() } )
        {
            const float pixels[]{ 1, invalid, 1, 1 };
            REQUIRE_FALSE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
        }
    }
    SECTION( "shutdown cancels publication" )
    {
        fixture.m_producer.m_shutdown = true;
        const float pixels[]{ 1, 1, 1, 1 };
        REQUIRE_FALSE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
    }

    REQUIRE( fixture.m_source.md->cnt0 == counter );
    REQUIRE_FALSE( fixture.m_producer.m_syncopationOutstanding );
}

/// Reject a competing source writer and refuse malformed protocol images without altering their contents.
TEST_CASE( "producer claims one source and validates transport geometry", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPoll;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationClose;
    #endif
    // clang-format on
    Fixture        fixture;
    synco::Request slot;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    Producer competing;
    competing.m_syncopationTiming = fixture.m_producer.m_syncopationTiming;
    REQUIRE( competing.syncopationPoll( slot, fixture.m_source ) == -1 );
    fixture.m_producer.syncopationClose();
    REQUIRE( competing.syncopationPoll( slot, fixture.m_source ) == 1 );
    competing.syncopationClose();

    fixture.m_schedule.md->write    = 1;
    fixture.m_schedule.md->datatype = _DATATYPE_FLOAT;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == -1 );
    fixture.m_schedule.md->datatype = _DATATYPE_INT64;
    fixture.m_schedule.md->write    = 0;
}

/// Schema validation independently rejects stale identities, future completion stamps and unsupported dispositions.
TEST_CASE( "producer command records require exact completed slot identity", "[dmSyncopationProducer]" )
{
    Fixture        fixture;
    synco::Request slot;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    const float pixels[]{ 0, 0, 0, 0 };
    REQUIRE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
    const auto command = fixture.commit();

    for( auto field : { synco::cmdEpoch,
                        synco::cmdGeneration,
                        synco::cmdCycle,
                        synco::cmdStep,
                        synco::cmdTarget,
                        synco::cmdCutoff } )
    {
        auto wrong = command;
        ++wrong[field];
        REQUIRE_FALSE( synco::commandMatches( wrong, slot ) );
    }

    auto late               = command;
    late[synco::cmdReadyNs] = late[synco::cmdCutoff] + 1;
    REQUIRE_FALSE( synco::validCommand( late, late[synco::cmdReadyNs] ) );
    REQUIRE_FALSE( synco::validCommand( command, command[synco::cmdReadyNs] - 1 ) );
    synco::AckRecord ack{};
    std::copy( command.begin(), command.end(), ack.begin() );
    ack[synco::ackStatus] = 999;
    REQUIRE_FALSE( synco::acknowledges( ack, command ) );
}

/// Nonfinite configuration is rejected even under the application's fast-math optimization flags.
TEST_CASE( "producer validates completion lead without fast-math assumptions", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::setupConfig;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::loadConfig;
    #endif
    // clang-format on
    Producer                 producer;
    mx::app::appConfigurator config;
    REQUIRE( producer.setupConfig( config ) == 0 );

    for( const auto invalid : { std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(),
                                0.0,
                                -1.0,
                                1000001.0 } )
    {
        producer.m_writeLeadUs = invalid;
        REQUIRE( producer.loadConfig( config ) < 0 );
    }

    producer.m_writeLeadUs = 50;
    REQUIRE( producer.loadConfig( config ) == 0 );
}

/// Metadata changes after a request cannot change the validated private-payload extent or source ownership.
TEST_CASE( "producer freezes source geometry before accessing private pixels", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPoll;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPublish;
    #endif
    // clang-format on
    Fixture        fixture;
    synco::Request slot;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    const auto  original = *fixture.m_source.md;
    auto *const data     = fixture.m_source.array.F;
    const float pixels[]{ 1, 2, 3, 4 };

    SECTION( "element count cannot enlarge the private read" )
    {
        fixture.m_source.md->nelement = 1048576;
    }
    SECTION( "mapped byte extent must contain exactly the validated image" )
    {
        fixture.m_source.md->imdatamemsize = 1;
    }
    SECTION( "shape cannot change even with the same pixel count" )
    {
        fixture.m_source.md->size[0] = 1;
        fixture.m_source.md->size[1] = 4;
    }
    SECTION( "a different source inode cannot reuse the original sidecars" )
    {
        ++fixture.m_source.md->inode;
    }
    SECTION( "source datatype remains float32" )
    {
        fixture.m_source.md->datatype = _DATATYPE_DOUBLE;
    }
    SECTION( "an unavailable mapped pixel pointer cannot be accessed" )
    {
        fixture.m_source.array.F = nullptr;
    }

    REQUIRE_FALSE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
    REQUIRE( fixture.m_source.md->cnt0 == original.cnt0 );
    REQUIRE_FALSE( fixture.m_producer.m_syncopationOutstanding );
    *fixture.m_source.md     = original;
    fixture.m_source.array.F = data;
    REQUIRE( fixture.m_producer.syncopationPublish( pixels, slot, fixture.m_source ) );
}

/// Fixed record utilities check both logical shape and mapped-byte extent before copying sidecar data.
TEST_CASE( "producer protocol record utilities reject inconsistent extents", "[dmSyncopationProducer]" )
{
    Fixture       fixture;
    const auto    original = *fixture.m_schedule.md;
    synco::Record value{};

    SECTION( "a record element count cannot disagree with its one-dimensional shape" )
    {
        --fixture.m_schedule.md->nelement;
    }
    SECTION( "a truncated record allocation is rejected" )
    {
        --fixture.m_schedule.md->imdatamemsize;
    }

    REQUIRE_FALSE( synco::readRecord( fixture.m_schedule, value ) );
    REQUIRE_FALSE( synco::writeRecord( fixture.m_schedule, fixture.m_record ) );
    IMAGE reader{};
    REQUIRE_FALSE( synco::openRecord( reader, fixture.m_name + "_timing", synco::words, false ) );
    REQUIRE( reader.md == nullptr );
    *fixture.m_schedule.md = original;
    REQUIRE( synco::readRecord( fixture.m_schedule, value ) );
    REQUIRE( value == fixture.m_record );
}

/// Replacing a producer instance does not revoke ownership of its predecessor's committed payload.
TEST_CASE( "producer restart adopts outstanding transport ownership", "[dmSyncopationProducer]" )
{
    // clang-format off
    #ifdef DMSYNCOPATIONPRODUCER_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationOpen;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationPoll;
    MagAOX::app::dev::dmSyncopationProducer<Producer>::syncopationClose;
    #endif
    // clang-format on
    Fixture        fixture;
    synco::Request slot;
    REQUIRE( fixture.m_producer.syncopationPoll( slot, fixture.m_source ) == 1 );
    const float priorPixels[]{ 1, 2, 3, 4 };
    REQUIRE( fixture.m_producer.syncopationPublish( priorPixels, slot, fixture.m_source ) );
    const auto prior   = fixture.commit();
    const auto counter = fixture.m_source.md->cnt0;
    fixture.m_producer.syncopationClose();

    Producer replacement;
    replacement.m_syncopationTiming = fixture.m_producer.m_syncopationTiming;
    replacement.m_writeLeadUs       = 100000;
    fixture.advertise();
    REQUIRE( replacement.syncopationPoll( slot, fixture.m_source ) == 0 );
    REQUIRE( replacement.m_syncopationOutstanding );
    REQUIRE( replacement.m_syncopationPending == prior );
    REQUIRE( fixture.m_source.md->cnt0 == counter );
    REQUIRE( std::equal( std::begin( priorPixels ), std::end( priorPixels ), fixture.m_source.array.F ) );

    SECTION( "an exact old-session transport acknowledgement releases the buffer" )
    {
        fixture.acknowledge( prior );
    }
    SECTION( "a coherent new scheduler generation revokes the old command" )
    {
        fixture.advertise( 1, 2 );
    }

    REQUIRE( replacement.syncopationPoll( slot, fixture.m_source ) == 1 );
    REQUIRE_FALSE( replacement.m_syncopationOutstanding );
    const float nextPixels[]{ 5, 6, 7, 8 };
    REQUIRE( replacement.syncopationPublish( nextPixels, slot, fixture.m_source ) );
    const auto next = fixture.commit();
    REQUIRE( next[synco::cmdSession] != prior[synco::cmdSession] );
    REQUIRE( next[synco::cmdSequence] == 1 );
    REQUIRE( std::equal( std::begin( nextPixels ), std::end( nextPixels ), fixture.m_source.array.F ) );
    replacement.syncopationClose();
}

} // namespace dmSyncopationProducerTest
} // namespace libXWCTest
