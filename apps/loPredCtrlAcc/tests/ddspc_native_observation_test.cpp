/** \file ddspc_native_observation_test.cpp
 * \brief Native observation accounting and actual reader/writer lifecycle tests using private API-double inputs.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../native_observer.hpp"
#include "adc_transport_fixture.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
namespace
{
/// Own only a fresh local observation directory; the test never creates a native shared-memory producer.
struct ObservationDirectory
{
    std::filesystem::path m_path; ///< Removed after all readers/writers have joined.

    /// Allocate a private ordinary-file directory.
    ObservationDirectory();

    /// Remove only this test's owned artifacts after worker destruction.
    ~ObservationDirectory();
};
ObservationDirectory::ObservationDirectory()
{
    auto  pattern = ( std::filesystem::temp_directory_path() / "ddspc-observation-XXXXXX" ).string();
    auto *created = mkdtemp( pattern.data() );
    if( !created )
        throw std::runtime_error( "Cannot create private observation directory" );
    m_path = created;
}
ObservationDirectory::~ObservationDirectory()
{
    std::filesystem::remove_all( m_path );
}

/// Bound asynchronous offline worker checks without opening any native namespace.
template <typename Predicate>
bool observed( Predicate predicate /**< [in] Thread-safe diagnostic condition. */ )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( std::chrono::steady_clock::now() < end )
    {
        if( predicate() )
            return true;
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    return false;
}
/// Establish explicit private names and keep every source independent of a live instrument name.
DDSPC::NativeObservationConfig
observation_config( const ObservationDirectory &directory /**< [in] Private output owner. */ )
{
    DDSPC::NativeObservationConfig c;
    c.m_directory         = directory.m_path;
    c.m_durationSec       = 2;
    c.m_streams[0].m_name = "obs-adc";
    c.m_streams[1].m_name = "obs-modal";
    c.m_streams[2].m_name = "obs-camera";
    c.m_streams[3].m_name = "obs-dm";
    return c;
}
/// Build one owned synthetic source envelope; timestamps remain test clock values, never native evidence.
DDSPC::NativeObservation observation_frame( DDSPC::ObservationSource source /**< [in] Source role. */,
                                            std::uint64_t            counter = 1 /**< [in] Publication identity. */ )
{
    DDSPC::NativeObservation r;
    r.m_source         = source;
    r.m_generation     = 1;
    r.m_counter        = counter;
    r.m_imageInode     = 30;
    r.m_acquiredRawNs  = 1000000000;
    r.m_publishedRawNs = 1000001000;
    r.m_receivedNs = r.m_availableNs = 2000000000 + std::int64_t( counter ) * 500000;
    r.m_clockNowNs                   = 1000010000;
    r.m_clockBracketNs               = 20;
    return r;
}
/// Encode the real shared software-completion protocol, with no physical application assertion.
MagAOX::dmCompletion::Wire completion_wire( MagAOX::dmCompletion::Kind kind /**< [in] Distinct endpoint contract. */ )
{
    MagAOX::dmCompletion::Record record;
    record.kind            = kind;
    record.session         = 11;
    record.epoch           = 12;
    record.sequence        = 1;
    record.state           = MagAOX::dmCompletion::State::complete;
    record.inputInode      = 30;
    record.inputCounter    = 1;
    record.publicationSec  = 1;
    record.publicationNsec = 1000;
    record.startedNs       = 100000;
    record.finishedNs      = 200000;
    record.outcome         = MagAOX::dmTiming::Outcome::success;
    return MagAOX::dmCompletion::encode( record );
}
/// Borrow stable process-private ImageStreamIO memory; mutate only before worker startup or after it stops.
struct ObservationImage
{
    IMAGE_METADATA               m_metadata;                     ///< Immutable source metadata during a threaded test.
    std::array<float, 4>         m_floats{ .1f, .2f, .3f, .4f }; ///< Modal or command payload.
    std::array<std::uint16_t, 2> m_camera{}; ///< Camera metadata source pixels, deliberately unused by the observer.
    MagAOX::dmCompletion::Wire   m_wire;     ///< Immutable software endpoint words.
    IMAGE                        m_image;    ///< Borrowed API-double view; never an actual ImageStreamIO stream.

    /// Prepare fixed layout, identity and recent local clock metadata.
    explicit ObservationImage( DDSPC::ObservationSource source /**< [in] Observed source format. */ );
};
ObservationImage::ObservationImage( DDSPC::ObservationSource source )
{
    m_metadata.naxis   = 2;
    m_metadata.size[0] = 4;
    m_metadata.size[1] = m_metadata.size[2] = 1;
    m_metadata.datatype                     = _DATATYPE_FLOAT;
    m_metadata.cnt0                         = 1;
    m_metadata.cnt1                         = 314;
    m_metadata.inode                        = source == DDSPC::ObservationSource::Dm ? 30 : 40;
    clock_gettime( CLOCK_REALTIME, &m_metadata.atime );
    m_metadata.writetime = m_metadata.atime;
    m_image.md           = &m_metadata;
    m_image.array.raw    = m_floats.data();
    if( source == DDSPC::ObservationSource::Camera )
    {
        m_metadata.datatype = _DATATYPE_UINT16;
        m_metadata.size[0]  = 2;
        m_image.array.raw   = m_camera.data();
    }
    if( source == DDSPC::ObservationSource::Completion || source == DDSPC::ObservationSource::Sdk )
    {
        m_metadata.naxis    = 1;
        m_metadata.size[0]  = 16;
        m_metadata.datatype = _DATATYPE_UINT64;
        m_wire =
            completion_wire( source == DDSPC::ObservationSource::Completion ? MagAOX::dmCompletion::Kind::completion
                                                                            : MagAOX::dmCompletion::Kind::sdk );
        m_image.array.raw = m_wire.data();
    }
}
} // namespace
/// \endcond

/** \brief Pure observation retains original times, records losses and refuses to invent WFS exposure associations. */
TEST_CASE( "Native observation records counter discontinuities clocks and unavailable WFS provenance",
           "[native-observation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::NativeObservationCapture::accept;
    DDSPC::NativeObservationCapture::tick;
#endif
    // clang-format on
    ObservationDirectory directory;
    auto                 c            = observation_config( directory );
    c.m_streams[1].m_acquisitionClock = c.m_streams[1].m_publicationClock = "realtime";
    DDSPC::NativeObservationCapture capture( c );
    auto                            r        = observation_frame( DDSPC::ObservationSource::Modal );
    auto                            accepted = capture.accept( r );
    REQUIRE( accepted.m_flags & DDSPC::ObservationAssociationMissing );
    REQUIRE( accepted.m_acquiredRawNs == r.m_acquiredRawNs );
    REQUIRE( accepted.m_acquiredMonoNs == -1 );
    r.m_counter = 4;
    r.m_availableNs += 500000;
    capture.accept( r );
    REQUIRE( capture.status().m_sources[1].m_counterGaps == 2 );
    REQUIRE( capture.status().m_sources[1].m_receivedRateHz == Approx( 2000 ) );
    r.m_counter    = 1;
    r.m_generation = 2;
    r.m_availableNs += 500000;
    capture.accept( r );
    REQUIRE( capture.status().m_sources[1].m_restarts == 1 );
    REQUIRE( capture.status().m_sources[1].m_receivedRateHz == 0 );
    REQUIRE( capture.status().m_sources[1].m_counterGaps == 2 );
    r.m_counter = 2;
    r.m_publishedRawNs += 40000000000LL;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationClockMismatch );
    REQUIRE( capture.status().m_sources[1].m_clockMismatches == 1 );
    capture.tick( r.m_availableNs + c.m_staleNs + 1 );
    REQUIRE( capture.status().m_sources[1].m_stale );
    auto camera = observation_frame( DDSPC::ObservationSource::Camera );
    accepted    = capture.accept( camera );
    REQUIRE( accepted.m_flags & DDSPC::ObservationUnknownClock );
    REQUIRE( accepted.m_publicationAgeNs == -1 );
    r.m_event = DDSPC::ObservationEvent::Disconnected;
    capture.accept( r );
    REQUIRE_FALSE( capture.status().m_sources[1].m_connected );
    REQUIRE_THROWS_AS( DDSPC::observation_clock( "auto" ), std::invalid_argument );
}

/** \brief Software completion kinds, malformed records, progression and metadata joins never certify application. */
TEST_CASE( "Native DM endpoint observation rejects incomplete mismatched and restarted records",
           "[native-observation]" )
{
    ObservationDirectory            directory;
    DDSPC::NativeObservationCapture capture( observation_config( directory ) );
    auto                            dm = observation_frame( DDSPC::ObservationSource::Dm );
    capture.accept( dm );
    auto r   = observation_frame( DDSPC::ObservationSource::Completion );
    r.m_wire = completion_wire( MagAOX::dmCompletion::Kind::completion );
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationAssociationMatched );
    r.m_wire[5]  = std::uint64_t( MagAOX::dmCompletion::State::busy );
    r.m_wire[11] = 0;
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationProtocolIncomplete );
    r.m_wire = completion_wire( MagAOX::dmCompletion::Kind::sdk );
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationProtocolInvalid );
    r.m_wire    = completion_wire( MagAOX::dmCompletion::Kind::completion );
    r.m_wire[4] = 4;
    ++r.m_counter;
    capture.accept( r );
    REQUIRE( capture.status().m_sources[4].m_sequenceGaps == 2 );
    r.m_wire[4] = 3;
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationProtocolInvalid );
    r.m_wire[2] = 99;
    r.m_wire[4] = 1;
    ++r.m_counter;
    capture.accept( r );
    REQUIRE( capture.status().m_sources[4].m_publisherRestarts == 1 );
    r.m_wire[7] = 999;
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationAssociationMissing );
    dm.m_generation = 2;
    dm.m_counter    = 2;
    capture.accept( dm );
    r.m_wire[7] = 1;
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags &
             DDSPC::ObservationAssociationMissing ); // Old DM generation no longer retained.
    dm.m_counter = 3;
    capture.accept( dm );
    dm.m_generation = 3;
    dm.m_event      = DDSPC::ObservationEvent::Attached;
    capture.accept( dm );
    r.m_wire[7] = 3;
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationAssociationMissing );
    r.m_wire[0] = 0;
    ++r.m_counter;
    REQUIRE( capture.accept( r ).m_flags & DDSPC::ObservationProtocolInvalid );
}

/** \brief Missing sources never create shared images, block a finite run, or prevent private output completion. */
TEST_CASE( "Native observation missing streams stop and record unqualified sources", "[native-observation][threaded]" )
{
    ObservationDirectory directory;
    auto                 c = observation_config( directory );
    c.m_durationSec        = .03;
    DDSPC::AdcConfig adc;
    adc.m_stream = c.m_streams[0].m_name;
    DDSPC::AdcReader      reader( adc ); // Not started: the actual observer must also tolerate an absent ADC source.
    DDSPC::NativeObserver observer( c );
    observer.start( reader );
    REQUIRE( observed( [&] { return observer.status().m_finished; } ) );
    observer.stop();
    REQUIRE_FALSE( observer.status().m_ioError );
    for( const auto &s : observer.status().m_sources )
    {
        REQUIRE( s.m_samples == 0 );
        REQUIRE( s.m_duplicates == 0 );
        REQUIRE_FALSE( s.m_connected );
    }
    std::ifstream     summary( directory.m_path / "summary.json" );
    const std::string text( ( std::istreambuf_iterator<char>( summary ) ), {} );
    REQUIRE( text.find( "\"training_enabled\":false" ) != std::string::npos );
    REQUIRE( text.find( "\"physical_application_qualified\":false" ) != std::string::npos );
    REQUIRE_THROWS_AS( observer.start( reader ), std::runtime_error ); // Existing journal is never overwritten.
}

/** \brief Queue overflow counts only journal loss and preserves chronological owned records after draining. */
TEST_CASE( "Native recording queue is bounded and recovers after overflow", "[native-observation]" )
{
    DDSPC::NativeObservationQueue queue( 2 );
    auto                          r = observation_frame( DDSPC::ObservationSource::Adc );
    REQUIRE( queue.push( r ) );
    r.m_counter = 2;
    REQUIRE( queue.push( r ) );
    r.m_counter = 3;
    REQUIRE_FALSE( queue.push( r ) );
    REQUIRE( queue.drops() == 1 );
    REQUIRE( queue.pop( r ) );
    REQUIRE( r.m_counter == 1 );
    REQUIRE( queue.pop( r ) );
    REQUIRE( r.m_counter == 2 );
    REQUIRE_FALSE( queue.pop( r ) );
    r.m_counter = 4;
    REQUIRE( queue.push( r ) );
    REQUIRE( queue.pop( r ) );
    REQUIRE( r.m_counter == 4 );
}

/** \brief Concurrent queue publication preserves whole records and accounts every overflow at sustained load. */
TEST_CASE( "Native recording queue publishes owned records across workers", "[native-observation][threaded]" )
{
    DDSPC::NativeObservationQueue queue( 128 );
    std::atomic<bool>             done{ false };
    std::uint64_t                 accepted = 0, consumed = 0, last = 0;
    bool                          valid = true;
    std::thread                   writer(
        [&]
        {
            for( ;; )
            {
                DDSPC::NativeObservation r;
                if( !queue.pop( r ) )
                {
                    if( !done.load( std::memory_order_acquire ) )
                        continue;
                    if( !queue.pop( r ) )
                        break;
                }
                valid = valid && r.m_counter > last && r.m_values[0] == double( r.m_counter ) &&
                        r.m_payloadHash == ( r.m_counter ^ 0x12345678ULL );
                last  = r.m_counter;
                ++consumed;
            }
        } );
    for( std::uint64_t n = 1; n <= 100000; ++n )
    {
        DDSPC::NativeObservation r;
        r.m_counter     = n;
        r.m_values[0]   = double( n );
        r.m_payloadHash = n ^ 0x12345678ULL;
        accepted += queue.push( r );
    }
    done.store( true, std::memory_order_release );
    writer.join();
    REQUIRE( valid );
    REQUIRE( consumed == accepted );
    REQUIRE( accepted + queue.drops() == 100000 );
}

/** \brief The actual observer forwards decoded ADC and reads immutable modal/camera/DM API doubles, then reconnects. */
TEST_CASE( "Native observation existing-only workers reconnect and preserve ADC decoding",
           "[native-observation][threaded]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::NativeObserver::start;
    DDSPC::NativeObserver::stop;
    DDSPC::AdcReader::copy_after;
#endif
    // clang-format on
    ObservationDirectory directory;
    auto                 c = observation_config( directory );
    c.m_streams[4].m_name  = "obs-completion";
    c.m_recordDmCommands   = true;
    c.m_streams[5].m_name  = "obs-sdk";
    ObservationImage modal( DDSPC::ObservationSource::Modal ), camera( DDSPC::ObservationSource::Camera ),
        dm( DDSPC::ObservationSource::Dm ), completion( DDSPC::ObservationSource::Completion ),
        sdk( DDSPC::ObservationSource::Sdk );
    AdcTransportFixture adcInput( c.m_streams[0].m_name, 4 ), modalInput( c.m_streams[1].m_name, modal.m_image ),
        cameraInput( c.m_streams[2].m_name, camera.m_image ), dmInput( c.m_streams[3].m_name, dm.m_image ),
        completionInput( c.m_streams[4].m_name, completion.m_image ), sdkInput( c.m_streams[5].m_name, sdk.m_image );
    DDSPC::AdcConfig adc;
    adc.m_stream   = c.m_streams[0].m_name;
    adc.m_maxAgeNs = 2000000000;
    DDSPC::AdcReader reader( adc );
    reader.start();
    DDSPC::NativeObserver observer( c );
    observer.start( reader );
    REQUIRE( observed(
        [&]
        {
            const auto s = observer.status();
            return s.m_sources[0].m_samples == 1 && s.m_sources[5].m_samples == 1;
        } ) );
    auto s = observer.status();
    REQUIRE( s.m_sources[0].m_latest.m_raw[0] == 2049 );
    REQUIRE( s.m_sources[0].m_latest.m_values[1] == Approx( -.0083799 / 1.02 ) );
    REQUIRE( s.m_sources[1].m_latest.m_values[0] == Approx( .1 ) );
    REQUIRE( s.m_sources[1].m_latest.m_hasCounter1 );
    REQUIRE( s.m_sources[1].m_latest.m_counter1 == 314 );
    REQUIRE_FALSE( s.m_sources[0].m_latest.m_hasCounter1 );
    REQUIRE( s.m_sources[3].m_latest.m_imageInode == 30 );
    REQUIRE( s.m_sources[3].m_latest.m_fileInode != 30 );
    REQUIRE( s.m_sources[4].m_protocolInvalid == 0 );
    adcInput.disconnect();
    REQUIRE( observed( [&] { return !observer.status().m_sources[0].m_connected; } ) );
    adcInput.reconnect();
    REQUIRE( observed( [&] { return observer.status().m_sources[0].m_samples == 2; } ) );
    modalInput.disconnect();
    REQUIRE( observed( [&] { return !observer.status().m_sources[1].m_connected; } ) );
    modalInput.reconnect();
    REQUIRE( observed(
        [&]
        {
            return observer.status().m_sources[1].m_latest.m_generation == 2 &&
                   observer.status().m_sources[1].m_samples == 2;
        } ) );
    dmInput.replace();
    REQUIRE( observed( [&] { return observer.status().m_sources[3].m_restarts == 1; } ) );
    const auto before = std::chrono::steady_clock::now();
    observer.stop();
    reader.stop();
    REQUIRE( std::chrono::steady_clock::now() - before < std::chrono::seconds( 1 ) );
    s = observer.status();
    REQUIRE_FALSE( s.m_ioError );
    REQUIRE( s.m_written > 8 );
    REQUIRE( s.m_sources[1].m_associationFailures == 2 ); // No guessed camera association.
    REQUIRE( s.m_sources[1].m_restarts == 1 );
    REQUIRE( s.m_commandsWritten == s.m_sources[3].m_samples );
    REQUIRE( s.m_commandDrops == 0 );
    REQUIRE( s.m_commandOversize == 0 );
    std::ifstream                 commandFile( directory.m_path / "dm_commands.bin", std::ios::binary );
    std::array<char, 8>           magic{};
    std::array<std::uint64_t, 17> header{};
    std::array<float, 4>          values{};
    commandFile.read( magic.data(), magic.size() );
    commandFile.read( reinterpret_cast<char *>( header.data() ), sizeof( header ) );
    commandFile.read( reinterpret_cast<char *>( values.data() ), sizeof( values ) );
    REQUIRE( commandFile.good() );
    REQUIRE( std::string( magic.data(), magic.size() ) == "DDDM0001" );
    REQUIRE( header[0] == 4 );
    REQUIRE( header[3] == 30 );
    REQUIRE( header[5] == 1 );
    REQUIRE( header[7] ==
             std::uint64_t( dm.m_metadata.writetime.tv_sec ) * 1000000000 + dm.m_metadata.writetime.tv_nsec );
    REQUIRE( values == dm.m_floats );
    observer.stop();
}

/** \brief Complete-command queue overflow never blocks source observation or truncates retained payloads. */
TEST_CASE( "Full native command queue preserves payloads and reports bounded overflow", "[native-observation]" )
{
    DDSPC::NativeCommandQueue queue( 2, 4 );
    auto                      r = observation_frame( DDSPC::ObservationSource::Dm );
    r.m_elements                = 4;
    std::array<float, 4> values{ 1, 2, 3, 4 }, output{};
    REQUIRE( queue.push( r, values ) );
    REQUIRE( queue.push( r, values ) );
    REQUIRE_FALSE( queue.push( r, values ) );
    REQUIRE( queue.drops() == 1 );
    REQUIRE( queue.pop( r, output ) );
    REQUIRE( output == values );
    REQUIRE( queue.pop( r, output ) );
    REQUIRE_FALSE( queue.pop( r, output ) );
    REQUIRE_THROWS_AS( queue.push( r, std::span<const float>( values.data(), 3 ) ), std::invalid_argument );
}

/** \brief Oversized command images remain metadata observations and are never serialized as truncated inputs. */
TEST_CASE( "Full native command recording diagnoses oversize without dropping metadata",
           "[native-observation][threaded]" )
{
    ObservationDirectory directory;
    auto                 c = observation_config( directory );
    c.m_recordDmCommands   = true;
    c.m_commandMaxElements = 2;
    ObservationImage      dm( DDSPC::ObservationSource::Dm );
    AdcTransportFixture   input( c.m_streams[3].m_name, dm.m_image );
    DDSPC::AdcReader      reader( DDSPC::AdcConfig{} );
    DDSPC::NativeObserver observer( c );
    observer.start( reader );
    REQUIRE( observed( [&] { return observer.status().m_commandOversize == 1; } ) );
    observer.stop();
    const auto status = observer.status();
    REQUIRE( status.m_sources[3].m_samples == 1 );
    REQUIRE( status.m_sources[3].m_latest.m_elements == 4 );
    REQUIRE( status.m_commandsWritten == 0 );
    REQUIRE( status.m_commandDrops == 0 );
    REQUIRE_FALSE( status.m_ioError );
    REQUIRE( std::filesystem::file_size( directory.m_path / "dm_commands.bin" ) == 8 );
}

/** \brief Invalid layouts are observed without FLOAT reinterpretation or out-of-bounds copying. */
TEST_CASE( "Native observation rejects incompatible source dimensions and datatypes", "[native-observation][threaded]" )
{
    const int            fault = GENERATE( 0, 1, 2 );
    ObservationDirectory directory;
    auto                 c = observation_config( directory );
    ObservationImage     modal( DDSPC::ObservationSource::Modal );
    if( fault == 0 )
        modal.m_metadata.datatype = _DATATYPE_UINT16;
    if( fault == 1 )
        modal.m_metadata.size[1] = 2;
    if( fault == 2 )
        modal.m_metadata.size[0] = 1;
    AdcTransportFixture   input( c.m_streams[1].m_name, modal.m_image );
    DDSPC::AdcReader      reader( DDSPC::AdcConfig{} );
    DDSPC::NativeObserver observer( c );
    observer.start( reader );
    REQUIRE( observed( [&] { return observer.status().m_sources[1].m_incompatible > 0; } ) );
    observer.stop();
    REQUIRE( observer.status().m_sources[1].m_samples == 0 );
    REQUIRE_FALSE( observer.status().m_ioError );
}
} // namespace libXWCTest::loPredCtrlAccTest
