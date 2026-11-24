/** \file ocamFrameDelay_test.cpp
 * \brief Offline cadence, timing handoff, and rolling frame history tests.
 * \ingroup ocamFrameDelay_unit_test
 */
#include <fstream>

#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/app/dev/frameGrabberTiming.hpp"
#include "../../../libMagAOX/logger/logBinarySchemata.hpp"
#include "../../../libMagAOX/logger/generated/logMemberAccessor.hpp"
#include "../ocamFrameDelay.hpp"

namespace libXWCTest::ocam2KCtrlTest
{
using Monitor = MagAOX::app::ocamFrameDelay;
using namespace MagAOX::fgTiming;

/// \cond OCAM_FRAME_DELAY_TEST_HELPERS
namespace
{
/// Supply a complete event with separate controllable clock domains.
Event frame( uint64_t               sequence /**< [in] Timing attempt sequence. */,
             std::array<int64_t, 3> stamps /**< [in] Three endpoint timestamps, in ns. */,
             double                 fps = 4 /**< [in] Effective frame rate. */ )
{
    Event event;
    event.sequence      = sequence;
    event.session       = 1;
    event.frameNumber   = sequence;
    event.outputCounter = sequence;
    event.streamInode   = 10;
    event.fps           = fps;
    event.frameValid = event.outputValid = true;
    event.atime                          = { stamps[0] / 1000000000, stamps[0] % 1000000000 };
    event.arrivalNs                      = stamps[1];
    event.writetime                      = { stamps[2] / 1000000000, stamps[2] % 1000000000 };
    event.outcome                        = Outcome::published;
    return event;
}

/// Own a unique temporary directory for dump files.
struct DumpDirectory
{
    std::filesystem::path path; ///< Directory removed at test exit.

    /// Create the temporary directory.
    DumpDirectory();

    /// Remove only the test-owned temporary directory.
    ~DumpDirectory();
};

DumpDirectory::DumpDirectory()
{
    char name[] = "/tmp/ocam-frame-delay-XXXXXX";
    REQUIRE( mkdtemp( name ) != nullptr );
    path = name;
}

DumpDirectory::~DumpDirectory()
{
    std::error_code ignored;
    std::filesystem::remove_all( path, ignored );
}
} //namespace
/// \endcond

/// Each endpoint observes the inclusive fractional threshold independently of its enable switch.
/** \ingroup ocamFrameDelay_unit_test */
TEST_CASE( "OCAM single-delay alarms use an inclusive fractional frame", "[ocamFrameDelay]" )
{
    for( double fraction : { 0.5, 0.99, 1.0 } )
    {
        for( size_t index = 0; index < 3; ++index )
        {
            Monitor detector;
            detector.configure( 900, 6, fraction );
            constexpr int64_t      start  = 1000000000;
            constexpr int64_t      period = 250000000;
            const auto             limit  = static_cast<int64_t>( std::round( ( 1 + fraction ) * period ) );
            std::array<int64_t, 3> stamps{ start, start, start };
            CHECK( detector.observe( frame( 1, stamps ), Monitor::allChecks ) == 0 );
            stamps.fill( start + period );
            stamps[index] = start + limit - 1;
            CHECK( detector.observe( frame( 2, stamps ), Monitor::allChecks ) == 0 );

            for( size_t i = 0; i < stamps.size(); ++i )
            {
                stamps[i] += i == index ? limit : period;
            }

            const unsigned bit = 1u << index;
            CHECK( detector.observe( frame( 3, stamps ), Monitor::allChecks & ~bit ) == 0 );
            CHECK( detector.reading().elapsed[index] == limit );

            for( size_t i = 0; i < stamps.size(); ++i )
            {
                stamps[i] += i == index ? limit : period;
            }

            CHECK( detector.observe( frame( 4, stamps ), Monitor::allChecks ) == bit );
            CHECK( detector.reading().latched == bit );
            detector.restart( 5 );
            CHECK( detector.reading().latched == 0 );
            CHECK( detector.reading().elapsed[index] == -1 );
        }
    }
}

/// Gradual drift trips the four-interval sum while sequence and clock discontinuities reset only affected baselines.
/** \ingroup ocamFrameDelay_unit_test */
TEST_CASE( "OCAM integrates adjacent intervals and restarts cleanly on gaps", "[ocamFrameDelay]" )
{
    Monitor detector;
    detector.configure( 900, 6, 1 );

    for( uint64_t i = 0; i < 5; ++i )
    {
        int64_t nominal   = 1000000000 + i * 250000000;
        int64_t late      = 1000000000 + i * 312500000;
        auto    event     = frame( i + 1, { nominal, late, nominal } );
        event.frameNumber = static_cast<uint32_t>( UINT32_MAX - 1 + i );
        CHECK( detector.observe( event, Monitor::allChecks ) == ( i == 4 ? 16 : 0 ) );
    }

    const auto &reading = detector.reading();
    CHECK( reading.elapsed[4] == 1250000000 );
    CHECK( reading.elapsed[3] == 1000000000 );
    CHECK( reading.latched == 16 );
    detector.observe( frame( 7, { 2250000000, 2500000000, 2250000000 }, 3.5 ), Monitor::allChecks );
    CHECK( reading.samples == 4 );
    CHECK( reading.elapsed[4] == -1 );
    CHECK( reading.latched == 16 );
    detector.observe( frame( 8, { 2500000000, -1, 2500000000 }, 3.5 ), Monitor::allChecks );
    CHECK( reading.elapsed[0] == 250000000 );
    CHECK( reading.elapsed[1] == -1 );
    detector.observe( frame( 9, { 2499999999, 2750000000, 2750000000 }, 3.5 ), Monitor::allChecks );
    CHECK( reading.elapsed[0] == -1 );
    CHECK( reading.elapsed[2] == 250000000 );
}

/// The raw observer gets the SDK return without detailed stage timing, and the fence covers dropped attempts.
/** \ingroup ocamFrameDelay_unit_test */
TEST_CASE( "OCAM raw timing handoff includes arrival and a loss-aware fence", "[ocamFrameDelay]" )
{
    MagAOX::app::dev::frameGrabberTiming timing;
    std::vector<Event>                   events;
    timing.observe(
        []( void *context, const Event &event )
        {
            static_cast<std::vector<Event> *>( context )->push_back( event );
        },
        &events );
    timing.configure( 2, 1 );
    timing.begin( 2000 );
    timing.mark( Marker::waitReturn );
    timing.acquired( { 100, 20 } );
    timing.finish( Outcome::rejected );
    timing.begin( 2000 );
    timing.finish( Outcome::failed );
    timing.observeRestart();
    timing.update();
    REQUIRE( events.size() == 1 );
    CHECK( events.front().arrivalNs >= 0 );
    CHECK( events.front().valid == 0 );
    CHECK( events.front().atime.tv_nsec == 20 );
    CHECK( events.front().fps == 2000 );
    CHECK( timing.drainedSequence() == 2 );
    CHECK( timing.droppedEvents() == 1 );
    CHECK( timing.observedRestart() == 3 );
    timing.begin( 3000 );
    timing.finish();
    timing.update();
    REQUIRE( events.size() == 2 );
    CHECK( events.back().restart == 3 );
    CHECK( events.back().fps == 3000 );
}

/// Keep the requested chronological slice across sealed chunks and rotate binary output after successful writes.
/** \ingroup ocamFrameDelay_unit_test */
TEST_CASE( "OCAM manual snapshots share chunks and enforce both time and sequence cutoff", "[ocamFrameDelay]" )
{
    DumpDirectory directory;
    Monitor       detector;
    detector.configure( 1, 6000, 1 );

    for( uint64_t i = 1; i <= 4600; ++i )
    {
        const int64_t ns = 1000000000 + i * 250000;
        detector.observe( frame( i, { ns, ns, ns }, 4000 ), 0 );
    }

    Monitor::Snapshot snapshot;
    snapshot.cutoff   = 2100000000;
    snapshot.boundary = 4300;
    snapshot.dropped  = 4;
    detector.freeze( snapshot );
    REQUIRE( snapshot.rows == 3901 );
    REQUIRE( snapshot.epochs.size() == 1 );
    detector.expire( 4200000000 );
    std::filesystem::path file;

    for( uint64_t i = 1; i <= 6; ++i )
    {
        snapshot.id          = i;
        snapshot.requestedAt = 123456780 + i;
        file                 = Monitor::write( snapshot, directory.path, "camwfs" );
    }

    CHECK( std::distance( std::filesystem::directory_iterator( directory.path ),
                          std::filesystem::directory_iterator() ) == 5 );
    std::ifstream       input( file, std::ios::binary );
    std::array<char, 8> magic{};
    input.read( magic.data(), magic.size() );
    CHECK( std::string( magic.data(), magic.size() ) == "OCAMFD01" );
    std::array<uint64_t, 9> header{};
    input.read( reinterpret_cast<char *>( header.data() ), sizeof( header ) );
    CHECK( header[3] == 4300 );
    CHECK( header[4] == 4 );
    CHECK( header[6] == 3901 );
    input.seekg( sizeof( Monitor::Epoch ), std::ios::cur );
    Monitor::Row row{};
    input.read( reinterpret_cast<char *>( &row ), sizeof( row ) );
    CHECK( row.sequence == 400 );
    input.seekg( -static_cast<std::streamoff>( sizeof( row ) ), std::ios::end );
    input.read( reinterpret_cast<char *>( &row ), sizeof( row ) );
    CHECK( row.sequence == 4300 );
    CHECK_THROWS( Monitor::write( snapshot, file / "directory", "camwfs" ) );
}

/// Preserve elapsed and signed excess measurements through the binary logger, metadata, text and JSON.
/** \ingroup ocamFrameDelay_unit_test */
TEST_CASE( "OCAM cadence telemetry round trips through the logger", "[ocamFrameDelay]" )
{
    using namespace MagAOX::logger;
    telem_ocamdelay::messageT input( 2000,
                                     2000,
                                     0.99,
                                     { 499000, 500000, -1, 1000999999, 1000200000, -1 },
                                     { -1000, 0, 0, 999999, 200000, 0 },
                                     63,
                                     8,
                                     2 );
    flatlogs::bufferPtrT      envelope;
    REQUIRE( flatlogs::logHeader::createLog<telem_ocamdelay>(
                 envelope, { 1, 2 }, input, flatlogs::logPrio::LOG_TELEM ) == 0 );
    CHECK( verifyLogEntry( telem_ocamdelay::eventCode, envelope.get() ) );
    auto *payload = input.builder.GetBufferPointer();
    CHECK( telem_ocamdelay::msgString( payload, input.builder.GetSize() ).find( "atime_sum: 1000999999" ) !=
           std::string::npos );
    const auto meta = logMemberAccessor( telem_ocamdelay::eventCode, "atime_sum" );
    REQUIRE( meta.accessor != nullptr );
    CHECK( reinterpret_cast<double ( * )( void * )>( meta.accessor )( payload ) == 1000999999 );
    const auto *schema = reinterpret_cast<const uint8_t *>( telem_ocamdelay_bfbs );
    const auto  json   = telem_ocamdelay::msgJSON( payload, input.builder.GetSize(), schema, telem_ocamdelay_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( schema, telem_ocamdelay_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *record = GetTelem_ocamdelay_fb( parser.builder_.GetBufferPointer() );
    CHECK( record->elapsed()->write() == -1 );
    CHECK( record->excess()->atime() == -1000 );
    CHECK( record->excess()->atime_sum() == 999999 );
    CHECK( record->fraction() == 0.99 );
    CHECK( record->latched() == 8 );
    CHECK( record->dropped() == 2 );
}
} //namespace libXWCTest::ocam2KCtrlTest
