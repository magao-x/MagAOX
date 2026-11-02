/** \file dlDataCollection_test.cpp
 * \brief CPU regressions for production capture, bank and file-failure logic.
 * \ingroup dlDataCollection_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../dlCollection.hpp"
#include "../dlObservation.hpp"

#include <atomic>
#include <thread>

namespace libXWCTest
{
/// Tests of the end application's hardware-independent production state machine.
/** \ingroup dlDataCollection_unit_test */
namespace dlDataCollectionTest
{
using namespace MagAOX::app::dlCollection;

/** \cond DLDATACOLLECTION_TEST_SUPPORT */
/// Own an exclusive temporary fixture and its real background worker.
class Fixture
{
  public:
    /// Temporary directory removed after the worker exits.
    std::filesystem::path m_root;

    /// Small valid bank/run settings.
    Settings m_settings;

    /// Current fake producer counter, read by the publisher.
    std::atomic<uint64_t> m_counter{ 0 };

    /// Count nonzero-command publications, independent of release requests.
    std::atomic<size_t> m_publications{ 0 };

    /// Current fake instrument channel value.
    std::atomic<float> m_command{ 0 };

    /// Release a deliberately stalled writer even when an assertion fails.
    std::atomic<bool> m_release{ false };

    /// Real production session with fake hardware, not a mirrored state machine.
    Session m_session;

    /// Worker lifetime is always joined before fixture destruction.
    std::thread m_thread;

    /// Build a small two-set bank, optionally injecting output faults.
    explicit Fixture( Session::FileWriter writer /**< [in] Optional output adapter. */ = writeFile,
                      size_t              samples /**< [in] Samples per set. */        = 2,
                      size_t              sets /**< [in] Available sets. */            = 2 );

    /// Stop and join the worker before removing its files.
    ~Fixture();

    /// Wait boundedly for one phase to appear.
    bool wait( const std::string &phase /**< [in] Expected phase. */ );

    /// Submit a stable constant image with one producer counter.
    void frame( uint64_t counter /**< [in] Producer counter. */, bool stable /**< [in] Snapshot stability. */ = true );
};

/// Release a deliberately stalled output even when a test assertion throws.
struct Release
{
    /// Test latch to unblock before the fixture joins its worker.
    std::atomic<bool> &m_flag;

    /// Set the latch on scope exit.
    ~Release();
};

/// Build a named-position readback without connecting to INDI or hardware.
pcf::IndiProperty
preset( const std::string                   &name /**< [in] Selected position. */,
        pcf::IndiProperty::PropertyStateType state /**< [in] Motion/health state. */ = pcf::IndiProperty::Idle );

pcf::IndiProperty preset( const std::string &name, pcf::IndiProperty::PropertyStateType state )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch );
    property.setState( state );
    property.add( pcf::IndiElement( name ) );
    property[name].setSwitchState( pcf::IndiElement::On );
    return property;
}

/// Build a controller status readback without contacting an INDI server.
pcf::IndiProperty controllerState( const std::string &state /**< [in] FSM state name. */ );

pcf::IndiProperty controllerState( const std::string &state )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text );
    property.add( pcf::IndiElement( "state" ) );
    property["state"] = state;
    return property;
}

/// Build a camera number property without contacting an INDI server.
pcf::IndiProperty cameraSetting( const std::string &value /**< [in] Reported current value. */ );

pcf::IndiProperty cameraSetting( const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number );
    property.setState( pcf::IndiProperty::Ok );
    property.add( pcf::IndiElement( "current" ) );
    property["current"] = value;
    return property;
}

Release::~Release()
{
    m_flag = true;
}

Fixture::Fixture( Session::FileWriter writer, size_t samples, size_t sets )
    : m_session(
          [this]( const float *command )
          {
              if( command )
              {
                  ++m_publications;
                  m_command = command[0];
              }
              else
                  m_command = 0;
              return m_counter.load();
          },
          std::move( writer ) )
{
    char        name[] = "/tmp/dlDataCollection-test-XXXXXX";
    const char *path   = ::mkdtemp( name );
    if( !path )
        throw std::runtime_error( "temporary fixture creation failed" );
    m_root               = path;
    m_settings.m_bank    = m_root / "bank";
    m_settings.m_output  = m_root / "output";
    m_settings.m_samples = samples;
    m_settings.m_sets    = sets;
    m_settings.m_pupil   = 6;
    m_settings.m_height  = 4;
    m_settings.m_width   = 4;
    m_settings.m_x       = { 0, 2 };
    m_settings.m_y       = { 0, 2 };
    m_settings.m_skip    = 8;
    std::filesystem::create_directory( m_settings.m_bank );
    std::vector<float> mask( 2500, 1.F );
    mask[1744] = 0;
    writeFile( m_settings.m_bank / "actuator_mask.bin", mask.data(), 10000 );
    const std::string generation = "{\"test_fixture\":true}\n";
    writeText( m_settings.m_bank / "generation.json", generation );
    std::ostringstream manifest;
    manifest << "format=dlDataCollection-1\ndtype=float32-le\nunits=um_surface\nmodes=2500\nsamples=" << samples
             << "\nsets=" << sets << "\nmask_sha256=" << digest( mask.data(), 10000 )
             << "\ngeneration_sha256=" << digest( generation.data(), generation.size() ) << '\n';
    for( size_t set = 0; set < sets; ++set )
    {
        std::vector<float> values( samples * 2500 );
        for( size_t sample = 0; sample < samples; ++sample )
            for( size_t i = 0; i < 2500; ++i )
                values[sample * 2500 + i] = mask[i] * float( 1 + set * samples + sample );
        const auto name = "modeval_dataset_" + std::to_string( set ) + ".bin";
        writeFile( m_settings.m_bank / name, values.data(), values.size() * sizeof( float ) );
        manifest << "set_" << set << "_sha256=" << digest( values.data(), values.size() * sizeof( float ) ) << '\n';
    }
    writeText( m_settings.m_bank / "manifest.conf", manifest.str() );
    m_thread = std::thread( [this] { m_session.ioLoop(); } );
}

Fixture::~Fixture()
{
    m_release = true;
    m_session.shutdown();
    if( m_thread.joinable() )
        m_thread.join();
    std::filesystem::remove_all( m_root );
}

bool Fixture::wait( const std::string &phase )
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );
    while( std::chrono::steady_clock::now() < deadline )
    {
        if( m_session.progress().m_phase == phase )
            return true;
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    return false;
}

void Fixture::frame( uint64_t counter, bool stable )
{
    m_counter = counter;
    std::array<float, 16> image;
    image.fill( m_command.load() );
    m_session.onFrame( counter, counter * 1000000, image.data(), [stable] { return stable; } );
}
/** \endcond */

/// Capture exactly the requested sets, with one publication per image and matching labels.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection completes exactly Nset with matched commands", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::start;
    MagAOX::app::dlCollection::Session::onFrame;
    #endif
    // clang-format on
    Fixture f;
    REQUIRE( f.m_publications == 0 );
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    REQUIRE( f.m_publications == 0 );
    for( uint64_t counter = 0; counter < 1000 && f.m_session.progress().m_saved < 4; ++counter )
    {
        f.frame( counter );
        std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
    }
    REQUIRE( f.wait( "complete" ) );
    REQUIRE( f.m_session.progress().m_saved == 4 );
    REQUIRE( f.m_session.progress().m_written == 2 );
    REQUIRE( f.m_publications == 4 );
    REQUIRE( f.m_command == 0 );
    const auto run = std::filesystem::path( f.m_session.progress().m_directory );
    REQUIRE_FALSE( std::filesystem::exists( run / "images_dataset_2.bin" ) );
    for( size_t set = 0; set < 2; ++set )
    {
        const auto suffix   = "dataset_" + std::to_string( set );
        const auto images   = readFloats( run / ( "images_" + suffix + ".bin" ), 2 * 4 * 36 );
        const auto commands = readFloats( run / ( "commands_" + suffix + ".bin" ), 2 * 2500 );
        REQUIRE( images[0] == 0.F );
        REQUIRE( images[14] == commands[0] );
        REQUIRE( images[4 * 36 + 14] == commands[2500] );
        REQUIRE( commands[1744] == 0.F );
        REQUIRE( std::filesystem::exists( run / ( "complete_" + suffix + ".conf" ) ) );
    }
    const auto old = f.m_session.progress().m_directory;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.m_session.abort();
    REQUIRE( f.wait( "cancelled" ) );
    REQUIRE( f.m_session.progress().m_directory != old );
    REQUIRE( std::filesystem::exists( std::filesystem::path( old ) / "images_dataset_0.bin" ) );
}

/// Duplicate notifications and torn snapshots never advance a command's settling interval.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection waits on producer counters and rejects torn images", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::onFrame;
    #endif
    // clang-format on
    Fixture f;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 100 );
    for( int i = 0; i < 30; ++i )
        f.frame( 100 );
    REQUIRE( f.m_publications == 1 );
    REQUIRE( f.m_session.progress().m_saved == 0 );
    REQUIRE( f.m_session.progress().m_duplicates == 30 );
    f.frame( 107 );
    REQUIRE( f.m_session.progress().m_saved == 0 );
    f.frame( 109, false );
    REQUIRE( f.m_session.progress().m_saved == 0 );
    REQUIRE( f.m_session.progress().m_torn == 1 );
    f.frame( 110 );
    REQUIRE( f.m_session.progress().m_saved == 1 );
    REQUIRE( f.m_publications == 2 );
    for( int i = 0; i < 30; ++i )
        f.frame( 110 );
    f.frame( 117 );
    REQUIRE( f.m_session.progress().m_saved == 1 );
    f.frame( 118 );
    REQUIRE( f.m_session.progress().m_saved == 2 );
    f.m_session.abort();
    REQUIRE( f.wait( "cancelled" ) );
}

/// Zero skipped frames still requires a new frame after command publication.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection zero skip captures subsequent fresh frames", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::onFrame;
    #endif
    // clang-format on
    Fixture f;
    f.m_settings.m_skip = 0;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 1 );
    f.frame( 1 );
    REQUIRE( f.m_session.progress().m_saved == 0 );
    f.frame( 2 );
    REQUIRE( f.m_session.progress().m_saved == 1 );
    f.frame( 3 );
    REQUIRE( f.m_session.progress().m_saved == 2 );
}

/// Stop, counter reset and reconnect preserve only the valid accepted prefix.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection preserves partial data when acquisition is interrupted", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::abort;
    #endif
    // clang-format on
    Fixture f;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 100 );
    f.frame( 109 );
    REQUIRE( f.m_session.progress().m_saved == 1 );
    SECTION( "operator Stop" )
    {
        f.m_session.abort();
        REQUIRE( f.wait( "cancelled" ) );
    }
    SECTION( "producer reset" )
    {
        f.frame( 0 );
        REQUIRE( f.wait( "failed" ) );
    }
    SECTION( "camera reconnect" )
    {
        f.m_session.abort( "source stream reconnected; partial run aborted" );
        REQUIRE( f.wait( "failed" ) );
    }
    REQUIRE( f.m_command == 0 );
    const auto run = std::filesystem::path( f.m_session.progress().m_directory );
    REQUIRE( std::filesystem::file_size( run / "images_partial_dataset_0.bin" ) == 4 * 36 * sizeof( float ) );
    REQUIRE( std::filesystem::file_size( run / "commands_partial_dataset_0.bin" ) == 2500 * sizeof( float ) );
    REQUIRE_FALSE( std::filesystem::exists( run / "complete_dataset_0.conf" ) );
    REQUIRE( f.m_session.progress().m_saved == 1 );
}

/// A stalled source is a visible failure and releases the collector's command.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection watchdog stops a stalled source", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::checkTimeout;
    #endif
    // clang-format on
    Fixture f;
    f.m_settings.m_timeout = .01;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 100 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
    f.m_session.checkTimeout();
    REQUIRE( f.wait( "failed" ) );
    REQUIRE( f.m_command == 0 );
    REQUIRE( f.m_session.progress().m_error.find( "timeout" ) != std::string::npos );
}

/// Wrong counts, hashes, units, masks, and non-finite values fail before hardware publication.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection preflights the entire requested bank", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::readBank;
    MagAOX::app::dlCollection::readSet;
    #endif
    // clang-format on
    Fixture f;
    SECTION( "missing later set" )
    {
        std::filesystem::remove( f.m_settings.m_bank / "modeval_dataset_1.bin" );
    }
    SECTION( "trailing bytes" )
    {
        std::ofstream out( f.m_settings.m_bank / "modeval_dataset_1.bin", std::ios::binary | std::ios::app );
        out.put( 'x' );
    }
    SECTION( "checksum changed" )
    {
        std::fstream out( f.m_settings.m_bank / "modeval_dataset_1.bin",
                          std::ios::binary | std::ios::in | std::ios::out );
        out.put( 'x' );
    }
    SECTION( "Nperset disagreement" )
    {
        f.m_settings.m_samples = 3;
    }
    SECTION( "Nset outside bank" )
    {
        f.m_settings.m_sets = 3;
    }
    SECTION( "unsupported units" )
    {
        const auto    path = f.m_settings.m_bank / "manifest.conf";
        std::ifstream input( path );
        std::string   text( ( std::istreambuf_iterator<char>( input ) ), {} );
        text.replace( text.find( "um_surface" ), 10, "volts" );
        std::ofstream( path ) << text;
    }
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "failed" ) );
    REQUIRE( f.m_publications == 0 );
    REQUIRE_FALSE( f.m_session.progress().m_error.empty() );
}

/// Matching checksums do not permit excluded-actuator commands or NaN/Inf.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection rejects unsafe float32 values independently of hashes", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::readSet;
    #endif
    // clang-format on
    Fixture f;
    auto    bank   = readBank( f.m_settings.m_bank );
    auto    values = readSet( bank, 0 );
    SECTION( "excluded actuator" )
    {
        values[1744] = 1;
    }
    SECTION( "NaN" )
    {
        const uint32_t bits = 0x7fc00000U;
        std::memcpy( values.data(), &bits, sizeof( bits ) );
    }
    SECTION( "Inf" )
    {
        const uint32_t bits = 0x7f800000U;
        std::memcpy( values.data(), &bits, sizeof( bits ) );
    }
    const auto    path = f.m_settings.m_bank / "modeval_dataset_0.bin";
    std::ofstream out( path, std::ios::binary | std::ios::trunc );
    out.write( reinterpret_cast<const char *>( values.data() ), values.size() * sizeof( float ) );
    out.close();
    bank.m_hashes[0] = digest( values.data(), values.size() * sizeof( float ) );
    REQUIRE_THROWS( readSet( bank, 0 ) );
}

/// A preflight failure has no unsaved payload and can be corrected in the same process.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection permits recovery after preflight failure", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::canStart;
    MagAOX::app::dlCollection::Session::start;
    #endif
    // clang-format on
    Fixture f;
    f.m_settings.m_samples = 3;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "failed" ) );
    REQUIRE( f.m_session.canStart() );
    f.m_settings.m_samples = 2;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 1 );
    f.m_session.abort();
    REQUIRE( f.wait( "cancelled" ) );
}

/// Pupil layout is unchanged from the old column-major Eigen map, including padding.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection preserves pupil orientation and normalization", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::onFrame;
    #endif
    // clang-format on
    Fixture f( writeFile, 1, 1 );
    f.m_settings.m_imageNorm = 2;
    f.m_settings.m_skip      = 0;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 1 );
    std::array<float, 16> image;
    for( size_t i = 0; i < image.size(); ++i )
        image[i] = float( i );
    f.m_counter = 2;
    f.m_session.onFrame( 2, 2000000, image.data(), [] { return true; } );
    REQUIRE( f.wait( "complete" ) );
    const auto output =
        readFloats( std::filesystem::path( f.m_session.progress().m_directory ) / "images_dataset_0.bin", 144 );
    REQUIRE( output[14] == 0.F );
    REQUIRE( output[15] == 2.F );
    REQUIRE( output[20] == 8.F );
    REQUIRE( output[36 + 14] == 16.F );
    REQUIRE( output[72 + 14] == 4.F );
    REQUIRE( output[108 + 14] == 20.F );
    REQUIRE( output[13] == 0.F );
    REQUIRE( output[26] == 0.F );
}

/// A conflicting command writer invalidates the pending image before any sample is accepted.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection rejects a lost-command snapshot before saving", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::onFrame;
    #endif
    // clang-format on
    Fixture f;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 1 );
    f.m_counter = 9;
    std::array<float, 16> image{};
    f.m_session.onFrame( 9,
                         9000000,
                         image.data(),
                         []() -> bool
                         { throw std::runtime_error( "command channel changed or another writer intervened" ); } );
    REQUIRE( f.wait( "failed" ) );
    REQUIRE( f.m_session.progress().m_saved == 0 );
    REQUIRE( f.m_command == 0 );
    REQUIRE( f.m_session.progress().m_error.find( "another writer" ) != std::string::npos );
}

/// A stalled writer bounds capture at two set buffers and releases the command before waiting.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection applies bounded backpressure outside the capture thread", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::ioLoop;
    MagAOX::app::dlCollection::Session::onFrame;
    #endif
    // clang-format on
    std::atomic<bool> blocked{ false }, release{ false };
    auto              writer = [&]( const auto &path, const void *data, size_t size )
    {
        if( path.filename() == "images_dataset_0.bin" )
        {
            blocked = true;
            while( !release.load() )
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        }
        writeFile( path, data, size );
    };
    Fixture f( writer, 1, 3 );
    // Unblock on every exit path before the fixture destructor joins the worker.
    Release unblock{ release };
    f.m_settings.m_skip = 0;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    for( uint64_t c = 1; c < 500 && !blocked; ++c )
    {
        f.frame( c );
        std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
    }
    REQUIRE( blocked );
    for( uint64_t c = 501; c < 1501; ++c )
        f.frame( c );
    REQUIRE( f.m_session.progress().m_saved <= 2 );
    REQUIRE( f.m_session.progress().m_phase == "waiting_io" );
    REQUIRE( f.m_command == 0 );
    release = true;
    for( uint64_t c = 1501; c < 2000 && f.m_session.progress().m_saved < 3; ++c )
    {
        f.frame( c );
        std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
    }
    REQUIRE( f.wait( "complete" ) );
    REQUIRE( f.m_session.progress().m_saved == 3 );
    REQUIRE( f.m_publications == 3 );
}

/// File errors prevent completion, expose the failure, and keep the accepted sample count.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection never reports success after a payload write failure", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::Session::ioLoop;
    MagAOX::app::dlCollection::Session::canStart;
    #endif
    // clang-format on
    auto writer = []( const auto &path, const void *data, size_t size )
    {
        if( path.filename().string().find( "images_" ) == 0 )
            throw std::runtime_error( "injected full disk" );
        writeFile( path, data, size );
    };
    Fixture f( writer, 1, 1 );
    f.m_settings.m_skip = 0;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    f.frame( 1 );
    f.frame( 2 );
    REQUIRE( f.wait( "failed" ) );
    REQUIRE( f.m_command == 0 );
    REQUIRE( f.m_session.progress().m_saved == 1 );
    REQUIRE( f.m_session.progress().m_written == 0 );
    REQUIRE_FALSE( f.m_session.start( f.m_settings ) );
    REQUIRE_FALSE( std::filesystem::exists( std::filesystem::path( f.m_session.progress().m_directory ) /
                                            "complete_dataset_0.conf" ) );
}

/// Collision protection preserves existing bytes, and crop/size validation rejects unsafe input.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection checks file collisions and geometry", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::writeFile;
    MagAOX::app::dlCollection::validate;
    #endif
    // clang-format on
    Fixture    f;
    const auto path  = f.m_root / "existing.bin";
    float      first = 42, next = 84;
    writeFile( path, &first, sizeof( first ) );
    REQUIRE_THROWS( writeFile( path, &next, sizeof( next ) ) );
    REQUIRE( readFloats( path, 1 )[0] == 42.F );
    f.m_settings.m_x[1] = 3;
    REQUIRE_THROWS( validate( f.m_settings ) );
    REQUIRE_THROWS( product( std::numeric_limits<size_t>::max(), 2 ) );
}

/// Only unambiguous, settled switch selections can label an acquisition.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection validates observing-position readbacks", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::PresetReadback::update;
    MagAOX::app::dlCollection::PresetReadback::valid;
    MagAOX::app::dlCollection::PresetReadback::updateState;
    #endif
    // clang-format on
    PresetReadback readback;
    REQUIRE_FALSE( readback.valid( std::chrono::steady_clock::now() ) );
    readback.update( preset( "ND2" ) );
    REQUIRE_FALSE( readback.valid( std::chrono::steady_clock::now() ) );
    readback.updateState( controllerState( "READY" ) );
    for( const auto state : { pcf::IndiProperty::Idle, pcf::IndiProperty::Ok } )
    {
        readback.update( preset( "ND2", state ) );
        REQUIRE( readback.m_value == "ND2" );
        REQUIRE( readback.valid( std::chrono::steady_clock::now() ) );
        REQUIRE( readback.m_time > 0 );
    }
    for( const auto state :
         { pcf::IndiProperty::Busy, pcf::IndiProperty::Alert, pcf::IndiProperty::UnknownPropertyState } )
    {
        readback.update( preset( "ND2", state ) );
        REQUIRE( readback.m_value.empty() );
    }
    auto ambiguous = preset( "ND1" );
    ambiguous.add( pcf::IndiElement( "ND2" ) );
    ambiguous["ND2"].setSwitchState( pcf::IndiElement::On );
    readback.update( ambiguous );
    REQUIRE( readback.m_value.empty() );
    ambiguous["ND1"].setSwitchState( pcf::IndiElement::Off );
    ambiguous["ND2"].setSwitchState( pcf::IndiElement::Off );
    readback.update( ambiguous );
    REQUIRE( readback.m_value.empty() );
    ambiguous["ND1"].setSwitchState( pcf::IndiElement::UnknownSwitchState );
    readback.update( ambiguous );
    REQUIRE( readback.m_value.empty() );
    for( const auto &bad : { "ND2\nflipwfsf=in", "ND2\r", "ND=2" } )
    {
        readback.update( preset( bad ) );
        REQUIRE( readback.m_value.empty() );
    }
    pcf::IndiProperty text( pcf::IndiProperty::Text );
    readback.update( text );
    REQUIRE( readback.m_value.empty() );
    readback.update( preset( "ND2" ) );
    REQUIRE_FALSE( readback.valid( readback.m_received + std::chrono::seconds( 5 ) ) );
    REQUIRE_FALSE( readback.valid( readback.m_received - std::chrono::seconds( 1 ) ) );
    readback.update( preset( "ND2" ) );
    for( const auto state : { "NOTCONNECTED", "ERROR", "HOMING", "NODEVICE", "UNKNOWN" } )
    {
        readback.updateState( controllerState( state ) );
        REQUIRE_FALSE( readback.valid( std::chrono::steady_clock::now() ) );
    }
    for( const auto state : { "READY", "OPERATING", "POWEROFF" } )
    {
        readback.updateState( controllerState( state ) );
        REQUIRE( readback.valid( std::chrono::steady_clock::now() ) );
    }
    readback.m_stateReceived -= std::chrono::seconds( 6 );
    REQUIRE_FALSE( readback.valid( std::chrono::steady_clock::now() ) );
    readback.m_received      = {};
    readback.m_stateReceived = {};
    REQUIRE_FALSE( readback.valid( std::chrono::steady_clock::time_point( std::chrono::seconds( 1 ) ) ) );
}

/// Reject missing, malformed, nonpositive, changing and stale camera readbacks.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection validates camera setting readbacks", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::CameraReadback::update;
    MagAOX::app::dlCollection::CameraReadback::valid;
    #endif
    // clang-format on
    CameraReadback value;
    REQUIRE_FALSE( value.valid( std::chrono::steady_clock::now() ) );
    for( const auto state : { pcf::IndiProperty::Idle, pcf::IndiProperty::Ok } )
    {
        auto setting = cameraSetting( "2000.125" );
        setting.setState( state );
        value.update( setting );
        REQUIRE( value.valid( std::chrono::steady_clock::now() ) );
        REQUIRE( value.m_value == 2000.125 );
        REQUIRE( value.m_time > 0 );
    }
    REQUIRE_FALSE( value.valid( value.m_received - std::chrono::seconds( 1 ) ) );
    REQUIRE_FALSE( value.valid( value.m_received + std::chrono::seconds( 5 ) ) );
    for( const auto *bad : { "", "broken", "100junk", "nan", "inf", "1e999", "0", "-1" } )
    {
        value.update( cameraSetting( bad ) );
        REQUIRE_FALSE( value.valid( std::chrono::steady_clock::now() ) );
    }
    for( const auto state :
         { pcf::IndiProperty::Busy, pcf::IndiProperty::Alert, pcf::IndiProperty::UnknownPropertyState } )
    {
        auto setting = cameraSetting( "100" );
        setting.setState( state );
        value.update( setting );
        REQUIRE_FALSE( value.valid( std::chrono::steady_clock::now() ) );
    }
    auto targetOnly = cameraSetting( "100" );
    targetOnly.remove( "current" );
    targetOnly.add( pcf::IndiElement( "target", 200 ) );
    value.update( targetOnly );
    REQUIRE_FALSE( value.valid( std::chrono::steady_clock::now() ) );
    pcf::IndiProperty wrongType( pcf::IndiProperty::Text );
    wrongType.setState( pcf::IndiProperty::Ok );
    wrongType.add( pcf::IndiElement( "current", 100 ) );
    value.update( wrongType );
    REQUIRE_FALSE( value.valid( std::chrono::steady_clock::now() ) );
}

/// Freeze automatic labels and camera settings in run.conf for later training.
/** \ingroup dlDataCollection_unit_test */
TEST_CASE( "dlDataCollection saves automatically observed ND and flipper positions", "[dlDataCollection]" )
{
    // clang-format off
    #ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlCollection::observationContext;
    MagAOX::app::dlCollection::Session::start;
    #endif
    // clang-format on
    PresetReadback nd, flip;
    CameraReadback fps, emgain;
    fps.update( cameraSetting( "2000.125" ) );
    emgain.update( cameraSetting( "100" ) );
    nd.updateState( controllerState( "READY" ) );
    flip.updateState( controllerState( "OPERATING" ) );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
    nd.update( preset( "ND2" ) );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
    flip.update( preset( "out" ) );
    const auto snapshot = observationContext( nd, flip, fps, emgain );
    REQUIRE( snapshot.find( "nd=ND2\nflipwfsf=out\n" ) == 0 );
    REQUIRE( snapshot.find( "fwtelsim_property=filterName\n" ) != std::string::npos );
    REQUIRE( snapshot.find( "camwfs_fps=2000.125\ncamwfs_emgain=100\n" ) != std::string::npos );
    REQUIRE( snapshot.find( "camwfs_fps_readback_ns=" + std::to_string( fps.m_time ) ) != std::string::npos );
    Fixture f;
    f.m_settings.m_context = snapshot;
    REQUIRE( f.m_session.start( f.m_settings ) );
    REQUIRE( f.wait( "collecting" ) );
    nd.update( preset( "ND3" ) );
    flip.update( preset( "in" ) );
    fps.update( cameraSetting( "1000" ) );
    emgain.update( cameraSetting( "200" ) );
    const auto        run = std::filesystem::path( f.m_session.progress().m_directory );
    std::ifstream     input( run / "run.conf" );
    const std::string saved{ std::istreambuf_iterator<char>( input ), std::istreambuf_iterator<char>() };
    REQUIRE( saved.find( "\n" + snapshot ) != std::string::npos );
    REQUIRE( observationContext( nd, flip, fps, emgain ).find( "nd=ND3\nflipwfsf=in\n" ) == 0 );
    REQUIRE( saved.find( "camwfs_fps=1000\n" ) == std::string::npos );
    f.m_session.abort( "observing configuration changed during collection" );
    REQUIRE( f.wait( "failed" ) );
    REQUIRE( f.m_command == 0 );
    flip.update( preset( "unknown" ) );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
    flip.update( preset( "in", pcf::IndiProperty::Busy ) );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
    flip.update( preset( "in" ) );
    nd.m_received -= std::chrono::seconds( 6 );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
    nd.update( preset( "ND3" ) );
    fps.m_received -= std::chrono::seconds( 6 );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
    fps.update( cameraSetting( "1000" ) );
    emgain.update( cameraSetting( "0" ) );
    REQUIRE_THROWS( observationContext( nd, flip, fps, emgain ) );
}

} // namespace dlDataCollectionTest
} // namespace libXWCTest
