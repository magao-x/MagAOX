/** \file alignLoop_test.cpp
 * \brief Offline alignLoop control and telemetry tests.
 *
 * \ingroup alignLoop_files
 */

#include "../../../tests/testXWC.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <sys/stat.h>

#include "../alignLoop.hpp"
#include "../../../libMagAOX/logger/generated/logMemberAccessor.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup alignLoop_unit_test alignLoop Unit Tests
 * \brief Unit tests for the alignLoop application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `alignLoop` unit tests.
/** \ingroup alignLoop_unit_test
 */
namespace alignLoopTest
{

/// \cond TEST_HARNESS
/// Own a private telemetry directory that outlives the test app and its writer.
class TelemetryDirectory
{
  public:
    /// Create the private scratch directory.
    TelemetryDirectory();

    /// Remove the test telemetry files and directory.
    ~TelemetryDirectory();

    std::filesystem::path m_path; ///< Directory holding this fixture's .bintel files.
};

TelemetryDirectory::TelemetryDirectory()
{
    char root[] = "/tmp/alignLoop_test_XXXXXX";
    REQUIRE( mkdtemp( root ) != nullptr );
    m_path = root;
}

TelemetryDirectory::~TelemetryDirectory()
{
    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}

/// Capture publications and stage commands in memory with an unactivated local INDI driver.
class LocalDriver : public indiDriver<MagAOXApp<true>>
{
  public:
    /// Use fixture-local file descriptors and disconnect the unused input after construction.
    LocalDriver( MagAOXApp<true> *app, /**< [in] owner of the local driver paths */
                 int              fd /**< [in] fixture descriptor for identifying the driver's input */ );

    /// Record commands before the production driver would open an INDI client.
    int sendNewProperty( const pcf::IndiProperty &prop /**< [in] stage target to record */ ) override;

    /// Capture serialized property publications.
    void sendXml( const std::string &xml /**< [in] INDI XML emitted by the app */ ) const override;

    std::vector<pcf::IndiProperty> m_commands; ///< Stage targets intercepted during frame processing.

    std::string m_failedDevice; ///< Optional device whose next-property sends fail locally.

    mutable std::string m_publications; ///< Property XML emitted without transport.
};

LocalDriver::LocalDriver( MagAOXApp<true> *app, int fd ) : indiDriver( app, "align-loop-test", "0", "1.7" )
{
    REQUIRE( good() );

    struct stat input{};
    REQUIRE( fstat( fd, &input ) == 0 );

    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        const int   candidate = std::stoi( entry.path().filename() );
        struct stat descriptor{};

        if( candidate != fd && fstat( candidate, &descriptor ) == 0 && descriptor.st_dev == input.st_dev &&
            descriptor.st_ino == input.st_ino )
        {
            close( candidate );
            setInputFd( -1 );
            enableResponseMode( true );
            return;
        }
    }

    FAIL( "The inactive driver input descriptor was not found" );
}

int LocalDriver::sendNewProperty( const pcf::IndiProperty &prop )
{
    m_commands.push_back( prop );
    return prop.getDevice() == m_failedDevice ? -1 : 0;
}

void LocalDriver::sendXml( const std::string &xml ) const
{
    m_publications += xml;
}

/// Exercise normal startup and callbacks with synthetic frames and local INDI properties.
class Harness : public alignLoop
{
  public:
    /// Configure two simulated stages and an idle stream monitor.
    explicit Harness( const std::filesystem::path &telemetryPath /**< [in] private telemetry directory */ );

    /// Stop the idle monitor before tearing down the local app.
    ~Harness();

    /// Construct a request for the supplied local property.
    pcf::IndiProperty offsetRequest( const pcf::IndiProperty &prop, /**< [in] axis property to address */
                                     const char              *raw,  /**< [in] requested numeric text or null */
                                     const char              *element = "target" /**< [in] element to write */ );

    /// Provide current readbacks for both simulated stages.
    void stageReadbacks();

    /// Feed a two-axis synthetic frame through the production monitor callback.
    int frame( float x, /**< [in] first measurement */
               float y /**< [in] second measurement */ );

    /// Access local publications and captured commands.
    LocalDriver &driver();

    /// Exercise the production periodic telemetry check.
    using alignLoop::checkRecordTimes;

  private:
    /// Metadata for the synthetic stream frames consumed by the app.
    IMAGE_METADATA m_metadata{};
};

Harness::Harness( const std::filesystem::path &telemetryPath )
{
    m_configName     = "align-loop-test";
    m_ctrlDevices    = { "local-stage-0", "local-stage-1" };
    m_ctrlProperties = { "position", "position" };
    m_ctrlCurrents   = { "current", "current" };
    m_ctrlTargets    = { "target", "target" };
    m_defaultGains   = { 0.5f, 0.25f };
    m_ggain          = 0.5f;
    m_smThreadPrio   = 0;
    m_width          = 2;
    m_height         = 1;

    m_tel.logPath( telemetryPath.string() );
    m_tel.logName( m_configName );
    m_tel.logExt( "bintel" );
    m_tel.m_logLevel                = logPrio::LOG_TELEM;
    m_metadata.atime                = { 123, 456 };
    shmimMonitorT::m_imageStream.md = &m_metadata;

    const auto closeInput = []( FILE *stream )
    {
        fclose( stream );
    };
    std::unique_ptr<FILE, decltype( closeInput )> input( tmpfile(), closeInput );
    REQUIRE( input );

    const int fd     = fileno( input.get() );
    m_driverInName   = "/proc/self/fd/" + std::to_string( fd );
    m_driverOutName  = "/dev/null";
    m_driverCtrlName = "/dev/null";
    m_indiDriver     = new LocalDriver( this, fd );
}

Harness::~Harness()
{
    m_shutdown = 1;
    appShutdown();
}

pcf::IndiProperty Harness::offsetRequest( const pcf::IndiProperty &prop, const char *raw, const char *element )
{
    pcf::IndiProperty request( pcf::IndiProperty::Number );
    request.setDevice( prop.getDevice() );
    request.setName( prop.getName() );

    if( raw )
    {
        request.add( pcf::IndiElement( element, raw ) );
    }

    return request;
}

void Harness::stageReadbacks()
{
    for( size_t axis = 0; axis < m_ctrlDevices.size(); ++axis )
    {
        pcf::IndiProperty readback( pcf::IndiProperty::Number );
        readback.setDevice( m_ctrlDevices[axis] );
        readback.setName( m_ctrlProperties[axis] );
        readback.add( pcf::IndiElement( "current", axis == 0 ? 10.0f : 20.0f ) );
        handleSetProperty( readback );
    }
}

int Harness::frame( float x, float y )
{
    float measurements[]{ x, y };
    ++m_metadata.cnt0;
    return processImage( measurements, MagAOX::app::dev::shmimT() );
}

LocalDriver &Harness::driver()
{
    return *static_cast<LocalDriver *>( m_indiDriver );
}
/// \endcond

/// Verify INDI defaults, validation and independent axis writes in both loop states.
/**
 * \ingroup alignLoop_unit_test
 */
TEST_CASE( "alignLoop offsets accept valid values while open or closed", "[alignLoop]" )
{
    // clang-format off
    #ifdef ALIGNLOOP_TEST_DOXYGEN_REF
    alignLoop::appStartup;
    alignLoop::newCallBack_m_indiP_offset0;
    alignLoop::newCallBack_m_indiP_offset1;
    alignLoop::toggleLoop;
    #endif
    // clang-format on

    TelemetryDirectory directory;
    Harness            app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    CHECK( app.m_indiP_offset0.getName() == "offset0" );
    CHECK( app.m_indiP_offset1.getName() == "offset1" );

    for( const auto *prop : { &app.m_indiP_offset0, &app.m_indiP_offset1 } )
    {
        CHECK( prop->getPerm() == pcf::IndiProperty::ReadWrite );
        CHECK( ( *prop )["current"].get<float>() == 0 );
        CHECK( ( *prop )["target"].get<float>() == 0 );
    }

    app.handleNewProperty( app.offsetRequest( app.m_indiP_offset0, "0.5" ) );
    CHECK( app.m_indiP_offset0["current"].get<float>() == 0.5f );
    CHECK( app.m_indiP_offset0["target"].get<float>() == 0.5f );
    CHECK( app.m_indiP_offset1["current"].get<float>() == 0 );

    app.handleNewProperty( app.offsetRequest( app.m_indiP_offset1, "-0.2", "current" ) );
    CHECK( app.m_indiP_offset1["current"].get<float>() == -0.2f );
    CHECK( app.m_indiP_offset1["target"].get<float>() == -0.2f );
    CHECK( app.driver().m_publications.find( "name=\"offset1\"" ) != std::string::npos );

    for( const char *value : { "", "0.9px", "nan", "inf", "1e16" } )
    {
        CHECK( app.newCallBack_m_indiP_offset0( app.offsetRequest( app.m_indiP_offset0, value ) ) < 0 );
    }

    CHECK( app.newCallBack_m_indiP_offset0( app.offsetRequest( app.m_indiP_offset0, nullptr ) ) < 0 );
    CHECK( app.newCallBack_m_indiP_offset0( app.offsetRequest( app.m_indiP_offset1, "0.9" ) ) < 0 );
    CHECK( app.m_indiP_offset0["target"].get<float>() == 0.5f );

    REQUIRE( app.toggleLoop( true ) == 0 );
    app.handleNewProperty( app.offsetRequest( app.m_indiP_offset0, "0.9" ) );
    CHECK( app.m_indiP_offset0["current"].get<float>() == 0.9f );
    CHECK( app.m_indiP_offset0["target"].get<float>() == 0.9f );
    CHECK( app.m_indiP_offset1["current"].get<float>() == -0.2f );
    CHECK( app.m_indiP_offset1["target"].get<float>() == -0.2f );
    CHECK( app.driver().m_commands.empty() );
}

/// Verify that publication and real stage command generation use the same corrected frame.
/**
 * \ingroup alignLoop_unit_test
 */
TEST_CASE( "alignLoop publishes and controls on offset-corrected residuals", "[alignLoop]" )
{
    // clang-format off
    #ifdef ALIGNLOOP_TEST_DOXYGEN_REF
    alignLoop::allocate;
    alignLoop::processImage;
    alignLoop::sendCommands;
    #endif
    // clang-format on

    TelemetryDirectory directory;
    Harness            app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.allocate( MagAOX::app::dev::shmimT() ) == 0 );

    SECTION( "Zero offsets preserve the original residual and stage correction" )
    {
        app.stageReadbacks();
        REQUIRE( app.toggleLoop( true ) == 0 );
        REQUIRE( app.frame( 0.25f, -0.1f ) == 0 );
    }

    SECTION( "An open loop publishes corrected deltas before controller readback" )
    {
        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset0, "0.5" ) );
        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset1, "-0.2" ) );
        REQUIRE( app.frame( 0.75f, -0.30f ) == 0 );
        CHECK( app.m_indiP_deltas["delta0"].get<float>() == Approx( 0.25 ) );
        CHECK( app.m_indiP_deltas["delta1"].get<float>() == Approx( -0.10 ) );
        CHECK( app.driver().m_commands.empty() );
        return;
    }

    SECTION( "Live offset writes reach the closed loop on its next frame" )
    {
        REQUIRE( app.frame( 0.5f, -0.2f ) == 0 );
        app.stageReadbacks();
        REQUIRE( app.toggleLoop( true ) == 0 );

        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset0, "0.5" ) );
        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset1, "-0.2" ) );
        CHECK( app.driver().m_commands.empty() );

        REQUIRE( app.frame( 0.5f, -0.2f ) == 0 );
        CHECK( app.m_indiP_deltas["delta0"].get<float>() == 0 );
        CHECK( app.m_indiP_deltas["delta1"].get<float>() == 0 );
        CHECK( app.driver().m_commands.empty() );

        REQUIRE( app.frame( 0.75f, -0.30f ) == 0 );
    }

    CHECK( app.m_indiP_deltas["delta0"].get<float>() == Approx( 0.25 ) );
    CHECK( app.m_indiP_deltas["delta1"].get<float>() == Approx( -0.10 ) );
    REQUIRE( app.driver().m_commands.size() == 2 );
    CHECK( app.driver().m_commands[0].getDevice() == "local-stage-0" );
    CHECK( app.driver().m_commands[1].getDevice() == "local-stage-1" );
    CHECK( app.driver().m_commands[0]["target"].get<float>() == Approx( 10.0115875 ) );
    CHECK( app.driver().m_commands[1]["target"].get<float>() == Approx( 19.9664375 ) );
}

/// Verify ordered state records, frame identity, residuals and command outcomes in the archive.
/** \ingroup alignLoop_unit_test
 */
TEST_CASE( "alignLoop telemetry records state and the result of each processed frame", "[alignLoop]" )
{
    // clang-format off
    #ifdef ALIGNLOOP_TEST_DOXYGEN_REF
    alignLoop::checkRecordTimes;
    alignLoop::recordAlignment;
    alignLoop::processImage;
    alignLoop::sendCommands;
    #endif
    // clang-format on

    TelemetryDirectory directory;

    {
        Harness app( directory.m_path );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.allocate( MagAOX::app::dev::shmimT() ) == 0 );
        REQUIRE( app.frame( 0.5f, -0.2f ) == 0 );
        app.stageReadbacks();
        REQUIRE( app.toggleLoop( true ) == 0 );
        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset0, "0.5" ) );
        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset1, "-0.2" ) );
        app.handleNewProperty( app.offsetRequest( app.m_indiP_offset1, "-0.2" ) );
        REQUIRE( app.frame( 0.5f, -0.2f ) == 0 );

        auto gain = app.offsetRequest( app.m_indiP_ggain, "0.25" );
        app.handleNewProperty( gain );
        REQUIRE( app.frame( 0.75f, -0.30f ) == 0 );
        app.driver().m_failedDevice = "local-stage-1";
        REQUIRE( app.frame( 0.75f, -0.30f ) == 0 );
        REQUIRE( app.toggleLoop( false ) == 0 );

        telem_alignloop::lastRecord = { 0, 0 };
        REQUIRE( app.checkRecordTimes() == 0 );
    } // Destroy the app and drain the writer before reading the archive.

    std::vector<std::filesystem::path> paths;
    for( const auto &file : std::filesystem::recursive_directory_iterator( directory.m_path ) )
    {
        if( file.path().extension() == ".bintel" )
        {
            paths.push_back( file.path() );
        }
    }

    std::sort( paths.begin(), paths.end() );
    std::vector<flatlogs::bufferPtrT> records;
    for( const auto &path : paths )
    {
        std::ifstream     input( path, std::ios::binary );
        std::vector<char> bytes( ( std::istreambuf_iterator<char>( input ) ), {} );
        size_t            offset = 0;

        while( offset < bytes.size() )
        {
            REQUIRE( bytes.size() - offset >= flatlogs::logHeader::minHeadSize );
            auto size = flatlogs::logHeader::totalSize( bytes.data() + offset );
            REQUIRE( size <= bytes.size() - offset );
            flatlogs::bufferPtrT record( static_cast<char *>( ::operator new( size ) ) );
            std::memcpy( record.get(), bytes.data() + offset, size );
            REQUIRE( flatlogs::logHeader::eventCode( record ) == telem_alignloop::eventCode );
            REQUIRE( logVerify( telem_alignloop::eventCode, record, flatlogs::logHeader::msgLen( record ) ) );
            records.push_back( record );
            offset += size;
        }
    }

    REQUIRE( records.size() == 11 );
    const auto record = [&]( size_t index )
    {
        return GetTelem_alignloop_fb( flatlogs::logHeader::messageBuffer( records[index] ) );
    };

    CHECK_FALSE( record( 0 )->closed() );
    CHECK( record( 0 )->gain() == 0.5f );
    CHECK( record( 0 )->axis_gains()->axis0() == 0.5f );
    CHECK( record( 0 )->axis_gains()->axis1() == 0.25f );
    CHECK( record( 0 )->sample() == nullptr );
    REQUIRE( record( 1 )->sample() );
    CHECK( record( 1 )->sample()->frame() == 1 );
    CHECK( record( 1 )->targets() == nullptr );
    CHECK( record( 2 )->closed() );
    CHECK( record( 2 )->sample() == nullptr );
    CHECK( record( 3 )->offsets()->axis0() == 0.5f );
    CHECK( record( 3 )->offsets()->axis1() == 0 );
    CHECK( record( 4 )->offsets()->axis1() == -0.2f );

    REQUIRE( record( 5 )->sample() );
    CHECK( record( 5 )->sample()->measurement().axis0() == 0.5f );
    CHECK( record( 5 )->sample()->measurement().axis1() == -0.2f );
    CHECK( record( 5 )->sample()->residual().axis0() == 0 );
    CHECK( record( 5 )->sample()->residual().axis1() == 0 );
    CHECK( record( 5 )->targets() == nullptr );
    CHECK( record( 6 )->gain() == 0.25f );

    REQUIRE( record( 7 )->sample() );
    CHECK( record( 7 )->sample()->frame() == 3 );
    CHECK( record( 7 )->sample()->acq_sec() == 123 );
    CHECK( record( 7 )->sample()->acq_nsec() == 456 );
    CHECK( record( 7 )->sample()->residual().axis0() == Approx( 0.25 ) );
    CHECK( record( 7 )->sample()->residual().axis1() == Approx( -0.1 ) );
    REQUIRE( record( 7 )->targets() );
    CHECK( record( 7 )->targets()->axis0() == Approx( 10.00579375 ) );
    CHECK( record( 7 )->targets()->axis1() == Approx( 19.98321875 ) );
    CHECK( record( 7 )->sent_mask() == 3 );
    REQUIRE( record( 8 )->targets() );
    CHECK( record( 8 )->sent_mask() == 1 );
    CHECK_FALSE( record( 9 )->closed() );
    CHECK( record( 9 )->sample() == nullptr );
    CHECK_FALSE( record( 10 )->closed() );
    CHECK( record( 10 )->sample() == nullptr );

    std::ostringstream text;
    logStdFormat( text, records[7] );
    CHECK( text.str().find( "frame=3" ) != std::string::npos );

    std::ostringstream json;
    logJsonFormat( json, records[7] );
    CHECK( json.str().find( "\"residual\"" ) != std::string::npos );
    CHECK( json.str().find( "\"sent_mask\": 3" ) != std::string::npos );

    const auto offset1 = logMemberAccessor( telem_alignloop::eventCode, "offset1" );
    REQUIRE( offset1.accessor );
    CHECK( reinterpret_cast<float ( * )( void * )>( offset1.accessor )(
               flatlogs::logHeader::messageBuffer( records[7] ) ) == -0.2f );
}

} // namespace alignLoopTest

} // namespace libXWCTest
