/** \file alignLoop_test.cpp
 * \brief Catch2 tests for the alignLoop app.
 *
 * \ingroup alignLoop_files
 */

#include "../../../tests/testXWC.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <sys/stat.h>

#include "../alignLoop.hpp"

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
    return 0;
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
    Harness();

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
};

Harness::Harness()
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

    Harness app;
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

    Harness app;
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

} // namespace alignLoopTest

} // namespace libXWCTest
