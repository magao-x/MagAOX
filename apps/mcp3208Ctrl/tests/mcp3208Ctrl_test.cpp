/** \file mcp3208Ctrl_test.cpp
 * \brief Catch2 tests for the mcp3208Ctrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup mcp3208Ctrl_files
 */

#include "../../../tests/testXWC.hpp"

#include <vector>

#define protected public
#include "../mcp3208Ctrl.hpp"
#undef protected

namespace
{

/// Stub state for MCP3208 hardware access during unit tests.
struct mcp3208StubState
{
    std::vector<unsigned short> m_channelValues;     ///< Values returned for each channel read.
    std::vector<int>            m_readOrder;         ///< Order in which channels were sampled.
    int                         m_connectCalls{ 0 }; ///< Number of stubbed connect calls.
};

/// Access the shared MCP3208 stub state.
mcp3208StubState &stubState()
{
    static mcp3208StubState state;
    return state;
}

/// Reset the shared MCP3208 stub state.
void resetStubState()
{
    stubState().m_channelValues.clear();
    stubState().m_readOrder.clear();
    stubState().m_connectCalls = 0;
}

} // namespace

namespace MCP3208Lib
{

MCP3208::MCP3208( const int dev, const int channel, const int baud, const int flags ) noexcept
    : _handle( -1 ), _dev( dev ), _channel( channel ), _baud( baud ), _flags( flags )
{
}

MCP3208::~MCP3208()
{
}

void MCP3208::connect()
{
    ++stubState().m_connectCalls;
}

void MCP3208::disconnect()
{
}

unsigned short MCP3208::read( const std::uint8_t channel, const Mode ) const
{
    stubState().m_readOrder.push_back( static_cast<int>( channel ) );

    if( static_cast<size_t>( channel ) < stubState().m_channelValues.size() )
    {
        return stubState().m_channelValues[static_cast<size_t>( channel )];
    }

    return 0;
}

} // namespace MCP3208Lib

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup mcp3208Ctrl_unit_test mcp3208Ctrl Unit Tests
 * \brief Unit tests for the mcp3208Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `mcp3208Ctrl` unit tests.
/** \ingroup mcp3208Ctrl_unit_test
 */
namespace mcp3208CtrlTest
{

namespace
{

/// Test harness exposing small setup helpers for `mcp3208Ctrl`.
class mcp3208Ctrl_test : public mcp3208Ctrl
{
  public:
    /// Construct the test harness with a stable config name.
    mcp3208Ctrl_test()
    {
        m_configName = "mcp3208Ctrl_test";
    }

    /// Initialize the local fps INDI property used by the callback tests.
    void setupFpsProperty()
    {
        m_indiP_fps = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_fps.setName( "fps" );
        m_indiP_fps.add( pcf::IndiElement( "current" ) );
        m_indiP_fps["current"].setValue( m_fps );
        m_indiP_fps.add( pcf::IndiElement( "target" ) );
        m_indiP_fps["target"].setValue( m_fps );
    }

    /// Initialize the external fps INDI property used by the callback tests.
    void setupFpsSourceProperty()
    {
        m_fpsDevice   = "fpsdev";
        m_fpsProperty = "fps";
        m_fpsElement  = "current";

        m_indiP_fpsSource = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_fpsSource.setDevice( m_fpsDevice );
        m_indiP_fpsSource.setName( m_fpsProperty );
    }

    /// Build an INDI property update for the local fps callback.
    pcf::IndiProperty makeFpsUpdate( const double target /**< [in] the requested fps */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setName( "fps" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].setValue( target );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].setValue( target );
        return ip;
    }

    /// Build an INDI property update for the external fps-source callback.
    pcf::IndiProperty makeFpsSourceUpdate( const double current /**< [in] the reported fps */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_fpsDevice );
        ip.setName( m_fpsProperty );
        ip.add( pcf::IndiElement( m_fpsElement ) );
        ip[m_fpsElement].setValue( current );
        return ip;
    }
};

} // namespace

/// Preserve Doxygen links for the real `mcp3208Ctrl` APIs exercised by the tests.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl Doxygen references are preserved", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsProperty();
    app.setupFpsSourceProperty();

    XWCTEST_DOXYGEN_REF( app.loadConfigImpl( app.config ) );
    XWCTEST_DOXYGEN_REF( app.configureAcquisition() );
    XWCTEST_DOXYGEN_REF( app.fps() );
    XWCTEST_DOXYGEN_REF( app.startAcquisition() );
    XWCTEST_DOXYGEN_REF( app.acquireAndCheckValid() );
    XWCTEST_DOXYGEN_REF( app.loadImageIntoStream( nullptr ) );
    XWCTEST_DOXYGEN_REF( app.reconfig() );
    XWCTEST_DOXYGEN_REF( app.synchroStreamStale() );
    XWCTEST_DOXYGEN_REF( app.checkRecordTimes() );
    XWCTEST_DOXYGEN_REF( app.recordTelem( nullptr ) );
    XWCTEST_DOXYGEN_REF( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( 1000.0 ) ) );
    XWCTEST_DOXYGEN_REF( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 1000.0 ) ) );

    SUCCEED();
}

/// Verify synchronized-acquisition defaults load from configuration.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl configuration defaults load synchronized settings", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test.conf", { "none" }, { "nada" }, { "0" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test.conf" );

    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_synchroShmimName.empty() );
    REQUIRE( app.m_synchroPostDelay == 0 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
}

/// Verify synchronized-acquisition overrides load from configuration.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl configuration overrides load synchronized settings", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test_override.conf",
                              { "synchro", "synchro", "accel" },
                              { "shmimName", "postDelay", "numChannels" },
                              { "camwfs_sync", "17", "3" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test_override.conf" );

    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_synchroShmimName == "camwfs_sync" );
    REQUIRE( app.m_synchroPostDelay == 17 );
    REQUIRE( app.m_numChannels == 3 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 17000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 17000.0f ) );
}

/// Verify the user fps callback still updates cadence metadata.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl fps callback updates trigger metadata", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsProperty();

    REQUIRE( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( 500.0 ) ) == 0 );
    REQUIRE( app.m_fps == Approx( 500.0f ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 500.0f ) );
    REQUIRE( app.nano_sec_target == Approx( 1e9f / 500.0f ) );
}

/// Verify the external fps source callback still updates cadence metadata.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl fps source callback updates trigger metadata", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsSourceProperty();

    REQUIRE( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 250.0 ) ) == 0 );
    REQUIRE( app.m_fps == Approx( 250.0f ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 250.0f ) );
    REQUIRE( app.nano_sec_target == Approx( 1e9f / 250.0f ) );
}

/// Verify timer-driven acquisition configures the published frame geometry.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timer mode configureAcquisition sizes the output frame", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels = 3;

    REQUIRE( app.configureAcquisition() == 0 );
    REQUIRE( app.m_values.size() == 3 );
    REQUIRE( app.m_width == 3 );
    REQUIRE( app.m_height == 1 );
    REQUIRE( app.m_dataType == _DATATYPE_UINT16 );
}

/// Verify timer-driven acquisition still reads one frame of ADC values.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timer mode reads configured channels", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 11, 22, 33 };

    app.m_numChannels = 3;
    app.m_values.assign( 3, 0 );
    app.m_gain       = 0;
    app.m_trigger    = 0;
    app.m_time_start = std::chrono::high_resolution_clock::now();

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 11, 22, 33 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1, 2 } ) );
}

/// Verify the current MCP3208 values are copied into the output image buffer.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl loadImageIntoStream copies the current values", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test      app;
    std::vector<uint16_t> dest( 3, 0 );

    app.m_values = { 5, 6, 7 };

    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( dest == std::vector<uint16_t>( { 5, 6, 7 } ) );
}

/// Verify synchronized acquisition performs one ADC sweep per semaphore wake.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized mode reads on semaphore wake", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 101, 202, 303 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 3;
    app.m_values.assign( 3, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelay       = 0;
    app.m_synchroDelayTarget = 0;
    app.m_gain               = 0;

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 101, 202, 303 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1, 2 } ) );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized timeout requests reconfiguration when the trigger stream is stale.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized timeout requests reconfig for a stale stream", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName  = "camwfs_sync";
    app.m_values            = { 7, 8 };
    app.m_synchroSemaphore  = &semaphore;
    app.m_synchroStreamOpen = false;

    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.m_reconfig == true );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 7, 8 } ) );
    REQUIRE( stubState().m_readOrder.empty() );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify stale-stream detection notices a missing synchronization stream backing file.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized stale helper detects missing streams", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    IMAGE_METADATA   metadata{};

    app.m_synchroShmimName  = "mcp3208Ctrl_unit_test_missing_stream";
    app.m_synchroStreamOpen = true;
    metadata.sem            = SEMAPHORE_MAXVAL;
    app.m_synchroStream.md  = &metadata;

    REQUIRE( app.synchroStreamStale() == true );
}

/// Verify `reconfig()` clears cached synchronization state.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl reconfig clears cached synchronization state", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_synchroSemaphore       = reinterpret_cast<sem_t *>( 0x1 );
    app.m_synchroSemaphoreNumber = 7;
    app.m_synchroStreamInode     = 1234;
    app.m_synchroStreamOpen      = false;

    REQUIRE( app.reconfig() == 0 );
    REQUIRE( app.m_synchroSemaphore == nullptr );
    REQUIRE( app.m_synchroSemaphoreNumber == 5 );
    REQUIRE( app.m_synchroStreamInode == 0 );
    REQUIRE( app.m_synchroStreamOpen == false );
}

} // namespace mcp3208CtrlTest
} // namespace libXWCTest
