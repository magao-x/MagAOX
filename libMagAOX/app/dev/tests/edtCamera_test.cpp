/** \file edtCamera_test.cpp
 * \brief Catch2 tests for the MagAOX::app::dev::edtCamera device mixin.
 *
 * The mixin is exercised through the harnesses declared in edtCamera_test.hpp, which never
 * connect to INDI. No EDT hardware or EDT library is used. This file defines extern "C"
 * replacements for every pdv_ and edt_ function the mixin calls. The replacements count
 * their calls and return scripted results from one shared edtStubState instance. Each test
 * sets the stub state it needs, so every failure branch in the mixin can be forced.
 *
 * The configuration test writes /tmp/edtCamera_test.conf. The harness sets its config
 * directory to /tmp so relative EDT config paths resolve there.
 *
 * \ingroup app_dev_unit_tests
 */

#include <deque>
#include <string>
#include <vector>

#include "edtCamera_test.hpp"

using namespace MagAOX::app;

namespace edtCamera_tests
{

/// Scripted state for the fake EDT PDV SDK used by these tests.
/** Flags force a specific SDK call to fail. Counters record how often each call ran.
 * Queues supply the results of successive serial wait and read calls.
 */
struct edtStubState
{
    // Controls for pdvConfig().
    bool allocDependentFail{ false }; ///< Force `pdv_alloc_dependent` to return nullptr.
    int  readcfgReturn{ 0 };          ///< Value returned by `pdv_readcfg`.
    bool openChannelFail{ false };    ///< Force `edt_open_channel` to return nullptr.
    int  initcamReturn{ 0 };          ///< Value returned by `pdv_initcam`.
    bool openPdvChannelFail{ false }; ///< Force `pdv_open_channel` to return nullptr.

    int         width{ 640 };
    int         height{ 512 };
    int         depth{ 16 };
    std::string cameraType{ "stub_pdv" };

    int allocDependentCalls{ 0 };
    int readcfgCalls{ 0 };
    int openChannelCalls{ 0 };
    int initcamCalls{ 0 };
    int closeCalls{ 0 };
    int edtCloseCalls{ 0 };
    int multibufCalls{ 0 };
    int lastNumBuffs{ -1 };

    // Controls for pdvStartAcquisition() and pdvAcquire().
    int      startImagesCalls{ 0 };
    int      lastStartNumBuffs{ -1 };
    int      startImageCalls{ 0 };
    unsigned waitTimeSec{ 7 };
    unsigned waitTimeNsec{ 11 };

    // Controls for pdvSerialWriteRead().
    int                      commandResult{ 0 }; ///< Value returned by `pdv_serial_command`.
    std::deque<int>          waitResults;         ///< Popped in order by successive `pdv_serial_wait` calls.
    std::deque<std::string>  readChunks;          ///< Popped in order by successive `pdv_serial_read` calls.
    bool                     waitcharAvailable{ true };
    char                     waitcharValue{ '\n' };
    std::vector<std::string> serialCommands;

    /// Reset all stub state to defaults before a test section.
    void reset()
    {
        *this = edtStubState{};
    }
};

/// The single shared stub state instance that the SDK replacements below read and update.
edtStubState g_edt;

/// Reset the shared stub state before a test section.
void resetStub()
{
    g_edt.reset();
}

/// Queue the flush-read placeholder followed by the given response chunks.
/** `pdvSerialWriteRead` always performs one throw-away `pdv_serial_read` call to flush the
 * channel before sending its command. Every scripted read sequence therefore needs a
 * leading placeholder entry that the flush consumes.
 */
void queueReadChunks( std::initializer_list<std::string> chunks )
{
    g_edt.readChunks.push_back( "" ); // Consumed by the flush read that precedes the command.
    for( auto &c : chunks )
    {
        g_edt.readChunks.push_back( c );
    }
}

} // namespace edtCamera_tests

using namespace edtCamera_tests;

// Replacements for the EDT SDK functions that edtCamera calls. They need C linkage so they
// satisfy the declarations in the SDK headers. Each one records its call in g_edt and returns
// whatever the current stub state says.
extern "C"
{

/// Returns a malloc'd Dependent, or nullptr when allocDependentFail is set.
Dependent *pdv_alloc_dependent()
{
    ++g_edt.allocDependentCalls;
    if( g_edt.allocDependentFail )
    {
        return nullptr;
    }
    return reinterpret_cast<Dependent *>( malloc( sizeof( Dependent ) ) );
}

int pdv_readcfg( const char *configFile, Dependent *dd_p, Edtinfo *edtinfo )
{
    static_cast<void>( configFile );
    static_cast<void>( dd_p );
    static_cast<void>( edtinfo );
    ++g_edt.readcfgCalls;
    return g_edt.readcfgReturn;
}

/// Returns a static EdtDev, or nullptr when openChannelFail is set.
EdtDev *edt_open_channel( const char *deviceName, int unit, int channel )
{
    static_cast<void>( deviceName );
    static_cast<void>( unit );
    static_cast<void>( channel );
    ++g_edt.openChannelCalls;
    if( g_edt.openChannelFail )
    {
        return nullptr;
    }
    static EdtDev device;
    return &device;
}

void edt_perror( char *errstr )
{
    if( errstr != nullptr )
    {
        errstr[0] = '\0';
    }
}

int pdv_initcam( EdtDev     *edt_p,
                 Dependent  *dd_p,
                 int         unit,
                 Edtinfo    *edtinfo,
                 const char *configFile,
                 char       *bitdir,
                 int         pdv_debug )
{
    static_cast<void>( edt_p );
    static_cast<void>( dd_p );
    static_cast<void>( unit );
    static_cast<void>( edtinfo );
    static_cast<void>( configFile );
    static_cast<void>( bitdir );
    static_cast<void>( pdv_debug );
    ++g_edt.initcamCalls;
    return g_edt.initcamReturn;
}

void edt_close( EdtDev *edt_p )
{
    static_cast<void>( edt_p );
    ++g_edt.edtCloseCalls;
}

/// Returns a static PdvDev, or nullptr when openPdvChannelFail is set.
PdvDev *pdv_open_channel( const char *deviceName, int unit, int channel )
{
    static_cast<void>( deviceName );
    static_cast<void>( unit );
    static_cast<void>( channel );
    if( g_edt.openPdvChannelFail )
    {
        return nullptr;
    }
    static PdvDev device;
    return &device;
}

void pdv_close( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
    ++g_edt.closeCalls;
}

void pdv_flush_fifo( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
}

void pdv_serial_read_enable( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
}

int pdv_get_width( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
    return g_edt.width;
}

int pdv_get_height( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
    return g_edt.height;
}

int pdv_get_depth( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
    return g_edt.depth;
}

char *pdv_get_cameratype( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
    return const_cast<char *>( g_edt.cameraType.c_str() );
}

void pdv_multibuf( PdvDev *pdv_p, int numBuffs )
{
    static_cast<void>( pdv_p );
    ++g_edt.multibufCalls;
    g_edt.lastNumBuffs = numBuffs;
}

void pdv_start_images( PdvDev *pdv_p, int numBuffs )
{
    static_cast<void>( pdv_p );
    ++g_edt.startImagesCalls;
    g_edt.lastStartNumBuffs = numBuffs;
}

/// Returns a static dummy image and fills the DMA timestamp from the stub state.
u_char *pdv_wait_last_image_timed( PdvDev *pdv_p, uint dmaTimeStamp[2] )
{
    static_cast<void>( pdv_p );
    static u_char dummyImage[16]{};
    dmaTimeStamp[0] = g_edt.waitTimeSec;
    dmaTimeStamp[1] = g_edt.waitTimeNsec;
    return dummyImage;
}

void pdv_start_image( PdvDev *pdv_p )
{
    static_cast<void>( pdv_p );
    ++g_edt.startImageCalls;
}

/// Copies the next scripted chunk into buf and returns its length. Returns 0 when the script is exhausted.
int pdv_serial_read( PdvDev *pdv_p, char *buf, int size )
{
    static_cast<void>( pdv_p );
    if( buf != nullptr && size > 0 )
    {
        buf[0] = '\0';
    }

    if( g_edt.readChunks.empty() )
    {
        return 0;
    }

    std::string chunk = g_edt.readChunks.front();
    g_edt.readChunks.pop_front();

    if( buf == nullptr || size <= 0 )
    {
        return 0;
    }

    size_t n = chunk.size();
    if( n > static_cast<size_t>( size - 1 ) )
    {
        n = static_cast<size_t>( size - 1 );
    }

    std::copy_n( chunk.data(), n, buf );
    buf[n] = '\0';

    return static_cast<int>( n );
}

/// Records the command string and returns the scripted commandResult.
int pdv_serial_command( PdvDev *pdv_p, const char *command )
{
    static_cast<void>( pdv_p );
    g_edt.serialCommands.emplace_back( command == nullptr ? "" : command );
    return g_edt.commandResult;
}

/// Pops and returns the next scripted wait result. Returns 0, meaning timeout, when the script is exhausted.
int pdv_serial_wait( PdvDev *pdv_p, int timeout, int count )
{
    static_cast<void>( pdv_p );
    static_cast<void>( timeout );
    static_cast<void>( count );

    if( g_edt.waitResults.empty() )
    {
        return 0;
    }

    int r = g_edt.waitResults.front();
    g_edt.waitResults.pop_front();
    return r;
}

/// Reports the scripted terminator character. Returns 0 when waitcharAvailable is false.
int pdv_get_waitchar( PdvDev *pdv_p, u_char *waitc )
{
    static_cast<void>( pdv_p );
    if( waitc != nullptr )
    {
        *waitc = static_cast<u_char>( g_edt.waitcharValue );
    }
    return g_edt.waitcharAvailable ? 1 : 0;
}

} // extern "C"

/// Verify pdvConfig() on success, on every SDK failure branch, and with both config path options.
/** A bad mode name must be rejected. A good mode must fill in the frame geometry and camera
 * type from the SDK. Each forced SDK failure must return a negative value.
 *
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "edtCamera pdvConfig configures the framegrabber or reports each failure", "[edtCamera]" )
{
    resetStub();

    SECTION( "empty mode name is rejected" )
    {
        edtCameraTestApp app;
        std::string      mode = "";
        REQUIRE( app.pdvConfig( mode ) < 0 );
    }

    SECTION( "unknown mode name is rejected" )
    {
        edtCameraTestApp app;
        std::string      mode = "missing";
        REQUIRE( app.pdvConfig( mode ) < 0 );
    }

    SECTION( "successful configuration populates frame geometry" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";

        REQUIRE( app.pdvConfig( mode ) == 0 );
        REQUIRE( app.m_modeName == "science" );
        REQUIRE( app.m_raw_width == g_edt.width );
        REQUIRE( app.m_raw_height == g_edt.height );
        REQUIRE( app.m_raw_depth == g_edt.depth );
        REQUIRE( app.m_cameraType == g_edt.cameraType );
        REQUIRE( app.m_pdv != nullptr );
        REQUIRE( g_edt.multibufCalls > 0 );
        REQUIRE( g_edt.lastNumBuffs == app.m_numBuffs );

        SECTION( "a second configuration call closes the prior pdv handle" )
        {
            app.addMode( "other" );
            std::string mode2 = "other";
            int         closesBefore = g_edt.closeCalls;
            REQUIRE( app.pdvConfig( mode2 ) == 0 );
            REQUIRE( g_edt.closeCalls > closesBefore );
        }
    }

    SECTION( "elevated log level requests EDT debug output" )
    {
        edtCameraTestApp app;
        app.addMode( "dbg" );
        std::string mode = "dbg";
        app.m_log.logLevel( logPrio::LOG_DEBUG );
        REQUIRE( app.pdvConfig( mode ) == 0 );
        app.m_log.logLevel( logPrio::LOG_INFO );
    }

    SECTION( "pdv_alloc_dependent failure is reported" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";
        g_edt.allocDependentFail = true;
        REQUIRE( app.pdvConfig( mode ) < 0 );
    }

    SECTION( "pdv_readcfg failure is reported" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";
        g_edt.readcfgReturn = -1;
        REQUIRE( app.pdvConfig( mode ) < 0 );
    }

    SECTION( "edt_open_channel failure is reported" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";
        g_edt.openChannelFail = true;
        REQUIRE( app.pdvConfig( mode ) < 0 );
    }

    SECTION( "pdv_initcam failure is reported" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";
        g_edt.initcamReturn = -1;
        REQUIRE( app.pdvConfig( mode ) < 0 );
        REQUIRE( g_edt.edtCloseCalls > 0 );
    }

    SECTION( "pdv_open_channel failure is reported" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";
        g_edt.openPdvChannelFail = true;
        REQUIRE( app.pdvConfig( mode ) < 0 );
    }

    SECTION( "an absolute EDT config path is used when configured" )
    {
        edtCameraTestAppAbs app;
        app.addMode( "science" );
        std::string mode = "science";
        REQUIRE( app.pdvConfig( mode ) == 0 );
        REQUIRE( app.m_pdv != nullptr );
    }
}

/// Verify every branch of pdvSerialWriteRead().
/** The cases are a command failure, an initial timeout with power off and with power on,
 * a matched terminator, an empty response with no terminator, a terminator that the SDK
 * cannot report, and the drain of trailing bytes after a match. The scripted wait results
 * and read chunks in the stub state steer the function through each path.
 *
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "edtCamera pdvSerialWriteRead handles every serial transaction outcome", "[edtCamera]" )
{
    resetStub();

    SECTION( "command failure is reported when logErrors is true" )
    {
        edtCameraTestApp app;
        g_edt.commandResult = -1;
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) < 0 );
    }

    SECTION( "command failure is silent when logErrors is false" )
    {
        edtCameraTestApp app;
        g_edt.commandResult = -1;
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", false ) < 0 );
    }

    SECTION( "an initial timeout while power is off returns without logging" )
    {
        edtCameraTestApp app;
        app.m_powerMgtEnabled = true;
        app.m_powerState      = 0;
        app.m_powerTargetState = 0;
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) < 0 );
    }

    SECTION( "an initial timeout while powered logs the timeout (logErrors true)" )
    {
        edtCameraTestApp app;
        std::string      response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) < 0 );
    }

    SECTION( "an initial timeout while powered is silent (logErrors false)" )
    {
        edtCameraTestApp app;
        std::string      response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", false ) < 0 );
    }

    SECTION( "a matched terminator returns the accumulated response" )
    {
        edtCameraTestApp app;
        g_edt.waitResults.push_back( 1 ); // The initial wait succeeds.
        queueReadChunks( { "OK\n" } );
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) == 0 );
        REQUIRE( response == "OK\n" );
    }

    SECTION( "an empty response with no terminator match logs a timeout (logErrors true)" )
    {
        edtCameraTestApp app;
        g_edt.waitResults.push_back( 1 ); // The initial wait succeeds.
        g_edt.waitResults.push_back( 0 ); // The wait inside the read loop times out and ends the loop.
        queueReadChunks( { "" } );
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) < 0 );
    }

    SECTION( "an empty response with no terminator match is silent (logErrors false)" )
    {
        edtCameraTestApp app;
        g_edt.waitResults.push_back( 1 );
        g_edt.waitResults.push_back( 0 );
        queueReadChunks( { "" } );
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", false ) < 0 );
    }

    SECTION( "an unavailable waitchar falls through to success despite an empty response" )
    {
        edtCameraTestApp app;
        g_edt.waitcharAvailable = false;
        g_edt.waitResults.push_back( 1 );
        g_edt.waitResults.push_back( 0 );
        queueReadChunks( { "" } );
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) == 0 );
        REQUIRE( response.empty() );
    }

    SECTION( "trailing bytes after a match are drained before returning success" )
    {
        edtCameraTestApp app;
        g_edt.waitResults.push_back( 1 ); // The initial wait succeeds.
        g_edt.waitResults.push_back( 5 ); // The first drain wait finds trailing bytes.
        g_edt.waitResults.push_back( 0 ); // The second drain wait times out and ends the drain.
        queueReadChunks( { "OK\n", "trailing" } );
        std::string response;
        REQUIRE( app.pdvSerialWriteRead( response, "cmd", true ) == 0 );
        REQUIRE( response == "OK\n" );
    }
}

/// Verify that setupConfig() and loadConfig() read the framegrabber section of a config file.
/** The unit, channel, and buffer count must come from the file. The serial timeouts must keep
 * their default values.
 *
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "edtCamera setupConfig and loadConfig manage unit, channel, and buffer settings", "[edtCamera]" )
{
    edtCameraTestApp app;

    mx::app::writeConfigFile( "/tmp/edtCamera_test.conf",
                              { "framegrabber", "framegrabber", "framegrabber" },
                              { "pdv_unit", "pdv_channel", "numBuffs" },
                              { "3", "2", "8" } );

    mx::app::appConfigurator config;

    REQUIRE( app.setupConfig( config ) == 0 );

    config.readConfig( "/tmp/edtCamera_test.conf" );

    REQUIRE( app.loadConfig( config ) == 0 );

    REQUIRE( app.m_unit == 3 );
    REQUIRE( app.m_channel == 2 );
    REQUIRE( app.m_numBuffs == 8 );
    REQUIRE( app.m_readTimeout == 1000 );
    REQUIRE( app.m_writeTimeout == 1000 );
}

/// Verify the remaining lifecycle entry points.
/** appStartup() must fail when the startup mode cannot be configured and succeed otherwise.
 * appLogic(), onPowerOff(), whilePowerOff(), appShutdown(), and updateINDI() must return 0.
 *
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "edtCamera lifecycle entry points succeed or report configuration failures", "[edtCamera]" )
{
    resetStub();

    SECTION( "appStartup fails when the startup mode cannot be configured" )
    {
        edtCameraTestApp app;
        app.m_startupMode = "";
        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup succeeds once a startup mode is configured" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        app.m_startupMode = "science";
        REQUIRE( app.appStartup() == 0 );
    }

    SECTION( "appLogic, onPowerOff, whilePowerOff, appShutdown, and updateINDI are no-ops" )
    {
        edtCameraTestApp app;
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.whilePowerOff() == 0 );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( app.updateINDI() == 0 );
    }
}

/// Verify the acquisition helpers that `dev::frameGrabber` calls.
/** pdvStartAcquisition() must start the configured number of buffers. pdvAcquire() must copy
 * the DMA timestamp into the caller's timespec. pdvReconfig() must clear the next mode on
 * success and keep it on failure.
 *
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "edtCamera acquisition helpers start images, fetch timestamps, and reconfigure", "[edtCamera]" )
{
    resetStub();

    SECTION( "pdvStartAcquisition starts the configured number of buffers" )
    {
        edtCameraTestApp app;
        app.m_numBuffs = 6;
        REQUIRE( app.pdvStartAcquisition() == 0 );
        REQUIRE( g_edt.startImagesCalls == 1 );
        REQUIRE( g_edt.lastStartNumBuffs == 6 );
    }

    SECTION( "pdvAcquire fills in the image timestamp" )
    {
        edtCameraTestApp app;
        g_edt.waitTimeSec  = 42;
        g_edt.waitTimeNsec = 99;
        timespec ts{ 0, 0 };
        REQUIRE( app.pdvAcquire( ts ) == 0 );
        REQUIRE( ts.tv_sec == 42 );
        REQUIRE( ts.tv_nsec == 99 );
        REQUIRE( g_edt.startImageCalls == 1 );
    }

    SECTION( "pdvReconfig succeeds and clears the next mode" )
    {
        edtCameraTestApp app;
        app.addMode( "science" );
        app.m_nextMode = "science";
        REQUIRE( app.pdvReconfig() == 0 );
        REQUIRE( app.m_nextMode.empty() );
    }

    SECTION( "pdvReconfig reports and retries on failure" )
    {
        edtCameraTestApp app;
        app.m_nextMode = ""; // An empty mode name always fails pdvConfig. pdvReconfig still returns 0 and sleeps one second so the caller can retry.
        REQUIRE( app.pdvReconfig() == 0 );
    }
}

/// Verify that destroying an `edtCamera` after a successful configuration closes the pdv handle.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "edtCamera destructor closes an open pdv handle", "[edtCamera]" )
{
    resetStub();

    {
        edtCameraTestApp app;
        app.addMode( "science" );
        std::string mode = "science";
        REQUIRE( app.pdvConfig( mode ) == 0 );
    }

    REQUIRE( g_edt.closeCalls > 0 );
}
