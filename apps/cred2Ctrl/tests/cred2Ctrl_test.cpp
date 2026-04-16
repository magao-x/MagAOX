/** \file cred2Ctrl_test.cpp
 * \brief Catch2 tests for the cred2Ctrl app.
 * \author OpenAI Codex
 *
 * \ingroup cred2Ctrl_files
 */

/** \defgroup cred2Ctrl_unit_test cred2Ctrl Unit Tests
 * \brief Unit tests for the cred2Ctrl application.
 *
 * \ingroup application_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define protected public
#include "../cred2Ctrl.hpp"
#undef protected

using namespace MagAOX::app;

namespace
{

using cred2MagAOXAppT = MagAOX::app::MagAOXApp<>;

/// Scripted serial response returned by the EDT serial-command stubs.
struct serialResponse
{
    std::string response;               ///< Response returned by the next serial read.
    int         commandResult{ 0 };     ///< Return code from `pdv_serial_command`.
    int         initialWaitResult{ 1 }; ///< Return code from the first `pdv_serial_wait`.
};

/// Shared EDT stub state used to verify acquisition and serial calls from `cred2Ctrl`.
struct edtStubState
{
    /// Reset the stub to a known default state before each test section.
    void reset()
    {
        startImagesCalls  = 0;
        lastStartNumBuffs = -1;
        startImageCalls   = 0;
        waitTimeSec       = 0;
        waitTimeNsec      = 0;
        waitImage.fill( 0 );
        readcfgReturn = 0;
        readcfgCalls  = 0;
        multibufCalls = 0;
        serialResponses.clear();
        serialCommands.clear();
        activeSerialResponse.clear();
        activeSerialReadPending = false;
        activeSerialTransaction = false;
        activeSerialWaitResult  = 0;
        activeSerialWaitServed  = false;
        baudSetResult           = 0;
        baudCurrent             = 115200;
        baudSetCalls            = 0;
        baudGetCalls            = 0;
        lastRequestedBaud       = -1;
    }

    int                        startImagesCalls{ 0 };   ///< Number of `pdv_start_images` calls observed.
    int                        lastStartNumBuffs{ -1 }; ///< Most recent requested EDT buffer count.
    int                        startImageCalls{ 0 };    ///< Number of `pdv_start_image` calls observed.
    uint                       waitTimeSec{ 0 };        ///< Seconds returned by `pdv_wait_last_image_timed`.
    uint                       waitTimeNsec{ 0 };       ///< Nanoseconds returned by `pdv_wait_last_image_timed`.
    std::array<u_char, 64>     waitImage{};             ///< Raw EDT frame buffer returned to the app.
    int                        readcfgReturn{ 0 };      ///< Return code forced from `pdv_readcfg`.
    int                        readcfgCalls{ 0 };       ///< Number of `pdv_readcfg` calls observed.
    int                        multibufCalls{ 0 };      ///< Number of `pdv_multibuf` calls observed.
    std::deque<serialResponse> serialResponses;         ///< Scripted serial responses queued for future commands.
    std::vector<std::string>   serialCommands;          ///< Serial commands issued by the app under test.
    std::string                activeSerialResponse;    ///< Active serial response returned by the current command.
    bool activeSerialReadPending{ false }; ///< Indicates that the next `pdv_serial_read` should return data.
    bool activeSerialTransaction{ false }; ///< Indicates that a serial command is mid-transaction.
    int  activeSerialWaitResult{ 0 };      ///< Return code for the first wait after a command.
    bool activeSerialWaitServed{ false };  ///< Tracks whether the first serial wait has been consumed.
    int  baudSetResult{ 0 };               ///< Return code forced from `pdv_serial_set_baud`.
    int  baudCurrent{ 115200 };            ///< Baud value returned by `pdv_serial_get_baud`.
    int  baudSetCalls{ 0 };                ///< Number of `pdv_serial_set_baud` calls observed.
    int  baudGetCalls{ 0 };                ///< Number of `pdv_serial_get_baud` calls observed.
    int  lastRequestedBaud{ -1 };          ///< Most recent baud requested by the app under test.
};

/// Global EDT stub state for the `extern "C"` wrappers below.
edtStubState g_edtStubState;

/// Reset all external-library stub state before a test.
[[maybe_unused]] void resetStubState()
{
    g_edtStubState.reset();
}

/// Queue one scripted serial response for the next `pdvSerialWriteRead` invocation.
[[maybe_unused]] void
queueSerialResponse( const std::string &response, int commandResult = 0, int initialWaitResult = 1 )
{
    g_edtStubState.serialResponses.push_back( { response, commandResult, initialWaitResult } );
}

} // namespace

extern "C"
{

    Dependent *pdv_alloc_dependent()
    {
        return reinterpret_cast<Dependent *>( malloc( sizeof( Dependent ) ) );
    }

    int pdv_readcfg( const char *configFile, Dependent *dd_p, Edtinfo *edtinfo )
    {
        static_cast<void>( configFile );
        static_cast<void>( dd_p );
        static_cast<void>( edtinfo );
        ++g_edtStubState.readcfgCalls;
        return g_edtStubState.readcfgReturn;
    }

    EdtDev *edt_open_channel( const char *deviceName, int unit, int channel )
    {
        static EdtDev device;
        static_cast<void>( deviceName );
        static_cast<void>( unit );
        static_cast<void>( channel );
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
        return 0;
    }

    void edt_close( EdtDev *edt_p )
    {
        static_cast<void>( edt_p );
    }

    PdvDev *pdv_open_channel( const char *deviceName, int unit, int channel )
    {
        static PdvDev device;
        static_cast<void>( deviceName );
        static_cast<void>( unit );
        static_cast<void>( channel );
        return &device;
    }

    void pdv_close( PdvDev *pdv_p )
    {
        static_cast<void>( pdv_p );
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
        return 640;
    }

    int pdv_get_height( PdvDev *pdv_p )
    {
        static_cast<void>( pdv_p );
        return 512;
    }

    int pdv_get_depth( PdvDev *pdv_p )
    {
        static_cast<void>( pdv_p );
        return 16;
    }

    char *pdv_get_cameratype( PdvDev *pdv_p )
    {
        static char cameraType[] = "stub_pdv";
        static_cast<void>( pdv_p );
        return cameraType;
    }

    void pdv_multibuf( PdvDev *pdv_p, int numBuffs )
    {
        static_cast<void>( pdv_p );
        static_cast<void>( numBuffs );
        ++g_edtStubState.multibufCalls;
    }

    void pdv_start_images( PdvDev *pdv_p, int numBuffs )
    {
        static_cast<void>( pdv_p );
        ++g_edtStubState.startImagesCalls;
        g_edtStubState.lastStartNumBuffs = numBuffs;
    }

    u_char *pdv_wait_last_image_timed( PdvDev *pdv_p, uint dmaTimeStamp[2] )
    {
        static_cast<void>( pdv_p );
        dmaTimeStamp[0] = g_edtStubState.waitTimeSec;
        dmaTimeStamp[1] = g_edtStubState.waitTimeNsec;
        return g_edtStubState.waitImage.data();
    }

    void pdv_start_image( PdvDev *pdv_p )
    {
        static_cast<void>( pdv_p );
        ++g_edtStubState.startImageCalls;
    }

    int pdv_serial_read( PdvDev *pdv_p, char *buf, int size )
    {
        static_cast<void>( pdv_p );
        if( buf != nullptr && size > 0 )
        {
            buf[0] = '\0';
        }

        if( !g_edtStubState.activeSerialReadPending || buf == nullptr || size <= 0 )
        {
            return 0;
        }

        size_t copyCount = g_edtStubState.activeSerialResponse.size();
        if( copyCount > static_cast<size_t>( size - 1 ) )
        {
            copyCount = static_cast<size_t>( size - 1 );
        }

        std::copy_n( g_edtStubState.activeSerialResponse.data(), copyCount, buf );
        buf[copyCount] = '\0';

        g_edtStubState.activeSerialReadPending = false;

        return static_cast<int>( copyCount );
    }

    int pdv_serial_command( PdvDev *pdv_p, const char *command )
    {
        static_cast<void>( pdv_p );
        g_edtStubState.serialCommands.emplace_back( command == nullptr ? "" : command );

        g_edtStubState.activeSerialResponse.clear();
        g_edtStubState.activeSerialReadPending = false;
        g_edtStubState.activeSerialTransaction = false;
        g_edtStubState.activeSerialWaitResult  = 0;
        g_edtStubState.activeSerialWaitServed  = false;

        if( g_edtStubState.serialResponses.empty() )
        {
            return 0;
        }

        serialResponse response = g_edtStubState.serialResponses.front();
        g_edtStubState.serialResponses.pop_front();

        if( response.commandResult < 0 )
        {
            return response.commandResult;
        }

        g_edtStubState.activeSerialResponse    = response.response;
        g_edtStubState.activeSerialReadPending = true;
        g_edtStubState.activeSerialTransaction = true;
        g_edtStubState.activeSerialWaitResult  = response.initialWaitResult;

        return response.commandResult;
    }

    int pdv_serial_wait( PdvDev *pdv_p, int timeout, int count )
    {
        static_cast<void>( pdv_p );
        static_cast<void>( timeout );
        static_cast<void>( count );

        if( !g_edtStubState.activeSerialTransaction )
        {
            return 0;
        }

        if( !g_edtStubState.activeSerialWaitServed )
        {
            g_edtStubState.activeSerialWaitServed = true;
            return g_edtStubState.activeSerialWaitResult;
        }

        return 0;
    }

    int pdv_get_waitchar( PdvDev *pdv_p, u_char *waitc )
    {
        static_cast<void>( pdv_p );
        if( waitc != nullptr )
        {
            *waitc = '\n';
        }
        return 1;
    }

    int pdv_serial_set_baud( PdvDev *pdv_p, int baud )
    {
        static_cast<void>( pdv_p );
        ++g_edtStubState.baudSetCalls;
        g_edtStubState.lastRequestedBaud = baud;
        if( g_edtStubState.baudSetResult == 0 )
        {
            g_edtStubState.baudCurrent = baud;
        }

        return g_edtStubState.baudSetResult;
    }

    int pdv_serial_get_baud( PdvDev *pdv_p )
    {
        static_cast<void>( pdv_p );
        ++g_edtStubState.baudGetCalls;
        return g_edtStubState.baudCurrent;
    }

} // extern "C"

namespace libXWCTest
{

/// Namespace for `cred2Ctrl` unit tests.
/** \ingroup cred2Ctrl_unit_test
 */
namespace cred2CtrlTest
{

namespace
{

/// Build a unique shmim name for one temporary test stream.
std::string uniqueShmimName( const std::string &suffix )
{
    static unsigned counter = 0;

    ++counter;

    return "cred2Ctrl_test_" + suffix + "_" + std::to_string( ::getpid() ) + "_" + std::to_string( counter );
}

/// Build a unique temporary config-file path for one test.
std::string uniqueConfigPath( const std::string &suffix )
{
    return "/tmp/cred2Ctrl_test_" + suffix + "_" + std::to_string( ::getpid() ) + ".conf";
}

/// Remove one temporary file and ignore missing-file errors.
void removeIfPresent( const std::string &path )
{
    if( !path.empty() )
    {
        static_cast<void>( ::unlink( path.c_str() ) );
    }
}

/// Test harness exposing protected `cred2Ctrl` helpers.
class cred2Ctrl_test : public MagAOX::app::cred2Ctrl
{
  public:
    /// Construct a testable controller instance with unique runtime identifiers.
    cred2Ctrl_test()
    {
        m_configName = uniqueShmimName( "config" );
        m_shmimName  = uniqueShmimName( "main" );
    }

    /// Remove the generated EDT config file after each test.
    ~cred2Ctrl_test() noexcept
    {
        removeIfPresent( m_configFile );
    }
};

/// Put the app into the nominal powered-on state used by the serial helpers.
[[maybe_unused]] void setPoweredOn( cred2Ctrl_test &app )
{
    static_cast<cred2MagAOXAppT &>( app ).m_powerState = 1;
    app.m_powerTargetState                             = 1;
}

/// Load the standard app configuration with optional overrides for one test.
[[maybe_unused]] void loadAppConfig( cred2Ctrl_test                 &app,
                                     const std::string              &suffix,
                                     const std::vector<std::string> &sections,
                                     const std::vector<std::string> &keywords,
                                     const std::vector<std::string> &values )
{
    const std::string configPath = uniqueConfigPath( suffix );

    app.setupConfig();
    mx::app::writeConfigFile( configPath, sections, keywords, values );
    app.config.readConfig( configPath );
    app.loadConfig();
    removeIfPresent( configPath );

    app.m_tel.logPath( "/tmp" );
    app.m_tel.logName( app.m_configName );
    app.m_tel.logExt( "bintel" );
}

/// Load the minimum configuration required for `cred2Ctrl` startup and runtime tests.
[[maybe_unused]] void loadDefaultConfig( cred2Ctrl_test &app, const std::string &suffix )
{
    loadAppConfig( app, suffix, { "framegrabber" }, { "shmimName" }, { uniqueShmimName( "stream" ) } );
}

/// Start a short-lived framegrabber thread so `frameGrabber::appLogic()` sees a running worker.
[[maybe_unused]] void startFgThread( cred2Ctrl_test &app, int sleepMs = 200 )
{
    app.m_fgThread =
        std::thread( [sleepMs]() { std::this_thread::sleep_for( std::chrono::milliseconds( sleepMs ) ); } );
}

/// Join the temporary framegrabber thread if a test started one.
[[maybe_unused]] void joinFgThread( cred2Ctrl_test &app )
{
    if( app.m_fgThread.joinable() )
    {
        app.m_fgThread.join();
    }
}

/// Keep a temporary framegrabber thread joined even when a test fails mid-scope.
struct fgThreadScope
{
    cred2Ctrl_test &m_app; ///< App whose temporary framegrabber thread is managed by this scope.

    /// Start a temporary framegrabber thread for the lifetime of this scope.
    explicit fgThreadScope( cred2Ctrl_test &app, int sleepMs = 200 ) : m_app( app )
    {
        startFgThread( m_app, sleepMs );
    }

    /// Join the temporary framegrabber thread on scope exit.
    ~fgThreadScope()
    {
        joinFgThread( m_app );
    }
};

/// Ensure startup-driven telemetry resources are shut down when a test exits early.
struct startupScope
{
    cred2Ctrl_test &m_app;     ///< App whose startup resources are managed by this scope.
    bool            m_started; ///< Tracks whether `appStartup()` completed successfully in the test.

    /// Construct a startup guard for one test app instance.
    explicit startupScope( cred2Ctrl_test &app ) : m_app( app ), m_started( false )
    {
    }

    /// Record whether startup completed so shutdown only runs when needed.
    void markStarted( bool started )
    {
        m_started = started;
    }

    /// Shut down telemetry logging and app resources on scope exit after a successful startup.
    ~startupScope()
    {
        if( m_started )
        {
            m_app.m_shutdown = 1;
            m_app.m_tel.logShutdown( true );
            static_cast<void>( m_app.appShutdown() );
        }
    }
};

/// Report whether a telemetry timestamp has been written at least once.
[[maybe_unused]] bool hasRecordedTime( const timespec &ts )
{
    return ts.tv_sec != 0 || ts.tv_nsec != 0;
}

} // namespace

#ifndef CRED2CTRL_TEST_SUPPORT_ONLY

/// Verify configuration loading writes the runtime EDT config and applies overrides.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl configuration loading writes a runtime EDT config", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setupConfig() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::loadConfig() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::writeConfig() );
    #endif
    // clang-format on

    SECTION( "loadConfig uses the default serial baud and seeds the runtime camera mode" )
    {
        cred2Ctrl_test app;

        loadAppConfig( app, "defaults", { "unused" }, { "value" }, { "0" } );

        REQUIRE( app.m_serialBaud == 115200 );
        REQUIRE( app.m_startupMode == "runtime" );
        REQUIRE( app.m_cameraModes.count( "runtime" ) == 1 );
        REQUIRE( app.m_cameraModes["runtime"].m_configFile == app.m_configFile );

        std::ifstream configFile( app.m_configFile );
        REQUIRE( configFile.good() );

        const std::string contents( ( std::istreambuf_iterator<char>( configFile ) ),
                                    std::istreambuf_iterator<char>() );
        REQUIRE( contents.find( "camera_model:                  \"C-RED 2\"" ) != std::string::npos );
        REQUIRE( contents.find( "width:                         640" ) != std::string::npos );
        REQUIRE( contents.find( "height:                        512" ) != std::string::npos );
        REQUIRE( contents.find( "serial_baud:                  115200" ) != std::string::npos );
    }

    SECTION( "loadConfig applies configured serial-baud overrides to the runtime config file" )
    {
        cred2Ctrl_test app;

        loadAppConfig( app,
                       "serial_baud",
                       { "camera", "framegrabber" },
                       { "serialBaud", "shmimName" },
                       { "57600", uniqueShmimName( "override" ) } );

        REQUIRE( app.m_serialBaud == 57600 );
        REQUIRE( app.m_configFile.find( "cred2_" ) != std::string::npos );
        REQUIRE( app.m_cameraModes["runtime"].m_configFile == app.m_configFile );

        std::ifstream configFile( app.m_configFile );
        REQUIRE( configFile.good() );

        const std::string contents( ( std::istreambuf_iterator<char>( configFile ) ),
                                    std::istreambuf_iterator<char>() );
        REQUIRE( contents.find( "serial_baud:                  57600" ) != std::string::npos );
    }

    SECTION( "loadConfig marks the app for shutdown when the runtime EDT config cannot be written" )
    {
        cred2Ctrl_test    app;
        const std::string configPath = uniqueConfigPath( "load_failure" );

        app.m_configName = "missing/cred2_write_failure";
        app.setupConfig();
        mx::app::writeConfigFile(
            configPath, { "framegrabber" }, { "shmimName" }, { uniqueShmimName( "load_failure_stream" ) } );
        app.config.readConfig( configPath );
        app.loadConfig();
        removeIfPresent( configPath );

        REQUIRE( app.m_shutdown == true );
    }

    SECTION( "heap deletion and direct EDT stub calls cover the remaining test harness wrappers" )
    {
        char shortError[2] = { 'x', '\0' };
        char shortRead[4]  = { '\0', '\0', '\0', '\0' };

        edt_perror( shortError );
        REQUIRE( shortError[0] == '\0' );

        queueSerialResponse( "abcdef\n" );
        REQUIRE( pdv_serial_command( nullptr, "stub command" ) == 0 );
        REQUIRE( pdv_serial_wait( nullptr, 0, 0 ) == 1 );
        REQUIRE( pdv_serial_read( nullptr, shortRead, sizeof( shortRead ) ) == 3 );
        REQUIRE( std::string( shortRead ) == "abc" );

        cred2Ctrl_test *heapApp = new cred2Ctrl_test;
        REQUIRE( heapApp != nullptr );
        delete heapApp;

        MagAOX::app::cred2Ctrl *baseHeapApp = new cred2Ctrl_test;
        REQUIRE( baseHeapApp != nullptr );
        delete baseHeapApp;

        MagAOX::app::cred2Ctrl *plainHeapApp = new MagAOX::app::cred2Ctrl;
        REQUIRE( plainHeapApp != nullptr );
        delete plainHeapApp;
    }
}

/// Verify local response helpers cover the C-RED 2-specific preset mappings.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl helper mappings normalize responses and preset names", "[cred2Ctrl]" )
{
    bool               enabled        = false;
    float              parsedValue    = 0;
    float              centerX        = 0;
    float              centerY        = 0;
    int                fanRangeFirst  = 0;
    int                fanRangeSecond = 0;
    std::string        gainName;
    std::string        commandGain;
    int                fanPercent  = -1;
    int                startColumn = 0;
    int                endColumn   = 0;
    int                startRow    = 0;
    int                endRow      = 0;
    int                width       = 0;
    int                height      = 0;
    cred2Roi           roi;
    std::vector<float> parsedValues;

    REQUIRE( cred2LowerResponse( " Medium\r\nfli-cli>" ) == "medium" );

    REQUIRE( cred2FanPresetName( 0.0f ) == "off" );
    REQUIRE( cred2FanPresetName( 25.0f ) == "p25" );
    REQUIRE( cred2FanPresetName( 50.0f ) == "p50" );
    REQUIRE( cred2FanPresetName( 75.0f ) == "p75" );
    REQUIRE( cred2FanPresetName( 100.0f ) == "p100" );

    REQUIRE( cred2FanPresetPercent( fanPercent, "off" ) == 0 );
    REQUIRE( fanPercent == 0 );
    REQUIRE( cred2FanPresetPercent( fanPercent, "p25" ) == 0 );
    REQUIRE( fanPercent == 25 );
    REQUIRE( cred2FanPresetPercent( fanPercent, "p50" ) == 0 );
    REQUIRE( fanPercent == 50 );
    REQUIRE( cred2FanPresetPercent( fanPercent, "p75" ) == 0 );
    REQUIRE( fanPercent == 75 );
    REQUIRE( cred2FanPresetPercent( fanPercent, "p100" ) == 0 );
    REQUIRE( fanPercent == 100 );
    REQUIRE( cred2FanPresetPercent( fanPercent, "mystery" ) == -1 );

    REQUIRE( cred2AnalogGainName( gainName, "medium" ) == 0 );
    REQUIRE( gainName == "med" );
    REQUIRE( cred2AnalogGainName( gainName, "med" ) == 0 );
    REQUIRE( gainName == "med" );
    REQUIRE( cred2AnalogGainName( gainName, "high" ) == 0 );
    REQUIRE( gainName == "high" );
    REQUIRE( cred2AnalogGainName( gainName, "low" ) == 0 );
    REQUIRE( gainName == "low" );
    REQUIRE( cred2AnalogGainName( gainName, "unknown" ) == -1 );

    REQUIRE( cred2AnalogGainCommand( commandGain, "low" ) == 0 );
    REQUIRE( commandGain == "low" );
    REQUIRE( cred2AnalogGainCommand( commandGain, "med" ) == 0 );
    REQUIRE( commandGain == "medium" );
    REQUIRE( cred2AnalogGainCommand( commandGain, "high" ) == 0 );
    REQUIRE( commandGain == "high" );
    REQUIRE( cred2AnalogGainCommand( commandGain, "invalid" ) == -1 );

    REQUIRE( cred2ParseFloat( parsedValue, "12.5" ) == 0 );
    REQUIRE( parsedValue == Approx( 12.5f ) );
    REQUIRE( cred2ParseFloat( parsedValue, "" ) == -1 );
    REQUIRE( cred2ParseFloat( parsedValue, "12.5 junk" ) == -1 );

    REQUIRE( cred2ParseFloatVector( parsedValues, "1:2:3", 3 ) == 0 );
    REQUIRE( parsedValues == std::vector<float>{ 1.0f, 2.0f, 3.0f } );
    REQUIRE( cred2ParseFloatVector( parsedValues, "", 0 ) == -1 );
    REQUIRE( cred2ParseFloatVector( parsedValues, "1:bad:3", 3 ) == -1 );
    REQUIRE( parsedValues.empty() );

    REQUIRE( cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "off" ) == 0 );
    REQUIRE( enabled == false );
    REQUIRE( startColumn == 0 );
    REQUIRE( endColumn == 0 );
    REQUIRE( startRow == 0 );
    REQUIRE( endRow == 0 );
    REQUIRE( cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "" ) == -1 );
    REQUIRE( cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "maybe" ) == -1 );
    REQUIRE( cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "on:1-2" ) == -1 );
    REQUIRE( cred2ParseCropState( enabled, startColumn, endColumn, startRow, endRow, "on:bad:1-2" ) == -1 );

    REQUIRE( cred2ParseRange( fanRangeFirst, fanRangeSecond, "1-4" ) == 0 );
    REQUIRE( fanRangeFirst == 1 );
    REQUIRE( fanRangeSecond == 4 );
    REQUIRE( cred2ParseRange( fanRangeFirst, fanRangeSecond, "1" ) == -1 );

    REQUIRE( cred2RoiFromCenter( roi, 319.5f, 255.5f, 640, 512, 640, 512 ) == 0 );
    REQUIRE( roi.fullFrame == true );
    REQUIRE( cred2RoiFromCenter( roi, 319.5f, 255.5f, 0, 512, 640, 512 ) == -1 );
    REQUIRE( cred2RoiToCenter( centerX, centerY, width, height, roi, 640, 512 ) == 0 );
    REQUIRE( centerX == Approx( 319.5f ) );
    REQUIRE( centerY == Approx( 255.5f ) );
    REQUIRE( width == 640 );
    REQUIRE( height == 512 );

    roi.startColumn = 10;
    roi.endColumn   = 5;
    REQUIRE( cred2RoiToCenter( centerX, centerY, width, height, roi, 640, 512 ) == -1 );
}

/// Verify serial helper wrappers clean responses and report transport failures consistently.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl command helpers clean responses and validate acknowledgements", "[cred2Ctrl]" )
{
    cred2Ctrl_test app;
    std::string    response;

    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::sendCommand( response, "fps raw", true ) );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::issueCommand( "set fps 100", false ) );
    #endif
    // clang-format on

    SECTION( "sendCommand strips prompts and returns the cleaned response" )
    {
        setPoweredOn( app );
        queueSerialResponse( "400\r\nfli-cli>\n" );

        REQUIRE( app.sendCommand( response, "fps raw" ) == 0 );
        REQUIRE( response == "400" );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps raw" } );
    }

    SECTION( "sendCommand returns -1 without logging when the camera is powered off" )
    {
        queueSerialResponse( "", -1 );

        REQUIRE( app.sendCommand( response, "fps raw" ) == -1 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps raw" } );
    }

    SECTION( "issueCommand accepts missing responses when allowNoResponse is true" )
    {
        setPoweredOn( app );
        queueSerialResponse( "", -1 );

        REQUIRE( app.issueCommand( "set fps 100", true ) == 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set fps 100" } );
    }

    SECTION( "issueCommand rejects explicit error responses" )
    {
        setPoweredOn( app );
        queueSerialResponse( "Error: busy\n" );

        REQUIRE( app.issueCommand( "set fps 100", false ) == -1 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set fps 100" } );
    }

    SECTION( "issueCommand accepts normal acknowledgement responses" )
    {
        setPoweredOn( app );
        queueSerialResponse( "OK\n" );

        REQUIRE( app.issueCommand( "set fps 100", false ) == 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set fps 100" } );
    }
}

/// Verify ROI synchronization handles full-frame, cropped, and reconfiguring camera states.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl syncROIFromCamera tracks camera cropping state", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::syncROIFromCamera() );
    #endif
    // clang-format on

    SECTION( "full-frame cropping disables camera-side crop tracking" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_raw_width  = app.m_full_w;
        app.m_raw_height = app.m_full_h;
        queueSerialResponse( "off\n" );

        REQUIRE( app.syncROIFromCamera() == 0 );
        REQUIRE( app.m_cameraCropEnabled == false );
        REQUIRE( app.m_currentROI.x == Approx( app.m_full_x ) );
        REQUIRE( app.m_currentROI.y == Approx( app.m_full_y ) );
        REQUIRE( app.m_currentROI.w == app.m_full_w );
        REQUIRE( app.m_currentROI.h == app.m_full_h );
    }

    SECTION( "explicit crop coordinates update the cached ROI" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_raw_width  = 256;
        app.m_raw_height = 256;
        queueSerialResponse( "on:192-447:128-383\n" );

        REQUIRE( app.syncROIFromCamera() == 0 );
        REQUIRE( app.m_cameraCropEnabled == true );
        REQUIRE( app.m_currentROI.x == Approx( 319.5f ) );
        REQUIRE( app.m_currentROI.y == Approx( 255.5f ) );
        REQUIRE( app.m_currentROI.w == 256 );
        REQUIRE( app.m_currentROI.h == 256 );
        REQUIRE( app.m_width == 256 );
        REQUIRE( app.m_height == 256 );
    }

    SECTION( "querying separate row and column limits handles cameras that omit them from the summary response" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_raw_width  = 256;
        app.m_raw_height = 256;
        queueSerialResponse( "on\n" );
        queueSerialResponse( "192-447\n" );
        queueSerialResponse( "128-383\n" );

        REQUIRE( app.syncROIFromCamera() == 0 );
        REQUIRE( app.m_cameraCropEnabled == true );
        REQUIRE( app.m_currentROI.x == Approx( 319.5f ) );
        REQUIRE( app.m_currentROI.y == Approx( 255.5f ) );
        REQUIRE( app.m_currentROI.w == 256 );
        REQUIRE( app.m_currentROI.h == 256 );
        REQUIRE( g_edtStubState.serialCommands ==
                 std::vector<std::string>{ "cropping raw", "cropping columns raw", "cropping rows raw" } );
    }

    SECTION( "invalid crop responses return an error" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "garbage\n" );

        REQUIRE( app.syncROIFromCamera() == -1 );
    }

    SECTION( "mismatched EDT dimensions trigger a reconfiguration and config rewrite" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "roi_reconfig" );
        setPoweredOn( app );
        app.m_modeName   = "runtime";
        app.m_raw_width  = 320;
        app.m_raw_height = 256;
        queueSerialResponse( "off\n" );

        REQUIRE( app.syncROIFromCamera() == 0 );
        REQUIRE( g_edtStubState.readcfgCalls > 0 );
        REQUIRE( app.m_modeName == "runtime" );
    }

    SECTION( "missing crop-column details are reported as an error" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_raw_width  = 256;
        app.m_raw_height = 256;
        queueSerialResponse( "on\n" );
        queueSerialResponse( "bad\n" );

        REQUIRE( app.syncROIFromCamera() == -1 );
    }

    SECTION( "missing crop-row details are reported as an error" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_raw_width  = 256;
        app.m_raw_height = 256;
        queueSerialResponse( "on\n" );
        queueSerialResponse( "192-447\n" );
        queueSerialResponse( "bad\n" );

        REQUIRE( app.syncROIFromCamera() == -1 );
    }

    SECTION( "camera-reported crop corners that cannot map to a valid ROI are rejected" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_raw_width  = 256;
        app.m_raw_height = 256;
        queueSerialResponse( "on:447-192:128-383\n" );

        REQUIRE( app.syncROIFromCamera() == -1 );
    }

    SECTION( "reconfiguration failures bubble out when the EDT runtime reload fails" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "roi_reconfig_fail" );
        setPoweredOn( app );
        app.m_modeName               = "runtime";
        app.m_raw_width              = 320;
        app.m_raw_height             = 256;
        g_edtStubState.readcfgReturn = -1;
        queueSerialResponse( "off\n" );

        REQUIRE( app.syncROIFromCamera() == -1 );
    }

    SECTION( "runtime config rewrite failures bubble out while syncing a resized ROI" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "roi_write_fail" );
        setPoweredOn( app );
        app.m_modeName   = "runtime";
        app.m_configFile = "/tmp/cred2Ctrl_missing_dir/roi_runtime.cfg";
        app.m_raw_width  = 320;
        app.m_raw_height = 256;
        queueSerialResponse( "off\n" );

        REQUIRE( app.syncROIFromCamera() == -1 );
    }
}

/// Verify the primary getter helpers update cached state and tolerate malformed responses.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl getter helpers parse temperatures fps and limits", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::getTemps() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::getFPS() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::updateFPSLimits() );
    #endif
    // clang-format on

    SECTION( "getTemps caches valid bundled temperatures and setpoint state" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.00\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_temps.motherboard == Approx( 40.50f ) );
        REQUIRE( app.m_temps.frontend == Approx( 37.00f ) );
        REQUIRE( app.m_temps.powerboard == Approx( 40.25f ) );
        REQUIRE( app.m_temps.snake == Approx( -14.92f ) );
        REQUIRE( app.m_temps.setpoint == Approx( -15.0f ) );
        REQUIRE( app.m_temps.peltier == Approx( 2.29f ) );
        REQUIRE( app.m_temps.heatsink == Approx( 27.50f ) );
        REQUIRE( app.m_tempControlStatus == true );
        REQUIRE( app.m_tempControlStatusStr == "ON TARGET" );
        REQUIRE( app.m_tempControlOnTarget == true );
    }

    SECTION( "getTemps keeps the previous cache on malformed bundled temperatures" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_temps.motherboard = 1;
        app.m_temps.frontend    = 2;
        app.m_temps.powerboard  = 3;
        app.m_temps.snake       = 4;
        app.m_temps.setpoint    = 5;
        app.m_temps.peltier     = 6;
        app.m_temps.heatsink    = 7;
        app.m_ccdTemp           = 4;
        app.m_ccdTempSetpt      = 5;

        queueSerialResponse( "40.50:37.00\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_temps.motherboard == Approx( 1.0f ) );
        REQUIRE( app.m_ccdTemp == Approx( 4.0f ) );
        REQUIRE( app.m_ccdTempSetpt == Approx( 5.0f ) );
    }

    SECTION( "getTemps keeps the previous cache when the snake-setpoint query transport fails" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ccdTempSetpt = -5.0f;
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_ccdTempSetpt == Approx( -5.0f ) );
    }

    SECTION( "getTemps keeps the previous cache when the snake-setpoint response is malformed" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ccdTempSetpt = -6.0f;
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "bad-setpoint\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_ccdTempSetpt == Approx( -6.0f ) );
    }

    SECTION( "getTemps reports off-target cooling when the detector is still far from the setpoint" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "40.50:37.00:40.25:-10.00:2.29:27.50\n" );
        queueSerialResponse( "-15.00\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_tempControlStatus == true );
        REQUIRE( app.m_tempControlStatusStr == "OFF TARGET" );
        REQUIRE( app.m_tempControlOnTarget == false );
    }

    SECTION( "getTemps reports TEMP OFF once the warm setpoint is reached" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "40.50:37.00:40.25:20.00:2.29:27.50\n" );
        queueSerialResponse( "20.00\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "TEMP OFF" );
    }

    SECTION( "getTemps reports WARMING while returning to the warm setpoint" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "40.50:37.00:40.25:10.00:2.29:27.50\n" );
        queueSerialResponse( "20.00\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "WARMING" );
    }

    SECTION( "getFPS updates the cached frame rate on valid responses" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "150.5\n" );

        REQUIRE( app.getFPS() == 0 );
        REQUIRE( app.m_fps == Approx( 150.5f ) );
    }

    SECTION( "getFPS rejects malformed responses" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "Result: OK\n" );

        REQUIRE( app.getFPS() == -1 );
    }

    SECTION( "updateFPSLimits updates bounds and clamps the requested fps" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_fpsSet = 1000.0f;
        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );

        REQUIRE( app.updateFPSLimits() == 0 );
        REQUIRE( app.m_minFPS == Approx( 10.0f ) );
        REQUIRE( app.m_maxFPS == Approx( 500.0f ) );
        REQUIRE( app.m_fpsSet == Approx( 500.0f ) );
    }

    SECTION( "updateFPSLimits reports malformed minimum-fps responses" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "not-a-number\n" );

        REQUIRE( app.updateFPSLimits() == -1 );
    }

    SECTION( "updateFPSLimits reports malformed maximum-fps responses" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "not-a-number\n" );

        REQUIRE( app.updateFPSLimits() == -1 );
    }
}

/// Verify the remaining getter helpers parse discrete fan gain and LED state.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl discrete getter helpers cover fallback parsing paths", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::getFanSpeed() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::getAnalogGain() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::getLEDState() );
    #endif
    // clang-format on

    SECTION( "getFanSpeed accepts automatic mode without querying the manual percentage" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "automatic\n" );

        REQUIRE( app.getFanSpeed() == 0 );
        REQUIRE( app.m_fanSpeedName == "auto" );
        REQUIRE( app.m_fanSpeedNameSet == "auto" );
        REQUIRE( app.m_fanSpeedValid == true );
    }

    SECTION( "getFanSpeed falls back to the non-raw fan-speed response when needed" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "manual\n" );
        queueSerialResponse( "speed\n" );
        queueSerialResponse( "75\n" );

        REQUIRE( app.getFanSpeed() == 0 );
        REQUIRE( app.m_fanSpeedName == "p75" );
        REQUIRE( g_edtStubState.serialCommands ==
                 std::vector<std::string>{ "fan mode raw", "fan speed raw", "fan speed" } );
    }

    SECTION( "getFanSpeed falls back to the non-raw mode command when the raw response is not recognizable" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "mystery\n" );
        queueSerialResponse( "manual\n" );
        queueSerialResponse( "50\n" );

        REQUIRE( app.getFanSpeed() == 0 );
        REQUIRE( app.m_fanSpeedName == "p50" );
        REQUIRE( g_edtStubState.serialCommands ==
                 std::vector<std::string>{ "fan mode raw", "fan mode", "fan speed raw" } );
    }

    SECTION( "getFanSpeed rejects mode responses that never indicate auto or manual" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "mystery\n" );
        queueSerialResponse( "still mystery\n" );

        REQUIRE( app.getFanSpeed() == -1 );
    }

    SECTION( "getFanSpeed returns -1 when the raw mode query transport fails" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getFanSpeed() == -1 );
    }

    SECTION( "getFanSpeed returns -1 when the fallback mode query transport fails" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "mystery\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getFanSpeed() == -1 );
    }

    SECTION( "getFanSpeed rejects manual fan-speed responses that remain unparseable" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "manual\n" );
        queueSerialResponse( "speed\n" );
        queueSerialResponse( "still bad\n" );

        REQUIRE( app.getFanSpeed() == -1 );
    }

    SECTION( "getFanSpeed returns -1 when the manual fan-speed transport fails" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "manual\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getFanSpeed() == -1 );
    }

    SECTION( "getAnalogGain maps the reported sensibility name" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "medium\n" );

        REQUIRE( app.getAnalogGain() == 0 );
        REQUIRE( app.m_analogGainName == "med" );
        REQUIRE( app.m_analogGainNameSet == "med" );
        REQUIRE( app.m_analogGainValid == true );
    }

    SECTION( "getAnalogGain rejects unsupported sensibility responses" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "mystery\n" );

        REQUIRE( app.getAnalogGain() == -1 );
    }

    SECTION( "getAnalogGain returns -1 on transport failure" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getAnalogGain() == -1 );
    }

    SECTION( "getLEDState falls back from the raw command to the human-readable response" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "", -1 );
        queueSerialResponse( "on\n" );

        REQUIRE( app.getLEDState() == 0 );
        REQUIRE( app.m_ledState == true );
        REQUIRE( app.m_ledStateSet == true );
        REQUIRE( app.m_ledStateValid == true );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "led raw", "led" } );
    }

    SECTION( "getLEDState rejects responses that do not indicate on or off" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "unknown\n" );
        queueSerialResponse( "still unknown\n" );

        REQUIRE( app.getLEDState() == -1 );
    }

    SECTION( "getLEDState returns -1 when both raw and fallback LED queries fail" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "", -1 );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getLEDState() == -1 );
    }

    SECTION( "getLEDState recognizes descriptive raw responses that still contain off" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "led is off\n" );

        REQUIRE( app.getLEDState() == 0 );
        REQUIRE( app.m_ledState == false );
    }

    SECTION( "getLEDState recognizes descriptive raw responses that still contain on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "led is on\n" );

        REQUIRE( app.getLEDState() == 0 );
        REQUIRE( app.m_ledState == true );
    }

    SECTION( "getLEDState falls back to an off state on the human-readable retry" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "unknown\n" );
        queueSerialResponse( "off\n" );

        REQUIRE( app.getLEDState() == 0 );
        REQUIRE( app.m_ledState == false );
    }

    SECTION( "getLEDState returns -1 when the retry transport fails after an ambiguous raw response" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "unknown\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.getLEDState() == -1 );
    }

    SECTION( "getLEDState falls back to an on state on the human-readable retry" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        queueSerialResponse( "unknown\n" );
        queueSerialResponse( "on\n" );

        REQUIRE( app.getLEDState() == 0 );
        REQUIRE( app.m_ledState == true );
    }
}

/// Verify the setter helpers validate bounds and send the expected serial commands.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl setter helpers validate bounds and update state", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::powerOnDefaults() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setTempControl() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setTempSetPt() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setFPS() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setFanSpeed() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setAnalogGain() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setLED() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setExpTime() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::checkNextROI() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::setNextROI() );
    #endif
    // clang-format on

    SECTION( "powerOnDefaults restores controller-side defaults" )
    {
        cred2Ctrl_test app;

        app.m_tempControlStatusSet = true;
        app.m_tempControlStatus    = true;
        app.m_tempControlStatusStr = "ON TARGET";
        app.m_cameraCropEnabled    = true;
        app.m_fanSpeedValid        = true;
        app.m_analogGainValid      = true;
        app.m_ledStateValid        = true;

        REQUIRE( app.powerOnDefaults() == 0 );
        REQUIRE( app.m_tempControlStatusSet == false );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "TEMP OFF" );
        REQUIRE( app.m_cameraCropEnabled == false );
        REQUIRE( app.m_fanSpeedNameSet == "auto" );
        REQUIRE( app.m_analogGainNameSet == "med" );
        REQUIRE( app.m_ledStateSet == true );
    }

    SECTION( "setTempControl uses the configured setpoint when cooling is enabled" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ccdTempSetpt         = -15.5f;
        app.m_tempControlStatusSet = true;
        queueSerialResponse( "OK\n" );

        REQUIRE( app.setTempControl() == 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set temperatures snake -15.5" } );
        REQUIRE( app.m_tempControlStatus == true );
        REQUIRE( app.m_tempControlStatusStr == "OFF TARGET" );
    }

    SECTION( "setTempControl switches to the warm-up setpoint when cooling is disabled" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_tempControlStatusSet = false;
        queueSerialResponse( "OK\n" );

        REQUIRE( app.setTempControl() == 0 );
        REQUIRE( app.m_ccdTempSetpt == Approx( 20.0f ) );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "WARMING" );
    }

    SECTION( "setTempControl logs a notice when cooling is requested with a warm setpoint" )
    {
        cred2Ctrl_test app;

        app.m_ccdTempSetpt         = 20.0f;
        app.m_tempControlStatusSet = true;

        REQUIRE( app.setTempControl() == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setTempSetPt rejects out-of-range values" )
    {
        cred2Ctrl_test app;

        app.m_ccdTempSetpt = 50.0f;
        REQUIRE( app.setTempSetPt() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setFPS validates bounds before sending the command and refreshes the cached fps" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_minFPS = 10;
        app.m_maxFPS = 500;
        app.m_fpsSet = 125.5f;
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "125.5\n" );

        REQUIRE( app.setFPS() == 0 );
        REQUIRE( app.m_fps == Approx( 125.5f ) );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set fps 125.5", "fps raw" } );
    }

    SECTION( "setFPS rejects requests outside the current min-max bounds" )
    {
        cred2Ctrl_test app;

        app.m_minFPS = 10;
        app.m_maxFPS = 500;
        app.m_fpsSet = 600.0f;

        REQUIRE( app.setFPS() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setFPS returns an error when the camera rejects the command" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_minFPS = 10;
        app.m_maxFPS = 500;
        app.m_fpsSet = 125.5f;
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setFPS() == -1 );
    }

    SECTION( "setFanSpeed handles automatic and manual presets" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_fanSpeedNameSet = "p75";
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "manual\n" );
        queueSerialResponse( "75\n" );

        REQUIRE( app.setFanSpeed() == 0 );
        REQUIRE( app.m_fanSpeedName == "p75" );
        REQUIRE(
            g_edtStubState.serialCommands ==
            std::vector<std::string>{ "set fan mode manual", "set fan speed 75", "fan mode raw", "fan speed raw" } );
    }

    SECTION( "setFanSpeed handles the automatic preset" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_fanSpeedNameSet = "auto";
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "automatic\n" );

        REQUIRE( app.setFanSpeed() == 0 );
        REQUIRE( app.m_fanSpeedName == "auto" );
    }

    SECTION( "setFanSpeed rejects unknown presets before sending commands" )
    {
        cred2Ctrl_test app;

        app.m_fanSpeedNameSet = "mystery";

        REQUIRE( app.setFanSpeed() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setFanSpeed returns an error when automatic mode cannot be enabled" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_fanSpeedNameSet = "auto";
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setFanSpeed() == -1 );
    }

    SECTION( "setFanSpeed returns an error when the manual speed update is rejected" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_fanSpeedNameSet = "p25";
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setFanSpeed() == -1 );
    }

    SECTION( "setFanSpeed returns an error when manual mode cannot be enabled" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_fanSpeedNameSet = "p25";
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setFanSpeed() == -1 );
    }

    SECTION( "setAnalogGain rejects unknown presets and accepts valid ones" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_analogGainNameSet = "bad";
        REQUIRE( app.setAnalogGain() == -1 );

        resetStubState();
        setPoweredOn( app );
        app.m_analogGainNameSet = "high";
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "high\n" );

        REQUIRE( app.setAnalogGain() == 0 );
        REQUIRE( app.m_analogGainName == "high" );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set sensibility high", "sensibility" } );
    }

    SECTION( "setAnalogGain returns an error when the camera rejects the update" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_analogGainNameSet = "med";
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setAnalogGain() == -1 );
    }

    SECTION( "setLED sends the expected on and off commands" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ledStateSet = false;
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "off\n" );

        REQUIRE( app.setLED() == 0 );
        REQUIRE( app.m_ledState == false );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set led off", "led raw" } );
    }

    SECTION( "setLED sends the on command when the LED is enabled" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ledStateSet = true;
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "on\n" );

        REQUIRE( app.setLED() == 0 );
        REQUIRE( app.m_ledState == true );
    }

    SECTION( "setLED returns an error when the requested LED update is rejected" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ledStateSet = true;
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setLED() == -1 );
    }

    SECTION( "setLED returns an error when disabling the LED is rejected" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_ledStateSet = false;
        queueSerialResponse( "Error\n" );

        REQUIRE( app.setLED() == -1 );
    }

    SECTION( "setExpTime remains a no-op" )
    {
        cred2Ctrl_test app;

        REQUIRE( app.setExpTime() == 0 );
    }

    SECTION( "checkNextROI rounds sizes and centers to the hardware step sizes" )
    {
        cred2Ctrl_test app;
        cred2Roi       roi;

        app.m_nextROI.x = 140.0f;
        app.m_nextROI.y = 65.0f;
        app.m_nextROI.w = 250;
        app.m_nextROI.h = 130;

        REQUIRE( app.checkNextROI() == 0 );
        REQUIRE( app.m_nextROI.w == 256 );
        REQUIRE( app.m_nextROI.h == 132 );
        REQUIRE(
            cred2RoiFromCenter(
                roi, app.m_nextROI.x, app.m_nextROI.y, app.m_nextROI.w, app.m_nextROI.h, app.m_full_w, app.m_full_h ) ==
            0 );
        REQUIRE( roi.startColumn % 32 == 0 );
        REQUIRE( roi.startRow % 4 == 0 );
    }

    SECTION( "setNextROI requests a reconfiguration of the current mode" )
    {
        cred2Ctrl_test app;

        app.m_modeName      = "runtime";
        app.m_nextROI.x     = 127.5f;
        app.m_nextROI.y     = 63.5f;
        app.m_nextROI.w     = 256;
        app.m_nextROI.h     = 128;
        app.m_nextROI.bin_x = 1;
        app.m_nextROI.bin_y = 1;

        REQUIRE( app.setNextROI() == 0 );
        REQUIRE( app.state() == stateCodes::CONFIGURING );
        REQUIRE( app.m_reconfig == true );
        REQUIRE( app.m_nextMode == "runtime" );
    }
}

/// Verify framegrabber-facing helpers cover acquisition, copying, and reconfiguration logic.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl framegrabber helpers manage acquisition and ROI configuration", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::configureAcquisition() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::fps() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::startAcquisition() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::acquireAndCheckValid() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::loadImageIntoStream( nullptr ) );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::reconfig() );
    #endif
    // clang-format on

    SECTION( "configureAcquisition handles full-frame and cropped ROI requests" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_cameraCropEnabled = true;
        app.m_fpsSet            = 125.0f;
        app.m_nextROI.x         = app.m_full_x;
        app.m_nextROI.y         = app.m_full_y;
        app.m_nextROI.w         = app.m_full_w;
        app.m_nextROI.h         = app.m_full_h;
        app.m_nextROI.bin_x     = 1;
        app.m_nextROI.bin_y     = 1;

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( app.m_cameraCropEnabled == false );
        REQUIRE( app.m_width == static_cast<uint32_t>( app.m_full_w ) );
        REQUIRE( app.m_height == static_cast<uint32_t>( app.m_full_h ) );
        REQUIRE( app.m_fps == Approx( 125.0f ) );
        REQUIRE( app.m_roiSettleCounter == 5 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set cropping off" } );

        resetStubState();
        setPoweredOn( app );
        app.m_nextROI.x     = 319.5f;
        app.m_nextROI.y     = 255.5f;
        app.m_nextROI.w     = 256;
        app.m_nextROI.h     = 256;
        app.m_nextROI.bin_x = 1;
        app.m_nextROI.bin_y = 1;

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( app.m_cameraCropEnabled == true );
        REQUIRE( app.m_width == 256 );
        REQUIRE( app.m_height == 256 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "set cropping columns 192-447",
                                                                            "set cropping rows 128-383",
                                                                            "set cropping on" } );
    }

    SECTION( "configureAcquisition rejects invalid ROIs" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_nextROI.x = -10;
        app.m_nextROI.y = -10;
        app.m_nextROI.w = 256;
        app.m_nextROI.h = 256;

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "fps and acquisition helpers report the current rate and timestamps" )
    {
        cred2Ctrl_test app;

        app.m_fps      = 250.25f;
        app.m_numBuffs = 6;

        REQUIRE( app.fps() == Approx( 250.25f ) );
        REQUIRE( app.startAcquisition() == 0 );
        REQUIRE( app.state() == stateCodes::OPERATING );
        REQUIRE( g_edtStubState.startImagesCalls == 1 );
        REQUIRE( g_edtStubState.lastStartNumBuffs == 6 );

        g_edtStubState.waitTimeSec  = 12;
        g_edtStubState.waitTimeNsec = 345;

        REQUIRE( app.acquireAndCheckValid() == 0 );
        REQUIRE( app.m_currImageTimestamp.tv_sec == 12 );
        REQUIRE( app.m_currImageTimestamp.tv_nsec == 345 );
        REQUIRE( g_edtStubState.startImageCalls == 1 );
    }

    SECTION( "loadImageIntoStream copies the raw EDT frame into the destination buffer" )
    {
        cred2Ctrl_test          app;
        std::array<uint16_t, 4> source{ 1, 2, 3, 4 };
        std::array<uint16_t, 4> dest{ 0, 0, 0, 0 };

        app.m_image_p  = reinterpret_cast<u_char *>( source.data() );
        app.m_width    = 2;
        app.m_height   = 2;
        app.m_typeSize = sizeof( uint16_t );

        REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
        REQUIRE( dest == source );
    }

    SECTION( "reconfig reloads the runtime EDT configuration and restores READY state" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "reconfig" );
        app.m_modeName = "runtime";
        app.m_nextMode = "runtime";

        REQUIRE( app.reconfig() == 0 );
        REQUIRE( g_edtStubState.readcfgCalls > 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.m_nextMode == "runtime" );
    }

    SECTION( "writeConfig reports file-open failures" )
    {
        cred2Ctrl_test app;

        app.m_configFile = "/tmp/cred2Ctrl_missing_dir/runtime.cfg";

        REQUIRE( app.writeConfig() == -1 );
    }

    SECTION( "configureAcquisition reports failures while disabling full-frame cropping" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_cameraCropEnabled = true;
        app.m_nextROI.x         = app.m_full_x;
        app.m_nextROI.y         = app.m_full_y;
        app.m_nextROI.w         = app.m_full_w;
        app.m_nextROI.h         = app.m_full_h;
        app.m_nextROI.bin_x     = 1;
        app.m_nextROI.bin_y     = 1;
        queueSerialResponse( "Error\n" );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "configureAcquisition reports failures while programming cropped ROI commands" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.m_nextROI.x     = 319.5f;
        app.m_nextROI.y     = 255.5f;
        app.m_nextROI.w     = 256;
        app.m_nextROI.h     = 256;
        app.m_nextROI.bin_x = 1;
        app.m_nextROI.bin_y = 1;
        queueSerialResponse( "OK\n" );
        queueSerialResponse( "Error\n" );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "loadImageIntoStream reports a failure when the configured flip mode is invalid" )
    {
        cred2Ctrl_test          app;
        std::array<uint16_t, 4> dest{ 0, 0, 0, 0 };

        app.m_image_p     = reinterpret_cast<u_char *>( g_edtStubState.waitImage.data() );
        app.m_width       = 2;
        app.m_height      = 2;
        app.m_typeSize    = sizeof( uint16_t );
        app.m_defaultFlip = 999;

        REQUIRE( app.loadImageIntoStream( dest.data() ) == -1 );
    }

    SECTION( "reconfig reports file-write failures before touching the EDT driver" )
    {
        cred2Ctrl_test app;

        app.m_modeName   = "runtime";
        app.m_nextMode   = "runtime";
        app.m_configFile = "/tmp/cred2Ctrl_missing_dir/reconfig.cfg";

        REQUIRE( app.reconfig() == -1 );
    }

    SECTION( "reconfig returns the EDT reload error when pdvReconfig fails" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "reconfig_driver_fail" );
        app.m_modeName               = "runtime";
        app.m_nextMode               = "runtime";
        g_edtStubState.readcfgReturn = -1;

        REQUIRE( app.reconfig() == -1 );
    }
}

/// Verify telemetry wrappers emit their records and helper lifecycle methods reset cached state.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl telemetry wrappers and power-off helpers update cached state", "[cred2Ctrl]" )
{
    cred2Ctrl_test app;

    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::checkRecordTimes() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::recordTelem( static_cast<const MagAOX::logger::cred2_temps *>( nullptr ) ) );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::recordTelem( static_cast<const MagAOX::logger::telem_stdcam *>( nullptr ) ) );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::recordTelem( static_cast<const MagAOX::logger::telem_fgtimings *>( nullptr ) ) );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::recordTemps( true ) );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::onPowerOff() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::whilePowerOff() );
    XWCTEST_DOXYGEN_REF( cred2Ctrl::appShutdown() );
    #endif
    // clang-format on

    app.m_temps.motherboard    = 40.5f;
    app.m_temps.frontend       = 37.0f;
    app.m_temps.powerboard     = 40.25f;
    app.m_temps.snake          = -14.92f;
    app.m_temps.setpoint       = -15.0f;
    app.m_temps.peltier        = 2.29f;
    app.m_temps.heatsink       = 27.5f;
    app.m_modeName             = "runtime";
    app.m_currentROI.x         = 319.5f;
    app.m_currentROI.y         = 255.5f;
    app.m_currentROI.w         = 640;
    app.m_currentROI.h         = 512;
    app.m_currentROI.bin_x     = 1;
    app.m_currentROI.bin_y     = 1;
    app.m_fps                  = 150.0f;
    app.m_ccdTemp              = -14.92f;
    app.m_ccdTempSetpt         = -15.0f;
    app.m_tempControlStatus    = true;
    app.m_tempControlOnTarget  = true;
    app.m_tempControlStatusStr = "ON TARGET";
    app.m_mna                  = 1.0;
    app.m_vara                 = 0.01;
    app.m_mnw                  = 2.0;
    app.m_varw                 = 0.04;
    app.m_mnwa                 = 3.0;
    app.m_varwa                = 0.09;

    SECTION( "recordTelem overloads forward to the typed telemetry recorders" )
    {
        MagAOX::logger::cred2_temps::lastRecord     = { 0, 0 };
        MagAOX::logger::telem_stdcam::lastRecord    = { 0, 0 };
        MagAOX::logger::telem_fgtimings::lastRecord = { 0, 0 };

        REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::cred2_temps *>( nullptr ) ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::cred2_temps::lastRecord ) );

        REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_stdcam *>( nullptr ) ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_stdcam::lastRecord ) );

        REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_fgtimings *>( nullptr ) ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_fgtimings::lastRecord ) );
    }

    SECTION( "checkRecordTimes emits stale telemetry once intervals have elapsed" )
    {
        MagAOX::logger::cred2_temps::lastRecord     = { 0, 0 };
        MagAOX::logger::telem_stdcam::lastRecord    = { 0, 0 };
        MagAOX::logger::telem_fgtimings::lastRecord = { 0, 0 };

        app.m_maxInterval                                 = 10.0;
        static_cast<cred2MagAOXAppT &>( app ).m_loopPause = 0;

        REQUIRE( app.checkRecordTimes() == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::cred2_temps::lastRecord ) );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_stdcam::lastRecord ) );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_fgtimings::lastRecord ) );
    }

    SECTION( "recordTemps honors cached values and the force flag" )
    {
        MagAOX::logger::cred2_temps::lastRecord = { 0, 0 };
        app.m_temps.motherboard                 = 41.5f;
        app.m_temps.frontend                    = 38.0f;
        app.m_temps.powerboard                  = 41.25f;
        app.m_temps.snake                       = -13.92f;
        app.m_temps.setpoint                    = -14.0f;
        app.m_temps.peltier                     = 3.29f;
        app.m_temps.heatsink                    = 28.5f;

        REQUIRE( app.recordTemps( false ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::cred2_temps::lastRecord ) );

        const timespec firstRecord = MagAOX::logger::cred2_temps::lastRecord;

        REQUIRE( app.recordTemps( false ) == 0 );
        REQUIRE( MagAOX::logger::cred2_temps::lastRecord.tv_sec == firstRecord.tv_sec );
        REQUIRE( MagAOX::logger::cred2_temps::lastRecord.tv_nsec == firstRecord.tv_nsec );

        REQUIRE( app.recordTemps( true ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::cred2_temps::lastRecord ) );
    }

    SECTION( "power-off helpers invalidate cached temperatures and leave shutdown clean" )
    {
        app.m_powerOnCounter = -1;
        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.m_powerOnCounter == 0 );
        REQUIRE( app.m_temps.snake == Approx( -999.0f ) );
        REQUIRE( app.m_ccdTemp == Approx( -999.0f ) );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "UNKNOWN" );

        REQUIRE( app.whilePowerOff() == 0 );
        REQUIRE( app.appShutdown() == 0 );
    }
}

/// Verify appLogic covers connection transitions, steady-state refresh, and hardware-loss handling.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl appLogic handles connection and housekeeping flow", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::appLogic() );
    #endif
    // clang-format on

    SECTION( "POWERON returns immediately while the controller waits for the hardware" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 1;
        app.m_powerTargetState                             = 1;
        app.state( stateCodes::POWERON );
        app.m_powerOnCounter                              = 0;
        static_cast<cred2MagAOXAppT &>( app ).m_loopPause = 0;
        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::POWERON );
    }

    SECTION( "NOTCONNECTED returns immediately once the hardware power is already off" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::NOTCONNECTED );
        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    }

    SECTION( "a valid connection refresh transitions from NOTCONNECTED to READY" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::NOTCONNECTED );
        app.m_modeName     = "runtime";
        app.m_startupMode  = "runtime";
        app.m_raw_width    = 640;
        app.m_raw_height   = 512;
        app.m_ccdTempSetpt = -999;
        fgThreadScope fgThread( app );

        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "off\n" );
        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "medium\n" );
        queueSerialResponse( "on\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.m_fps == Approx( 150.5f ) );
        REQUIRE( app.m_fanSpeedName == "auto" );
        REQUIRE( app.m_analogGainName == "med" );
        REQUIRE( app.m_ledState == true );
    }

    SECTION( "a malformed initial fps response leaves the controller in NODEVICE" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::NOTCONNECTED );
        fgThreadScope fgThread( app );
        queueSerialResponse( "bad fps\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::NODEVICE );
    }

    SECTION( "a transport failure during the initial fps probe also leaves the controller in NODEVICE" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::NOTCONNECTED );
        fgThreadScope fgThread( app );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::NODEVICE );
    }

    SECTION( "connected-state sync failures are ignored after power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::CONNECTED );
        fgThreadScope fgThread( app );
        queueSerialResponse( "garbage\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::CONNECTED );
    }

    SECTION( "connected-state sync failures promote the controller to ERROR while power is still on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::CONNECTED );
        fgThreadScope fgThread( app );
        queueSerialResponse( "garbage\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "connected-state refresh failures promote the controller to ERROR while still powered" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::CONNECTED );
        app.m_raw_width  = app.m_full_w;
        app.m_raw_height = app.m_full_h;
        fgThreadScope fgThread( app );
        queueSerialResponse( "off\n" );
        queueSerialResponse( "bad-fps-limit\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "connected-state refresh failures are ignored after power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::CONNECTED );
        app.m_raw_width  = app.m_full_w;
        app.m_raw_height = app.m_full_h;
        fgThreadScope fgThread( app );
        queueSerialResponse( "off\n" );
        queueSerialResponse( "bad-fps-limit\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::CONNECTED );
    }

    SECTION( "connected-state setpoint update failures are logged while the camera is still powered" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::CONNECTED );
        app.m_raw_width  = app.m_full_w;
        app.m_raw_height = app.m_full_h;
        fgThreadScope fgThread( app );

        queueSerialResponse( "off\n" );
        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "medium\n" );
        queueSerialResponse( "on\n" );
        queueSerialResponse( "Error\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "connected-state setpoint update failures are ignored after power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::CONNECTED );
        app.m_raw_width  = app.m_full_w;
        app.m_raw_height = app.m_full_h;
        fgThreadScope fgThread( app );

        queueSerialResponse( "off\n" );
        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "medium\n" );
        queueSerialResponse( "on\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "ready-state housekeeping failures promote the controller to ERROR while powered on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "ready-state housekeeping failures are ignored once power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "ready-state returns immediately when the INDI mutex is already held elsewhere" )
    {
        cred2Ctrl_test    app;
        std::atomic<bool> mutexHeld{ false };

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        std::thread mutexHolder(
            [&app, &mutexHeld]()
            {
                std::lock_guard<std::mutex> lock( app.m_indiMutex );
                mutexHeld.store( true );
                std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
            } );

        while( !mutexHeld.load() )
        {
            std::this_thread::yield();
        }

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( g_edtStubState.serialCommands.empty() );

        mutexHolder.join();
    }

    SECTION( "ready-state update-limit failures are ignored once power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "bad-fps-limit\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "ready-state update-limit failures promote the controller to ERROR while powered on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "bad-fps-limit\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "ready-state fan-query failures promote the controller to ERROR while powered on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "ready-state fan-query failures are ignored once power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "ready-state analog-gain failures are ignored once power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "ready-state analog-gain failures promote the controller to ERROR while powered on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "ready-state LED failures promote the controller to ERROR while powered on" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "medium\n" );
        queueSerialResponse( "unknown\n" );
        queueSerialResponse( "still unknown\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
    }

    SECTION( "ready-state LED failures are ignored once power has already been lost" )
    {
        cred2Ctrl_test app;

        static_cast<cred2MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                             = 0;
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 0;
        fgThreadScope fgThread( app );

        queueSerialResponse( "10.0\n" );
        queueSerialResponse( "500.0\n" );
        queueSerialResponse( "40.50:37.00:40.25:-14.92:2.29:27.50\n" );
        queueSerialResponse( "-15.0\n" );
        queueSerialResponse( "150.5\n" );
        queueSerialResponse( "automatic\n" );
        queueSerialResponse( "medium\n" );
        queueSerialResponse( "unknown\n" );
        queueSerialResponse( "still unknown\n" );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "ready-state ROI settling skips one housekeeping cycle" )
    {
        cred2Ctrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );
        app.m_roiSettleCounter = 2;
        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_roiSettleCounter == 1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }
}

#endif // CRED2CTRL_TEST_SUPPORT_ONLY

} // namespace cred2CtrlTest
} // namespace libXWCTest
