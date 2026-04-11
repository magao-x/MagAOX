/** \file ocam2KCtrl_test.cpp
 * \brief Catch2 tests for the ocam2KCtrl app.
 * \author OpenAI Codex
 *
 * \ingroup ocam2KCtrl_files
 */

/** \defgroup ocam2KCtrl_unit_test ocam2KCtrl Unit Tests
 * \brief Unit tests for the ocam2KCtrl application.
 *
 * \ingroup application_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include <chrono>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define protected public
#include "../ocam2KCtrl.hpp"
#undef protected

using namespace MagAOX::app;

namespace
{

/// Scripted serial response returned by the EDT serial-command stubs.
struct serialResponse
{
    std::string response;               ///< Response returned by the next serial read.
    int         commandResult{ 0 };     ///< Return code from `pdv_serial_command`.
    int         initialWaitResult{ 1 }; ///< Return code from the first `pdv_serial_wait`.
};

/// Shared EDT stub state used to verify acquisition-related calls from `ocam2KCtrl`.
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
        serialResponses.clear();
        serialCommands.clear();
        activeSerialResponse.clear();
        activeSerialReadPending = false;
        activeSerialTransaction = false;
        activeSerialWaitResult  = 0;
        activeSerialWaitServed  = false;
    }

    int                        startImagesCalls{ 0 };   ///< Number of `pdv_start_images` calls observed.
    int                        lastStartNumBuffs{ -1 }; ///< Most recent requested EDT buffer count.
    int                        startImageCalls{ 0 };    ///< Number of `pdv_start_image` calls observed.
    uint                       waitTimeSec{ 0 };        ///< Seconds returned by `pdv_wait_last_image_timed`.
    uint                       waitTimeNsec{ 0 };       ///< Nanoseconds returned by `pdv_wait_last_image_timed`.
    std::array<u_char, 32>     waitImage{};             ///< Raw EDT frame buffer returned to the app.
    int                        readcfgReturn{ 0 };      ///< Return code forced from `pdv_readcfg`.
    std::deque<serialResponse> serialResponses;         ///< Scripted serial responses queued for future commands.
    std::vector<std::string>   serialCommands;          ///< Serial commands issued by the app under test.
    std::string                activeSerialResponse;    ///< Active serial response returned by the current command.
    bool activeSerialReadPending{ false }; ///< Indicates that the next `pdv_serial_read` should return data.
    bool activeSerialTransaction{ false }; ///< Indicates that a serial command is mid-transaction.
    int  activeSerialWaitResult{ 0 };      ///< Return code for the first wait after a command.
    bool activeSerialWaitServed{ false };  ///< Tracks whether the first serial wait has been consumed.
};

/// Shared OCAM SDK stub state used to drive descramble output in tests.
struct ocam2StubState
{
    /// Reset the stub to its default state before each test section.
    void reset()
    {
        lastId     = 0;
        lastRaw    = nullptr;
        lastOutput = nullptr;
        outputImage.clear();
        imageNumber  = 0;
        lastInitMode = OCAM2_NORMAL;
        lastDescrambleFile.clear();
        nextInitId   = 1;
        initReturn   = OCAM2_OK;
        exitCalls    = 0;
        lastExitId   = 0;
        reportedMode = OCAM2_NORMAL;
    }

    ocam2_id             lastId{ 0 };           ///< Most recent OCAM handle passed into `ocam2_descramble`.
    const short         *lastRaw{ nullptr };    ///< Most recent raw source frame passed to `ocam2_descramble`.
    short               *lastOutput{ nullptr }; ///< Most recent output frame passed to `ocam2_descramble`.
    std::vector<int16_t> outputImage;      ///< Pixel values the descramble stub should copy into the output buffer.
    unsigned int         imageNumber{ 0 }; ///< Frame number returned by the descramble stub.
    ocam2_mode           lastInitMode{ OCAM2_NORMAL }; ///< Most recent OCAM SDK mode requested by `ocam2_init`.
    std::string          lastDescrambleFile;           ///< Most recent descramble file passed to `ocam2_init`.
    ocam2_id             nextInitId{ 1 };              ///< OCAM id returned by the next successful `ocam2_init`.
    ocam2_rc             initReturn{ OCAM2_OK };       ///< Return code from `ocam2_init`.
    int                  exitCalls{ 0 };               ///< Number of `ocam2_exit` invocations.
    ocam2_id             lastExitId{ 0 };              ///< Most recent OCAM id passed to `ocam2_exit`.
    ocam2_mode           reportedMode{ OCAM2_NORMAL }; ///< Mode returned by `ocam2_getMode`.
};

/// Global EDT stub state for the `extern "C"` wrappers below.
edtStubState g_edtStubState;

/// Global OCAM SDK stub state for the `extern "C"` wrappers below.
ocam2StubState g_ocam2StubState;

/// Reset all external-library stub state before a test.
void resetStubState()
{
    g_edtStubState.reset();
    g_ocam2StubState.reset();
}

/// Store a frame number into the raw EDT image buffer at the OCAM metadata offset.
[[maybe_unused]] void setStubFrameNumber( unsigned int frameNumber )
{
    reinterpret_cast<int *>( g_edtStubState.waitImage.data() )[OCAM2_IMAGE_NB_OFFSET / 4] =
        static_cast<int>( frameNumber );
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
        return 240;
    }

    int pdv_get_height( PdvDev *pdv_p )
    {
        static_cast<void>( pdv_p );
        return 121;
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

    const char *ocam2_sdkVersion()
    {
        return "stub";
    }

    const char *ocam2_sdkBuild()
    {
        return "stub";
    }

    ocam2_rc ocam2_init( ocam2_mode mode, const char *descrbFile, ocam2_id *id )
    {
        g_ocam2StubState.lastInitMode       = mode;
        g_ocam2StubState.lastDescrambleFile = descrbFile == nullptr ? "" : descrbFile;
        if( id != nullptr )
        {
            *id = g_ocam2StubState.nextInitId;
        }
        g_ocam2StubState.reportedMode = mode;
        return g_ocam2StubState.initReturn;
    }

    void ocam2_descramble( ocam2_id id, unsigned int *number, short *image, const short *imageRaw )
    {
        g_ocam2StubState.lastId     = id;
        g_ocam2StubState.lastRaw    = imageRaw;
        g_ocam2StubState.lastOutput = image;
        if( number != nullptr )
        {
            *number = g_ocam2StubState.imageNumber;
        }
        if( image != nullptr )
        {
            for( size_t nn = 0; nn < g_ocam2StubState.outputImage.size(); ++nn )
            {
                image[nn] = g_ocam2StubState.outputImage[nn];
            }
        }
    }

    ocam2_rc ocam2_exit( ocam2_id id )
    {
        ++g_ocam2StubState.exitCalls;
        g_ocam2StubState.lastExitId = id;
        return OCAM2_OK;
    }

    ocam2_mode ocam2_getMode( ocam2_id id )
    {
        static_cast<void>( id );
        return g_ocam2StubState.reportedMode;
    }

    const char *ocam2_modeStr( ocam2_mode mode )
    {
        static_cast<void>( mode );
        return "stub";
    }

} // extern "C"

namespace libXWCTest
{

/// Namespace for `ocam2KCtrl` unit tests.
/** \ingroup ocam2KCtrl_unit_test
 */
namespace ocam2KCtrlTest
{

namespace
{

/// Build a unique shmim name for one temporary test stream.
std::string uniqueShmimName( const std::string &suffix )
{
    static unsigned counter = 0;

    ++counter;

    return "ocam2KCtrl_test_" + suffix + "_" + std::to_string( ::getpid() ) + "_" + std::to_string( counter );
}

/// Build a unique temporary config-file path for one test.
[[maybe_unused]] std::string uniqueConfigPath( const std::string &suffix )
{
    return "/tmp/ocam2KCtrl_test_" + suffix + "_" + std::to_string( ::getpid() ) + ".conf";
}

/// RAII wrapper for a temporary 1x1 uint8 ImageStreamIO stream used by tests.
class tempStream
{
  public:
    /// Create the temporary stream or throw if ImageStreamIO setup fails.
    explicit tempStream( const std::string &name,
                         uint32_t           width    = 1,
                         uint32_t           height   = 1,
                         uint32_t           depth    = 1,
                         uint8_t            dataType = _DATATYPE_UINT8 )
        : m_name( name )
    {
        uint32_t imsize[3] = { width, height, depth };

        if( ImageStreamIO_createIm_gpu( &m_image,
                                        m_name.c_str(),
                                        3,
                                        imsize,
                                        dataType,
                                        -1,
                                        1,
                                        IMAGE_NB_SEMAPHORE,
                                        0,
                                        CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                        0 ) != IMAGESTREAMIO_SUCCESS )
        {
            throw std::runtime_error( "failed to create temporary ImageStreamIO stream" );
        }

        m_image.md[0].cnt1 = 0;
    }

    /// Destroy the temporary stream.
    ~tempStream()
    {
        if( m_owner )
        {
            ImageStreamIO_destroyIm( &m_image );
        }
    }

    /// Return the underlying temporary stream.
    IMAGE *image()
    {
        return &m_image;
    }

    /// Release destruction ownership after the stream is handed off elsewhere.
    void dismiss()
    {
        m_owner = false;
    }

  private:
    std::string m_name;          ///< Unique shmim name for the temporary stream.
    IMAGE       m_image{};       ///< ImageStreamIO handle for the temporary stream.
    bool        m_owner{ true }; ///< Whether this wrapper should destroy the underlying shmim on teardown.
};

/// Test harness exposing protected `ocam2KCtrl` helpers.
class ocam2KCtrl_test : public MagAOX::app::ocam2KCtrl
{
  public:
    /// Construct a testable controller instance with unique shmim names.
    ocam2KCtrl_test()
    {
        m_configName    = "ocam2KCtrl_test";
        m_shmimName     = uniqueShmimName( "main" );
        m_syncShmimName = uniqueShmimName( "sync" );
    }

    /// Tear down any sync stream left behind by the test.
    ~ocam2KCtrl_test() noexcept
    {
        destroySyncStream();
    }
};

/// Put the app into the nominal powered-on state used by the serial helpers.
[[maybe_unused]] void setPoweredOn( ocam2KCtrl_test &app )
{
    static_cast<MagAOXAppT &>( app ).m_powerState = 1;
    app.m_powerTargetState                        = 1;
}

/// Seed one minimal but internally consistent camera-mode setup for startup and acquisition tests.
[[maybe_unused]] void configureStartupMode( ocam2KCtrl_test   &app,
                                            const std::string &modeName       = "science",
                                            const std::string &serialCommand  = "mode science",
                                            const std::string &configFileName = "stub.cfg" )
{
    dev::cameraConfig config;

    app.m_startupMode                                                  = modeName;
    app.m_modeName                                                     = modeName;
    app.m_configDir                                                    = "/tmp";
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_powerDevice    = "pwr";
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_powerChannel   = "cam";
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_dioDevice      = "dio";
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_sensorChannel  = "sensor";
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_triggerChannel = "trigger";
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_shutterWait    = 0;
    static_cast<dev::dssShutter<ocam2KCtrl> &>( app ).m_shutterTimeout = 0.1;
    app.m_tel.logPath( "/tmp" );
    app.m_tel.logName( app.m_configName );
    app.m_tel.logExt( "bintel" );
    config.m_serialCommand      = serialCommand;
    config.m_configFile         = configFileName;
    config.m_binningX           = 1;
    config.m_binningY           = 1;
    config.m_digitalBinX        = 1;
    config.m_digitalBinY        = 1;
    app.m_cameraModes[modeName] = config;
}

/// Start a short-lived framegrabber thread so `frameGrabber::appLogic()` sees a running worker.
void startFgThread( ocam2KCtrl_test &app, int sleepMs = 200 )
{
    app.m_fgThread =
        std::thread( [sleepMs]() { std::this_thread::sleep_for( std::chrono::milliseconds( sleepMs ) ); } );
}

/// Join the temporary framegrabber thread if a test started one.
void joinFgThread( ocam2KCtrl_test &app )
{
    if( app.m_fgThread.joinable() )
    {
        app.m_fgThread.join();
    }
}

/// Keep a temporary framegrabber thread joined even when a test fails mid-scope.
struct fgThreadScope
{
    ocam2KCtrl_test &m_app; ///< App whose temporary framegrabber thread is managed by this scope.

    /// Start a temporary framegrabber thread for the lifetime of this scope.
    explicit fgThreadScope( ocam2KCtrl_test &app, int sleepMs = 200 ) : m_app( app )
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
    ocam2KCtrl_test &m_app;     ///< App whose startup resources are managed by this scope.
    bool             m_started; ///< Tracks whether `appStartup()` completed successfully in the test.

    /// Construct a startup guard for one test app instance.
    explicit startupScope( ocam2KCtrl_test &app ) : m_app( app ), m_started( false )
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

#ifndef OCAM2KCTRL_TEST_SUPPORT_ONLY

/// Verify the sync stream is created as a 1x1 uint8 ImageStreamIO buffer.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl sync stream creation uses a 1x1 uint8 layout", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;
    IMAGE           openedSyncStream;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::ensureSyncStream() );
    #endif
    // clang-format on

    REQUIRE( app.ensureSyncStream() == 0 );
    REQUIRE( app.m_syncImageStream != nullptr );
    REQUIRE( app.m_syncImageStream->md[0].datatype == _DATATYPE_UINT8 );
    REQUIRE( app.m_syncImageStream->md[0].size[0] == 1 );
    REQUIRE( app.m_syncImageStream->md[0].size[1] == 1 );
    REQUIRE( app.m_syncImageStream->md[0].size[2] == 1 );
    REQUIRE( app.m_syncImageStream->md[0].cnt1 == 0 );

    REQUIRE( ImageStreamIO_openIm( &openedSyncStream, app.m_syncShmimName.c_str() ) == IMAGESTREAMIO_SUCCESS );
    REQUIRE( openedSyncStream.md[0].datatype == _DATATYPE_UINT8 );
    REQUIRE( openedSyncStream.md[0].size[0] == 1 );
    REQUIRE( openedSyncStream.md[0].size[1] == 1 );
    REQUIRE( openedSyncStream.md[0].size[2] == 1 );
    REQUIRE( ImageStreamIO_closeIm( &openedSyncStream ) == IMAGESTREAMIO_SUCCESS );
}

/// Verify the sync stream mirrors main-stream metadata and posts a semaphore.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl sync stream publication mirrors metadata and posts semaphores", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;
    tempStream      sourceStream( uniqueShmimName( "source" ) );
    int             semIndex = -1;
    int             semValue = -1;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::frameGrabberPostPublish( nullptr ) );
    #endif
    // clang-format on

    SECTION( "null source streams are ignored" )
    {
        REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
    }

    SECTION( "publication fails cleanly if the sync stream has not been prepared" )
    {
        REQUIRE( app.frameGrabberPostPublish( sourceStream.image() ) == -1 );
    }

    SECTION( "publication mirrors metadata and posts one semaphore on the sync stream" )
    {
        REQUIRE( app.ensureSyncStream() == 0 );

        sourceStream.image()->md[0].writetime.tv_sec  = 1234;
        sourceStream.image()->md[0].writetime.tv_nsec = 5678;
        sourceStream.image()->md[0].atime.tv_sec      = 1234;
        sourceStream.image()->md[0].atime.tv_nsec     = 4321;
        sourceStream.image()->md[0].cnt0              = 77;
        sourceStream.image()->md[0].cnt1              = 0;
        sourceStream.image()->writetimearray[0]       = sourceStream.image()->md[0].writetime;
        sourceStream.image()->atimearray[0]           = sourceStream.image()->md[0].atime;
        sourceStream.image()->cntarray[0]             = sourceStream.image()->md[0].cnt0;

        semIndex = ImageStreamIO_getsemwaitindex( app.m_syncImageStream, 0 );
        REQUIRE( semIndex >= 0 );
        ImageStreamIO_semflush( app.m_syncImageStream, semIndex );
        REQUIRE( sem_getvalue( app.m_syncImageStream->semptr[semIndex], &semValue ) == 0 );
        REQUIRE( semValue == 0 );

        app.m_syncImageStream->array.UI8[0] = 99;

        REQUIRE( app.frameGrabberPostPublish( sourceStream.image() ) == 0 );
        REQUIRE( sem_getvalue( app.m_syncImageStream->semptr[semIndex], &semValue ) == 0 );
        REQUIRE( semValue == 1 );
        REQUIRE( app.m_syncImageStream->array.UI8[0] == 0 );
        REQUIRE( app.m_syncImageStream->md[0].writetime.tv_sec == sourceStream.image()->md[0].writetime.tv_sec );
        REQUIRE( app.m_syncImageStream->md[0].writetime.tv_nsec == sourceStream.image()->md[0].writetime.tv_nsec );
        REQUIRE( app.m_syncImageStream->md[0].atime.tv_sec == sourceStream.image()->md[0].atime.tv_sec );
        REQUIRE( app.m_syncImageStream->md[0].atime.tv_nsec == sourceStream.image()->md[0].atime.tv_nsec );
        REQUIRE( app.m_syncImageStream->md[0].cnt0 == sourceStream.image()->md[0].cnt0 );
        REQUIRE( app.m_syncImageStream->md[0].cnt1 == 0 );
        REQUIRE( app.m_syncImageStream->writetimearray[0].tv_sec == sourceStream.image()->md[0].writetime.tv_sec );
        REQUIRE( app.m_syncImageStream->atimearray[0].tv_nsec == sourceStream.image()->md[0].atime.tv_nsec );
        REQUIRE( app.m_syncImageStream->cntarray[0] == sourceStream.image()->md[0].cnt0 );
    }
}

/// Verify sync-stream preparation reuses valid streams and recovers from stale or mismatched ones.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl ensureSyncStream reuses and replaces existing stream state", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::ensureSyncStream() );
    #endif
    // clang-format on

    SECTION( "a valid existing sync stream is reused in place" )
    {
        REQUIRE( app.ensureSyncStream() == 0 );

        IMAGE *existingStream = app.m_syncImageStream;

        REQUIRE( app.ensureSyncStream() == 0 );
        REQUIRE( app.m_syncImageStream == existingStream );
    }

    SECTION( "a mismatched in-memory sync stream is destroyed and recreated with the expected layout" )
    {
        app.m_syncImageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
        REQUIRE( app.m_syncImageStream != nullptr );

        {
            uint32_t wrongSize[3] = { 1, 2, 1 };

            REQUIRE( ImageStreamIO_createIm_gpu( app.m_syncImageStream,
                                                 app.m_syncShmimName.c_str(),
                                                 3,
                                                 wrongSize,
                                                 _DATATYPE_UINT8,
                                                 -1,
                                                 1,
                                                 IMAGE_NB_SEMAPHORE,
                                                 0,
                                                 CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                                 0 ) == IMAGESTREAMIO_SUCCESS );
        }

        void *wrongPtr = app.m_syncImageStream;

        REQUIRE( app.ensureSyncStream() == 0 );
        REQUIRE( app.m_syncImageStream != nullptr );
        REQUIRE( static_cast<void *>( app.m_syncImageStream ) == wrongPtr );
        REQUIRE( app.m_syncImageStream->md[0].datatype == _DATATYPE_UINT8 );
        REQUIRE( app.m_syncImageStream->md[0].size[0] == 1 );
        REQUIRE( app.m_syncImageStream->md[0].size[1] == 1 );
        REQUIRE( app.m_syncImageStream->md[0].size[2] == 1 );
        REQUIRE( app.m_syncImageStream->md[0].size[1] != 2 );
    }

    SECTION( "a stale on-disk sync stream is reopened, destroyed, and replaced" )
    {
        tempStream staleStream( app.m_syncShmimName, 1, 1, 1, _DATATYPE_UINT8 );
        staleStream.dismiss();

        REQUIRE( app.m_syncImageStream == nullptr );
        REQUIRE( app.ensureSyncStream() == 0 );
        REQUIRE( app.m_syncImageStream != nullptr );
        REQUIRE( app.m_syncImageStream->md[0].datatype == _DATATYPE_UINT8 );
        REQUIRE( app.m_syncImageStream->md[0].size[0] == 1 );
        REQUIRE( app.m_syncImageStream->md[0].size[1] == 1 );
        REQUIRE( app.m_syncImageStream->md[0].size[2] == 1 );
    }

    SECTION( "an invalid pre-existing sync-stream file returns an error" )
    {
        char syncFileName[1024];
        ImageStreamIO_filename( syncFileName, sizeof( syncFileName ), app.m_syncShmimName.c_str() );

        std::FILE *syncFile = std::fopen( syncFileName, "w" );
        REQUIRE( syncFile != nullptr );
        REQUIRE( std::fputs( "not an ImageStreamIO stream\n", syncFile ) >= 0 );
        REQUIRE( std::fclose( syncFile ) == 0 );

        REQUIRE( app.ensureSyncStream() == -1 );
        REQUIRE( app.m_syncImageStream == nullptr );
        REQUIRE( ::unlink( syncFileName ) == 0 );
    }
}

/// Verify the stdCamera state string is assembled from the OCAM state members.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl stateString reports mode fps gain and setpoint", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::stateString() );
    #endif
    // clang-format on

    app.m_modeName     = "science";
    app.m_fps          = 150.0f;
    app.m_emGain       = 42;
    app.m_ccdTempSetpt = 19.5;

    REQUIRE( app.stateString() == "science_150.000000_42.000000_19.500000" );
}

/// Verify the state string validity depends on operating state and temperature lock.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl stateStringValid requires OPERATING and on-target temperature control", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::stateStringValid() );
    #endif
    // clang-format on

    app.state( stateCodes::OPERATING );
    app.m_tempControlOnTarget = true;
    REQUIRE( app.stateStringValid() );

    app.m_tempControlOnTarget = false;
    REQUIRE( !app.stateStringValid() );

    app.state( stateCodes::READY );
    app.m_tempControlOnTarget = true;
    REQUIRE( !app.stateStringValid() );
}

/// Verify OCAM-specific configuration values load defaults, overrides, and gain clamps.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl configuration loading handles defaults and supported gain limits", "[ocam2KCtrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setupConfig() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::loadConfig() );
    #endif
    // clang-format on

    SECTION( "loadConfig falls back to the main shmim name for the sync stream" )
    {
        ocam2KCtrl_test app;

        app.setupConfig();

        mx::app::writeConfigFile( uniqueConfigPath( "defaults" ), { "unused" }, { "value" }, { "0" } );
        app.config.readConfig( uniqueConfigPath( "defaults" ) );

        app.m_shmimName     = "ocam_main_default";
        app.m_syncShmimName = "";

        app.loadConfig();

        REQUIRE( app.m_ocamDescrambleFile == "" );
        REQUIRE( app.m_maxEMGain == Approx( 600.0f ) );
        REQUIRE( app.m_syncShmimName == app.m_shmimName + "_sync" );
    }

    SECTION( "loadConfig applies configured overrides for the descramble and stream names" )
    {
        ocam2KCtrl_test app;

        app.setupConfig();

        mx::app::writeConfigFile( uniqueConfigPath( "overrides" ),
                                  { "camera", "camera", "framegrabber", "framegrabber" },
                                  { "ocamDescrambleFile", "maxEMGain", "shmimName", "syncShmimName" },
                                  { "descramble_stub.txt", "321", "ocam_main_override", "ocam_sync_override" } );
        app.config.readConfig( uniqueConfigPath( "overrides" ) );

        app.loadConfig();

        REQUIRE( app.m_ocamDescrambleFile == "descramble_stub.txt" );
        REQUIRE( app.m_maxEMGain == Approx( 321.0f ) );
        REQUIRE( app.m_shmimName == "ocam_main_override" );
        REQUIRE( app.m_syncShmimName == "ocam_sync_override" );
    }

    SECTION( "loadConfig clamps maxEMGain below the supported minimum" )
    {
        ocam2KCtrl_test app;

        app.setupConfig();

        mx::app::writeConfigFile( uniqueConfigPath( "gain_low" ), { "camera" }, { "maxEMGain" }, { "0" } );
        app.config.readConfig( uniqueConfigPath( "gain_low" ) );

        app.loadConfig();

        REQUIRE( app.m_maxEMGain == Approx( 1.0f ) );
    }

    SECTION( "loadConfig clamps maxEMGain above the supported maximum" )
    {
        ocam2KCtrl_test app;

        app.setupConfig();

        mx::app::writeConfigFile( uniqueConfigPath( "gain_high" ), { "camera" }, { "maxEMGain" }, { "700" } );
        app.config.readConfig( uniqueConfigPath( "gain_high" ) );

        app.loadConfig();

        REQUIRE( app.m_maxEMGain == Approx( 600.0f ) );
    }
}

/// Verify temperature queries handle valid and malformed serial responses.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl getTemps handles valid and malformed serial responses", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::getTemps() );
    #endif
    // clang-format on

    setPoweredOn( app );

    SECTION( "valid temperature response updates cached temperatures and status" )
    {
        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp" );
        REQUIRE( app.m_temps.CCD == Approx( 20.4f ) );
        REQUIRE( app.m_temps.CPU == Approx( 41.0f ) );
        REQUIRE( app.m_ccdTemp == Approx( 20.4f ) );
        REQUIRE( app.m_ccdTempSetpt == Approx( 20.5f ) );
        REQUIRE( app.m_tempControlStatus == true );
        REQUIRE( app.m_tempControlStatusStr == "ON TARGET" );
        REQUIRE( app.m_tempControlOnTarget == true );
    }

    SECTION( "malformed temperature response marks cached temperatures invalid without forcing reconfig" )
    {
        queueSerialResponse( "Temperatures : CCD[20.4]\n" );

        app.m_temps.CCD            = 5;
        app.m_ccdTemp              = 5;
        app.m_ccdTempSetpt         = 6;
        app.m_tempControlStatus    = true;
        app.m_tempControlOnTarget  = false;
        app.m_tempControlStatusStr = "ON TARGET";

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp" );
        REQUIRE( app.m_temps.CCD == Approx( -999.0f ) );
        REQUIRE( app.m_temps.SET == Approx( -999.0f ) );
        REQUIRE( app.m_ccdTemp == Approx( -999.0f ) );
        REQUIRE( app.m_ccdTempSetpt == Approx( -999.0f ) );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "UNKNOWN" );
        REQUIRE( app.m_tempControlOnTarget == false );
    }

    SECTION( "low cooling power with a warm detector reports temperature control off" )
    {
        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[165]\nCooling Power [2]mW.\n\n" );

        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_temps.CCD == Approx( 20.4f ) );
        REQUIRE( app.m_temps.SET == Approx( 16.5f ) );
        REQUIRE( app.m_tempControlStatus == false );
        REQUIRE( app.m_tempControlStatusStr == "TEMP OFF" );
        REQUIRE( app.m_tempControlOnTarget == false );
    }

    SECTION( "serial failures while powered off return -1 immediately" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;

        queueSerialResponse( "", -1 );

        REQUIRE( app.getTemps() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp" );
    }

    SECTION( "serial failures while powered on return a software error" )
    {
        queueSerialResponse( "", -1 );

        REQUIRE( app.getTemps() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp" );
    }

    SECTION( "malformed temperature responses while powered off return -1" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;

        queueSerialResponse( "Temperatures : CCD[20.4]\n" );

        REQUIRE( app.getTemps() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp" );
    }
}

/// Verify FPS queries handle valid and malformed serial responses, plus synchro mode.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl getFPS handles valid and malformed serial responses", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::getFPS() );
    #endif
    // clang-format on

    SECTION( "valid fps response updates the cached frame rate" )
    {
        setPoweredOn( app );
        app.m_synchro = false;
        app.m_fps     = 0;

        queueSerialResponse( "fps [150.5] Hz\n" );

        REQUIRE( app.getFPS() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps" );
        REQUIRE( app.m_fps == Approx( 150.5f ) );
    }

    SECTION( "malformed fps response is non-fatal and leaves the cached value unchanged" )
    {
        setPoweredOn( app );
        app.m_synchro = false;
        app.m_fps     = 75.0f;

        queueSerialResponse( "fps 150.5\n" );

        REQUIRE( app.getFPS() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps" );
        REQUIRE( app.m_fps == Approx( 75.0f ) );
    }

    SECTION( "malformed fps responses while powered off return -1" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_synchro                                 = false;
        app.m_fps                                     = 75.0f;

        queueSerialResponse( "fps 150.5\n" );

        REQUIRE( app.getFPS() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps" );
        REQUIRE( app.m_fps == Approx( 75.0f ) );
    }

    SECTION( "serial failures while powered on return a software error" )
    {
        setPoweredOn( app );
        app.m_synchro = false;
        app.m_fps     = 75.0f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.getFPS() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps" );
        REQUIRE( app.m_fps == Approx( 75.0f ) );
    }

    SECTION( "synchro mode bypasses serial queries and mirrors the sync frequency" )
    {
        app.m_synchro  = true;
        app.m_syncFreq = 205.25f;
        app.m_fps      = 0;

        REQUIRE( app.getFPS() == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
        REQUIRE( app.m_fps == Approx( 205.25f ) );
    }
}

/// Verify the temperature-control helpers handle safe, unsafe, and valid setpoint requests.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl temperature control helpers handle valid and invalid requests", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setTempControl() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setTempSetPt() );
    #endif
    // clang-format on

    setPoweredOn( app );

    SECTION( "setTempControl refuses to turn cooling off below the safe temperature" )
    {
        app.m_tempControlStatusSet = false;
        app.m_tempControlStatus    = true;
        app.m_ccdTemp              = 18.5f;

        REQUIRE( app.setTempControl() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
        REQUIRE( app.m_tempControlStatus == true );
    }

    SECTION( "setTempControl turns cooling off when the detector is warm enough" )
    {
        app.m_tempControlStatusSet = false;
        app.m_tempControlStatus    = true;
        app.m_ccdTemp              = 20.0f;

        queueSerialResponse( "temperature control off\n" );

        REQUIRE( app.setTempControl() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp off" );
        REQUIRE( app.m_tempControlStatusSet == false );
        REQUIRE( app.m_tempControlStatus == false );
    }

    SECTION( "setTempControl turns cooling on and reapplies the configured setpoint" )
    {
        app.m_tempControlStatusSet = true;
        app.m_tempControlStatus    = false;
        app.m_ccdTempSetpt         = 15.5f;

        queueSerialResponse( "temperature control on\n" );
        queueSerialResponse( "setpoint updated\n" );

        REQUIRE( app.setTempControl() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 2 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp on" );
        REQUIRE( g_edtStubState.serialCommands[1] == "temp 15.500000" );
        REQUIRE( app.m_tempControlStatusSet == true );
        REQUIRE( app.m_tempControlStatus == true );
    }

    SECTION( "setTempControl returns -1 on serial failure once power is already off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_tempControlStatusSet                    = true;
        app.m_tempControlStatus                       = false;
        app.m_ccdTempSetpt                            = 15.5f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setTempControl() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp on" );
    }

    SECTION( "setTempControl returns a software error on serial failure while still powered" )
    {
        app.m_tempControlStatusSet = true;
        app.m_tempControlStatus    = false;
        app.m_ccdTempSetpt         = 15.5f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setTempControl() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp on" );
    }

    SECTION( "setTempSetPt rejects out-of-range high setpoints without serial traffic" )
    {
        app.m_ccdTempSetpt = 30.0f;

        REQUIRE( app.setTempSetPt() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setTempSetPt rejects out-of-range low setpoints without serial traffic" )
    {
        app.m_ccdTempSetpt = -50.1f;

        REQUIRE( app.setTempSetPt() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setTempSetPt sends valid setpoints to the camera" )
    {
        app.m_ccdTempSetpt = 12.25f;

        queueSerialResponse( "setpoint updated\n" );

        REQUIRE( app.setTempSetPt() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp 12.250000" );
    }

    SECTION( "setTempSetPt returns -1 on serial failure once power is already off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_ccdTempSetpt                            = 12.25f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setTempSetPt() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp 12.250000" );
    }

    SECTION( "setTempSetPt returns a software error on serial failure while still powered" )
    {
        app.m_ccdTempSetpt = 12.25f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setTempSetPt() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "temp 12.250000" );
    }
}

/// Verify the shutter adapter rejects invalid requests before touching the DSS threads.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl setShutter rejects invalid shutter requests", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setShutter( 0 ) );
    #endif
    // clang-format on

    REQUIRE( app.setShutter( 2 ) == -1 );
}

/// Verify simple helper methods update local app state without hardware dependencies.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl helper interfaces reset local state and power-off lifecycle flags", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::powerOnDefaults() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setExpTime() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setNextROI() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::fps() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::onPowerOff() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::whilePowerOff() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::appShutdown() );
    #endif
    // clang-format on

    app.m_tempControlStatusSet = true;
    app.m_tempControlStatus    = true;
    REQUIRE( app.powerOnDefaults() == 0 );
    REQUIRE( app.m_tempControlStatusSet == false );
    REQUIRE( app.m_tempControlStatus == false );

    app.m_fps = 250.5f;
    REQUIRE( app.fps() == Approx( 250.5f ) );

    REQUIRE( app.setExpTime() == 0 );
    REQUIRE( app.setNextROI() == 0 );

    REQUIRE( app.ensureSyncStream() == 0 );
    REQUIRE( app.m_syncImageStream != nullptr );

    app.m_powerOnCounter = -1;
    app.m_poweredOn      = false;
    app.m_width          = 17;
    app.m_height         = 19;
    app.m_circBuffLength = 23;
    app.m_reconfig       = false;

    REQUIRE( app.onPowerOff() == 0 );
    REQUIRE( app.m_powerOnCounter == 0 );
    REQUIRE( app.m_poweredOn == true );
    REQUIRE( app.m_width == 0 );
    REQUIRE( app.m_height == 0 );
    REQUIRE( app.m_circBuffLength == 1 );
    REQUIRE( app.m_reconfig == true );

    REQUIRE( app.whilePowerOff() == 0 );

    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( app.m_syncImageStream == nullptr );
}

/// Verify acquisition startup requests EDT buffering and resets the image counter.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl startAcquisition resets frame tracking and starts EDT buffers", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::startAcquisition() );
    #endif
    // clang-format on

    app.m_numBuffs        = 6;
    app.m_lastImageNumber = 99;

    REQUIRE( app.startAcquisition() == 0 );
    REQUIRE( app.m_lastImageNumber == -1 );
    REQUIRE( g_edtStubState.startImagesCalls == 1 );
    REQUIRE( g_edtStubState.lastStartNumBuffs == 6 );
}

/// Verify frame acquisition timestamps and frame-number handling across valid and invalid sequences.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl acquireAndCheckValid handles valid, skipped, and corrupt frame numbers", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::acquireAndCheckValid() );
    #endif
    // clang-format on

    static_cast<MagAOXAppT &>( app ).m_powerState = 1;
    app.m_powerTargetState                        = 1;
    app.m_modeName                                = "science";

    SECTION( "first valid frame initializes the previous-frame tracker" )
    {
        resetStubState();

        g_edtStubState.waitTimeSec  = 12;
        g_edtStubState.waitTimeNsec = 345;
        setStubFrameNumber( 101 );

        app.m_lastImageNumber = -1;

        REQUIRE( app.acquireAndCheckValid() == 0 );
        REQUIRE( app.m_currImageTimestamp.tv_sec == 12 );
        REQUIRE( app.m_currImageTimestamp.tv_nsec == 345 );
        REQUIRE( app.m_currImageNumber == 101 );
        REQUIRE( app.m_lastImageNumber == 101 );
        REQUIRE( g_edtStubState.startImageCalls == 1 );
    }

    SECTION( "small skipped-frame gaps request reconfiguration" )
    {
        resetStubState();

        setStubFrameNumber( 15 );

        app.m_lastImageNumber = 12;
        app.m_nextMode        = "";
        app.m_reconfig        = false;

        REQUIRE( app.acquireAndCheckValid() == 1 );
        REQUIRE( app.m_lastImageNumber == -1 );
        REQUIRE( app.m_nextMode == "science" );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "large frame-number jumps on powered hardware are treated as corruption" )
    {
        resetStubState();

        setStubFrameNumber( 500 );

        app.m_lastImageNumber = 12;
        app.m_nextMode        = "";
        app.m_reconfig        = false;

        REQUIRE( app.acquireAndCheckValid() == 1 );
        REQUIRE( app.m_lastImageNumber == -1 );
        REQUIRE( app.m_nextMode == "science" );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "large frame-number jumps during power loss return an error instead of reconfiguring" )
    {
        resetStubState();

        setStubFrameNumber( 500 );

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_lastImageNumber                         = 12;
        app.m_nextMode                                = "";
        app.m_reconfig                                = false;

        REQUIRE( app.acquireAndCheckValid() == -1 );
        REQUIRE( app.m_nextMode == "" );
        REQUIRE( app.m_reconfig == false );
    }
}

/// Verify the raw image pointer is passed through to the OCAM descramble routine.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl loadImageIntoStream uses the OCAM descramble output", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test        app;
    std::array<int16_t, 4> rawImage{ 1, 2, 3, 4 };
    std::array<int16_t, 4> destImage{ 0, 0, 0, 0 };

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::loadImageIntoStream( nullptr ) );
    #endif
    // clang-format on

    SECTION( "the non-digital path writes the descrambled frame directly to the destination" )
    {
        app.m_ocam2_id   = 7;
        app.m_digitalBin = false;
        app.m_image_p    = reinterpret_cast<u_char *>( rawImage.data() );

        g_ocam2StubState.imageNumber = 44;
        g_ocam2StubState.outputImage = { 10, 20, 30, 40 };

        REQUIRE( app.loadImageIntoStream( destImage.data() ) == 0 );
        REQUIRE( g_ocam2StubState.lastId == 7 );
        REQUIRE( g_ocam2StubState.lastRaw == rawImage.data() );
        REQUIRE( g_ocam2StubState.lastOutput == destImage.data() );
        REQUIRE( destImage[0] == 10 );
        REQUIRE( destImage[1] == 20 );
        REQUIRE( destImage[2] == 30 );
        REQUIRE( destImage[3] == 40 );
    }

    SECTION( "the digital-binning path descrambles into the work image before filling the output frame" )
    {
        app.m_ocam2_id    = 8;
        app.m_digitalBin  = true;
        app.m_digitalBinX = 2;
        app.m_digitalBinY = 1;
        app.m_width       = 2;
        app.m_height      = 2;
        app.m_image_p     = reinterpret_cast<u_char *>( rawImage.data() );
        app.m_digitalBinWork.resize( 4, 2 );

        destImage.fill( 0 );
        g_ocam2StubState.imageNumber = 55;
        g_ocam2StubState.outputImage = { 7, 7, 7, 7, 7, 7, 7, 7 };

        REQUIRE( app.loadImageIntoStream( destImage.data() ) == 0 );
        REQUIRE( g_ocam2StubState.lastId == 8 );
        REQUIRE( g_ocam2StubState.lastRaw == rawImage.data() );
        REQUIRE( g_ocam2StubState.lastOutput == app.m_digitalBinWork.data() );
        REQUIRE( destImage[0] == 7 );
        REQUIRE( destImage[1] == 7 );
        REQUIRE( destImage[2] == 7 );
        REQUIRE( destImage[3] == 7 );
    }
}

/// Verify the serial gain helpers accept valid responses and handle malformed or tripped ones.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl serial gain helpers handle valid and invalid responses", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::getEMGain() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setEMGain() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::resetEMProtection() );
    #endif
    // clang-format on

    setPoweredOn( app );

    SECTION( "valid EM gain response updates the cached value" )
    {
        app.m_emGain = 1;

        queueSerialResponse( "Gain set to 42 \n\n" );

        REQUIRE( app.getEMGain() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain" );
        REQUIRE( app.m_emGain == 42 );
    }

    SECTION( "malformed EM gain response returns an error and keeps the previous value" )
    {
        app.m_emGain = 77;

        queueSerialResponse( "Gain set to \n\n" );

        REQUIRE( app.getEMGain() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain" );
        REQUIRE( app.m_emGain == 77 );
    }

    SECTION( "malformed EM gain responses while powered off return -1 immediately" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_emGain                                  = 77;

        queueSerialResponse( "Gain set to \n\n" );

        REQUIRE( app.getEMGain() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain" );
        REQUIRE( app.m_emGain == 77 );
    }

    SECTION( "HV trip response forces the cached EM gain back to the safe minimum" )
    {
        app.m_emGain = 77;

        queueSerialResponse( "HV trip\n" );

        REQUIRE( app.getEMGain() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain" );
        REQUIRE( app.m_emGain == 1 );
    }

    SECTION( "resetEMProtection marks the protection state as reset" )
    {
        app.m_protectionReset          = false;
        app.m_protectionResetConfirmed = 2;

        queueSerialResponse( "Protection reset\n" );

        REQUIRE( app.resetEMProtection() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "protection reset" );
        REQUIRE( app.m_protectionReset == true );
        REQUIRE( app.m_protectionResetConfirmed == 0 );
    }

    SECTION( "resetEMProtection returns a software error while still powered" )
    {
        app.m_indiP_emProt = pcf::IndiProperty( pcf::IndiProperty::Text );
        app.m_indiP_emProt.add( pcf::IndiElement( "status" ) );
        app.m_indiP_emProt["status"].set( "CONFIRM" );

        queueSerialResponse( "", -1 );

        REQUIRE( app.resetEMProtection() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "protection reset" );
    }

    SECTION( "resetEMProtection returns -1 once power is already off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;

        queueSerialResponse( "", -1 );

        REQUIRE( app.resetEMProtection() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "protection reset" );
    }

    SECTION( "setEMGain refuses unsafe requests before protection reset" )
    {
        app.m_protectionReset = false;
        app.m_emGainSet       = 10;

        REQUIRE( app.setEMGain() == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setEMGain refuses out-of-range requests without serial traffic" )
    {
        app.m_protectionReset = true;
        app.m_maxEMGain       = 100;
        app.m_emGainSet       = 200;

        REQUIRE( app.setEMGain() == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setEMGain sends the requested gain after protection reset" )
    {
        app.m_protectionReset = true;
        app.m_maxEMGain       = 600;
        app.m_emGainSet       = 25;

        queueSerialResponse( "Gain set to 25 \n\n" );

        REQUIRE( app.setEMGain() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain 25" );
    }

    SECTION( "setEMGain returns -1 on serial failure once power is already off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_protectionReset                         = true;
        app.m_maxEMGain                               = 600;
        app.m_emGainSet                               = 25;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setEMGain() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain 25" );
    }

    SECTION( "setEMGain returns a software error on serial failure while still powered" )
    {
        app.m_protectionReset = true;
        app.m_maxEMGain       = 600;
        app.m_emGainSet       = 25;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setEMGain() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "gain 25" );
    }
}

/// Verify the serial setter commands send the expected OCAM command sequence.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl serial setter commands send the expected sequence", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setFPS() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setSynchro() );
    #endif
    // clang-format on

    setPoweredOn( app );

    SECTION( "setFPS sends the requested rate and queues a reconfiguration" )
    {
        app.m_synchro  = false;
        app.m_fpsSet   = 250.5f;
        app.m_modeName = "science";
        app.m_nextMode = "";
        app.m_reconfig = false;

        queueSerialResponse( "fps updated\n" );

        REQUIRE( app.setFPS() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 250.500000" );
        REQUIRE( app.m_nextMode == "science" );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "setFPS returns a software error on serial failure while still powered" )
    {
        app.m_synchro = false;
        app.m_fpsSet  = 250.5f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setFPS() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 250.500000" );
    }

    SECTION( "setFPS uses the sync-device property path when synchro is enabled" )
    {
        app.m_synchro = true;
        app.m_fpsSet  = 88.5f;

        REQUIRE( app.setFPS() == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "setFPS returns -1 on serial failure once power is already off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_synchro                                 = false;
        app.m_fpsSet                                  = 250.5f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setFPS() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 250.500000" );
    }

    SECTION( "setSynchro off uses the OCAM serial path for both synchro and FPS" )
    {
        app.m_synchroSet    = false;
        app.m_fpsSet        = 175.0f;
        app.m_indiP_synchro = pcf::IndiProperty( pcf::IndiProperty::Switch );
        app.m_indiP_synchro.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );

        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.setSynchro() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 3 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[1] == "synchro off" );
        REQUIRE( g_edtStubState.serialCommands[2] == "fps 175.000000" );
        REQUIRE( app.m_synchro == false );
    }

    SECTION( "setSynchro on updates the local synchro state and leaves FPS to the sync device" )
    {
        app.m_synchroSet    = true;
        app.m_fpsSet        = 175.0f;
        app.m_indiP_synchro = pcf::IndiProperty( pcf::IndiProperty::Switch );
        app.m_indiP_synchro.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );

        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro on\n" );

        REQUIRE( app.setSynchro() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 2 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[1] == "synchro on" );
        REQUIRE( app.m_synchro == true );
    }

    SECTION( "setSynchro returns a software error when the initial fps-max command fails on powered hardware" )
    {
        app.m_synchroSet = false;
        app.m_fpsSet     = 175.0f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setSynchro() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 0" );
    }

    SECTION( "setSynchro returns a software error when the synchro command fails on powered hardware" )
    {
        app.m_synchroSet = true;
        app.m_fpsSet     = 175.0f;

        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.setSynchro() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 2 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[1] == "synchro on" );
    }

    SECTION( "setSynchro returns -1 when the initial fps-max command fails while powered off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_synchroSet                              = false;
        app.m_fpsSet                                  = 175.0f;

        queueSerialResponse( "", -1 );

        REQUIRE( app.setSynchro() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 0" );
    }

    SECTION( "setSynchro returns -1 when the synchro command fails while powered off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_synchroSet                              = true;
        app.m_fpsSet                                  = 175.0f;

        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.setSynchro() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 2 );
        REQUIRE( g_edtStubState.serialCommands[0] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[1] == "synchro on" );
    }
}

/// Verify acquisition configuration programs the OCAM mode and handles invalid frame shapes.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl configureAcquisition handles valid and invalid OCAM modes", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test   app;
    dev::cameraConfig config;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::configureAcquisition() );
    #endif
    // clang-format on

    setPoweredOn( app );

    app.m_configDir              = "/tmp";
    app.m_ocamDescrambleFile     = "ocam_descramble_stub.txt";
    app.m_modeName               = "science";
    config.m_serialCommand       = "mode science";
    config.m_binningX            = 1;
    config.m_binningY            = 1;
    config.m_digitalBinX         = 2;
    config.m_digitalBinY         = 3;
    app.m_cameraModes["science"] = config;

    SECTION( "configureAcquisition sets up the SDK, digital binning, and sync stream" )
    {
        app.m_raw_height = 121;
        app.m_fpsSet     = 50.0f;
        app.m_synchroSet = false;
        app.m_ocam2_id   = 9;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 5 );
        REQUIRE( g_edtStubState.serialCommands[0] == "mode science" );
        REQUIRE( g_edtStubState.serialCommands[1] == "fps 50.000000" );
        REQUIRE( g_edtStubState.serialCommands[2] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[3] == "synchro off" );
        REQUIRE( g_edtStubState.serialCommands[4] == "fps 50.000000" );
        REQUIRE( g_ocam2StubState.exitCalls == 1 );
        REQUIRE( g_ocam2StubState.lastExitId == 9 );
        REQUIRE( g_ocam2StubState.lastInitMode == OCAM2_NORMAL );
        REQUIRE( g_ocam2StubState.lastDescrambleFile == "/tmp/ocam_descramble_stub.txt" );
        REQUIRE( app.m_ocam2_id == 1 );
        REQUIRE( app.m_currentROI.x == Approx( 119.5 ) );
        REQUIRE( app.m_currentROI.y == Approx( 119.5 ) );
        REQUIRE( app.m_currentROI.w == Approx( 240.0 ) );
        REQUIRE( app.m_currentROI.h == Approx( 240.0 ) );
        REQUIRE( app.m_currentROI.bin_x == 1 );
        REQUIRE( app.m_currentROI.bin_y == 1 );
        REQUIRE( app.m_digitalBin == true );
        REQUIRE( app.m_width == 120 );
        REQUIRE( app.m_height == 80 );
        REQUIRE( app.m_dataType == _DATATYPE_UINT16 );
        REQUIRE( app.m_syncImageStream != nullptr );
        REQUIRE( app.state() == stateCodes::OPERATING );
    }

    SECTION( "configureAcquisition keeps the full OCAM frame size when digital binning is disabled" )
    {
        app.m_raw_height                           = 121;
        app.m_fpsSet                               = 0.0f;
        app.m_synchroSet                           = false;
        app.m_cameraModes["science"].m_digitalBinX = 1;
        app.m_cameraModes["science"].m_digitalBinY = 1;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( g_ocam2StubState.lastInitMode == OCAM2_NORMAL );
        REQUIRE( app.m_digitalBin == false );
        REQUIRE( app.m_width == 240 );
        REQUIRE( app.m_height == 240 );
    }

    SECTION( "configureAcquisition selects OCAM binning mode for 62-row raw frames" )
    {
        app.m_raw_height                           = 62;
        app.m_fpsSet                               = 0.0f;
        app.m_synchroSet                           = false;
        app.m_cameraModes["science"].m_digitalBinX = 1;
        app.m_cameraModes["science"].m_digitalBinY = 1;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( g_ocam2StubState.lastInitMode == OCAM2_BINNING );
        REQUIRE( app.m_width == 120 );
        REQUIRE( app.m_height == 120 );
    }

    SECTION( "configureAcquisition selects OCAM 1x3 binning mode for 41-row raw frames" )
    {
        app.m_raw_height                           = 41;
        app.m_fpsSet                               = 0.0f;
        app.m_synchroSet                           = false;
        app.m_cameraModes["science"].m_digitalBinX = 1;
        app.m_cameraModes["science"].m_digitalBinY = 1;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( g_ocam2StubState.lastInitMode == OCAM2_BINNING1x3 );
        REQUIRE( app.m_width == 240 );
        REQUIRE( app.m_height == 80 );
    }

    SECTION( "configureAcquisition selects OCAM 1x4 binning mode for 31-row raw frames" )
    {
        app.m_raw_height                           = 31;
        app.m_fpsSet                               = 0.0f;
        app.m_synchroSet                           = false;
        app.m_cameraModes["science"].m_digitalBinX = 1;
        app.m_cameraModes["science"].m_digitalBinY = 1;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( g_ocam2StubState.lastInitMode == OCAM2_BINNING1x4 );
        REQUIRE( app.m_width == 240 );
        REQUIRE( app.m_height == 60 );
    }

    SECTION( "configureAcquisition reports a set-mode serial failure on powered hardware" )
    {
        app.m_raw_height = 121;
        app.m_fpsSet     = 0.0f;
        app.m_synchroSet = false;

        queueSerialResponse( "", -1 );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "mode science" );
    }

    SECTION( "configureAcquisition returns -1 quietly once power is already off during the set-mode command" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_raw_height                              = 121;
        app.m_fpsSet                                  = 0.0f;
        app.m_synchroSet                              = false;

        queueSerialResponse( "", -1 );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "mode science" );
    }

    SECTION( "configureAcquisition logs but continues when setSynchro fails" )
    {
        app.m_raw_height                           = 121;
        app.m_fpsSet                               = 0.0f;
        app.m_synchroSet                           = false;
        app.m_cameraModes["science"].m_digitalBinX = 1;
        app.m_cameraModes["science"].m_digitalBinY = 1;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "", -1 );

        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 3 );
        REQUIRE( g_edtStubState.serialCommands[0] == "mode science" );
        REQUIRE( g_edtStubState.serialCommands[1] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[2] == "synchro off" );
        REQUIRE( g_ocam2StubState.lastInitMode == OCAM2_NORMAL );
    }

    SECTION( "configureAcquisition returns -1 if OCAM initialization fails after power is lost" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_raw_height                              = 121;
        app.m_fpsSet                                  = 0.0f;
        app.m_synchroSet                              = false;
        app.m_cameraModes["science"].m_digitalBinX    = 1;
        app.m_cameraModes["science"].m_digitalBinY    = 1;
        g_ocam2StubState.initReturn                   = OCAM2_ERROR;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( app.m_syncImageStream == nullptr );
    }

    SECTION( "configureAcquisition returns -1 if OCAM initialization fails on powered hardware" )
    {
        app.m_raw_height                           = 121;
        app.m_fpsSet                               = 0.0f;
        app.m_synchroSet                           = false;
        app.m_cameraModes["science"].m_digitalBinX = 1;
        app.m_cameraModes["science"].m_digitalBinY = 1;
        g_ocam2StubState.initReturn                = OCAM2_ERROR;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( app.m_syncImageStream == nullptr );
    }

    SECTION( "configureAcquisition rejects unsupported raw frame heights" )
    {
        app.m_raw_height = 100;
        app.m_fpsSet     = 0.0f;
        app.m_synchroSet = false;

        queueSerialResponse( "mode set\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );

        REQUIRE( app.configureAcquisition() == -1 );
        REQUIRE( g_edtStubState.serialCommands.size() == 4 );
        REQUIRE( g_edtStubState.serialCommands[0] == "mode science" );
        REQUIRE( g_edtStubState.serialCommands[1] == "fps 0" );
        REQUIRE( g_edtStubState.serialCommands[2] == "synchro off" );
        REQUIRE( g_edtStubState.serialCommands[3] == "fps 0.000000" );
        REQUIRE( g_ocam2StubState.exitCalls == 0 );
        REQUIRE( g_ocam2StubState.lastDescrambleFile.empty() );
        REQUIRE( app.m_syncImageStream == nullptr );
    }
}

/// Verify the INDI callbacks update confirmation state and sync-frequency-driven reconfiguration.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl INDI callbacks update local state", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::newCallBack_m_indiP_emProtReset( pcf::IndiProperty() ) );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::setCallBack_m_indiP_syncFreq( pcf::IndiProperty() ) );
    #endif
    // clang-format on

    setPoweredOn( app );

    SECTION( "EM protection reset callback requires a confirmation before issuing the serial reset" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Switch );

        app.m_indiP_emProtReset.setDevice( "ocam2KCtrl" );
        app.m_indiP_emProtReset.setName( "emProtReset" );

        ipRecv.setDevice( "ocam2KCtrl" );
        ipRecv.setName( "emProtReset" );
        ipRecv.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

        REQUIRE( app.newCallBack_m_indiP_emProtReset( ipRecv ) == 0 );
        REQUIRE( app.m_protectionResetConfirmed == 1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );

        queueSerialResponse( "Protection reset\n" );

        REQUIRE( app.newCallBack_m_indiP_emProtReset( ipRecv ) == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() == 1 );
        REQUIRE( g_edtStubState.serialCommands[0] == "protection reset" );
        REQUIRE( app.m_protectionReset == true );
        REQUIRE( app.m_protectionResetConfirmed == 0 );
    }

    SECTION( "EM protection reset callback ignores requests while power is off" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Switch );

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.m_indiP_emProtReset.setDevice( "ocam2KCtrl" );
        app.m_indiP_emProtReset.setName( "emProtReset" );
        ipRecv.setDevice( "ocam2KCtrl" );
        ipRecv.setName( "emProtReset" );
        ipRecv.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

        REQUIRE( app.newCallBack_m_indiP_emProtReset( ipRecv ) == 0 );
        REQUIRE( app.m_protectionResetConfirmed == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "EM protection reset callback rejects the wrong property name" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Switch );

        app.m_indiP_emProtReset.setDevice( "ocam2KCtrl" );
        app.m_indiP_emProtReset.setName( "emProtReset" );
        ipRecv.setDevice( "ocam2KCtrl" );
        ipRecv.setName( "wrongName" );
        ipRecv.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

        REQUIRE( app.newCallBack_m_indiP_emProtReset( ipRecv ) == -1 );
    }

    SECTION( "EM protection reset callback ignores requests without the request element or with it off" )
    {
        pcf::IndiProperty missingReq( pcf::IndiProperty::Switch );
        pcf::IndiProperty offReq( pcf::IndiProperty::Switch );

        app.m_indiP_emProtReset.setDevice( "ocam2KCtrl" );
        app.m_indiP_emProtReset.setName( "emProtReset" );

        missingReq.setDevice( "ocam2KCtrl" );
        missingReq.setName( "emProtReset" );
        offReq.setDevice( "ocam2KCtrl" );
        offReq.setName( "emProtReset" );
        offReq.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );

        REQUIRE( app.newCallBack_m_indiP_emProtReset( missingReq ) == 0 );
        REQUIRE( app.newCallBack_m_indiP_emProtReset( offReq ) == 0 );
        REQUIRE( app.m_protectionResetConfirmed == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "sync frequency callback queues a reconfiguration when synchro mode changes the effective FPS" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Number );

        app.m_indiP_syncFreq.setDevice( "syncDevice" );
        app.m_indiP_syncFreq.setName( "C1freq" );

        ipRecv.setDevice( "syncDevice" );
        ipRecv.setName( "C1freq" );
        ipRecv.add( pcf::IndiElement( "current" ) );
        ipRecv["current"] = 123.4;

        app.m_synchro  = true;
        app.m_fps      = 100.0f;
        app.m_modeName = "science";
        app.m_nextMode = "";
        app.m_reconfig = false;

        REQUIRE( app.setCallBack_m_indiP_syncFreq( ipRecv ) == 0 );
        REQUIRE( app.m_syncFreq == Approx( 123.4f ) );
        REQUIRE( app.m_fps == Approx( 123.4f ) );
        REQUIRE( app.m_nextMode == "science" );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "sync frequency callback rejects updates that do not include the current element" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Number );

        app.m_indiP_syncFreq.setDevice( "syncDevice" );
        app.m_indiP_syncFreq.setName( "C1freq" );
        ipRecv.setDevice( "syncDevice" );
        ipRecv.setName( "C1freq" );

        REQUIRE( app.setCallBack_m_indiP_syncFreq( ipRecv ) == -1 );
    }
}

/// Verify the generated static INDI wrappers forward into the instance callbacks.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl static INDI callback wrappers forward requests to the instance", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::st_newCallBack_m_indiP_emProtReset( nullptr, pcf::IndiProperty() ) );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::st_setCallBack_m_indiP_syncFreq( nullptr, pcf::IndiProperty() ) );
    #endif
    // clang-format on

    setPoweredOn( app );

    SECTION( "the new-property wrapper forwards EM protection reset requests" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Switch );

        app.m_indiP_emProtReset.setDevice( "ocam2KCtrl" );
        app.m_indiP_emProtReset.setName( "emProtReset" );

        ipRecv.setDevice( "ocam2KCtrl" );
        ipRecv.setName( "emProtReset" );
        ipRecv.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

        REQUIRE( ocam2KCtrl::st_newCallBack_m_indiP_emProtReset( &app, ipRecv ) == 0 );
        REQUIRE( app.m_protectionResetConfirmed == 1 );
    }

    SECTION( "the set-property wrapper forwards sync frequency updates" )
    {
        pcf::IndiProperty ipRecv( pcf::IndiProperty::Number );

        app.m_indiP_syncFreq.setDevice( "syncDevice" );
        app.m_indiP_syncFreq.setName( "C1freq" );

        ipRecv.setDevice( "syncDevice" );
        ipRecv.setName( "C1freq" );
        ipRecv.add( pcf::IndiElement( "current" ) );
        ipRecv["current"] = 222.5;

        app.m_synchro  = true;
        app.m_fps      = 100.0f;
        app.m_modeName = "science";

        REQUIRE( ocam2KCtrl::st_setCallBack_m_indiP_syncFreq( &app, ipRecv ) == 0 );
        REQUIRE( app.m_syncFreq == Approx( 222.5f ) );
        REQUIRE( app.m_fps == Approx( 222.5f ) );
        REQUIRE( app.m_nextMode == "science" );
        REQUIRE( app.m_reconfig == true );
    }
}

/// Verify telemetry wrapper helpers emit their records and interval checks trigger stale telemetry.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl telemetry wrappers record snapshots and stale intervals", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test app;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::checkRecordTimes() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::recordTelem( static_cast<const MagAOX::logger::ocam_temps *>( nullptr ) ) );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::recordTelem( static_cast<const MagAOX::logger::telem_stdcam *>( nullptr ) ) );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::recordTelem( static_cast<const MagAOX::logger::telem_fgtimings *>( nullptr ) ) );
    #endif
    // clang-format on

    app.m_temps.CCD            = 20.1f;
    app.m_temps.CPU            = 41.0f;
    app.m_temps.POWER          = 34.0f;
    app.m_temps.BIAS           = 47.0f;
    app.m_temps.WATER          = 24.2f;
    app.m_temps.LEFT           = 33.0f;
    app.m_temps.RIGHT          = 38.0f;
    app.m_temps.COOLING_POWER  = 102.0f;
    app.m_modeName             = "science";
    app.m_currentROI.x         = 119.5;
    app.m_currentROI.y         = 119.5;
    app.m_currentROI.w         = 240.0;
    app.m_currentROI.h         = 240.0;
    app.m_currentROI.bin_x     = 1;
    app.m_currentROI.bin_y     = 1;
    app.m_fps                  = 150.0f;
    app.m_emGain               = 25.0f;
    app.m_ccdTemp              = 20.1f;
    app.m_ccdTempSetpt         = 20.5f;
    app.m_tempControlStatus    = true;
    app.m_tempControlOnTarget  = true;
    app.m_tempControlStatusStr = "ON TARGET";
    app.m_shutterStatus        = "open";
    app.m_shutterState         = 1;
    app.m_synchro              = false;
    app.m_mna                  = 1.0;
    app.m_vara                 = 0.01;
    app.m_mnw                  = 2.0;
    app.m_varw                 = 0.04;
    app.m_mnwa                 = 3.0;
    app.m_varwa                = 0.09;

    SECTION( "recordTelem overloads forward directly to the per-type telemetry recorders" )
    {
        MagAOX::logger::ocam_temps::lastRecord      = { 0, 0 };
        MagAOX::logger::telem_stdcam::lastRecord    = { 0, 0 };
        MagAOX::logger::telem_fgtimings::lastRecord = { 0, 0 };

        REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::ocam_temps *>( nullptr ) ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::ocam_temps::lastRecord ) );

        REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_stdcam *>( nullptr ) ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_stdcam::lastRecord ) );

        REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_fgtimings *>( nullptr ) ) == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_fgtimings::lastRecord ) );
    }

    SECTION( "checkRecordTimes emits all telemetry records when their intervals have elapsed" )
    {
        MagAOX::logger::ocam_temps::lastRecord      = { 0, 0 };
        MagAOX::logger::telem_stdcam::lastRecord    = { 0, 0 };
        MagAOX::logger::telem_fgtimings::lastRecord = { 0, 0 };

        app.m_maxInterval                            = 10.0;
        static_cast<MagAOXAppT &>( app ).m_loopPause = 0;

        REQUIRE( app.checkRecordTimes() == 0 );
        REQUIRE( hasRecordedTime( MagAOX::logger::ocam_temps::lastRecord ) );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_stdcam::lastRecord ) );
        REQUIRE( hasRecordedTime( MagAOX::logger::telem_fgtimings::lastRecord ) );
    }
}

/// Verify appLogic covers connection transitions, error handling, and READY-state housekeeping.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl appLogic handles connection and housekeeping flow", "[ocam2KCtrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::appLogic() );
    #endif
    // clang-format on

    SECTION( "NOTCONNECTED returns early during power loss" )
    {
        ocam2KCtrl_test app;

        app.state( stateCodes::NOTCONNECTED );
        app.m_temps.CCD = 12.0f;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::NOTCONNECTED );
        REQUIRE( app.m_temps.CCD == Approx( -999.0f ) );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "POWEROFF falls through to the final return without camera traffic" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.state( stateCodes::POWEROFF );
        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::POWEROFF );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "NOTCONNECTED success falls through the READY housekeeping path in the same loop" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );

        app.state( stateCodes::NOTCONNECTED );
        app.m_fpsSet              = 0.0f;
        app.m_poweredOn           = true;
        app.m_ccdTempSetpt        = 18.5f;
        app.m_tempControlOnTarget = false;
        app.m_indiP_emProt        = pcf::IndiProperty( pcf::IndiProperty::Text );
        app.m_indiP_emProt.setDevice( "ocam2KCtrl" );
        app.m_indiP_emProt.setName( "emProtReset" );
        app.m_indiP_emProt.add( pcf::IndiElement( "status" ) );
        app.m_indiP_emProt["status"].set( "CONFIRMED" );
        app.m_protectionResetConfirmed = 1;
        app.m_protectionResetReqTime   = mx::sys::get_curr_time() - 11.0;

        queueSerialResponse( "camera online\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "setpoint updated\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps restored\n" );
        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[185]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "Gain set to 42 \n\n" );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::FAILURE );
        REQUIRE( app.m_fps == Approx( 150.5f ) );
        REQUIRE( app.m_poweredOn == false );
        REQUIRE( app.m_synchroSet == false );
        REQUIRE( app.m_protectionResetConfirmed == 0 );
        REQUIRE( app.m_shutdown == 1 );
        REQUIRE( g_edtStubState.serialCommands ==
                 std::vector<std::string>{
                     "fps", "fps", "temp 18.500000", "fps 0", "synchro off", "fps 0.000000", "temp", "fps", "gain" } );
    }

    SECTION( "NOTCONNECTED returns after a failed connectivity probe while power remains on" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::NOTCONNECTED );

        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::NOTCONNECTED );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps" } );
    }

    SECTION( "CONNECTED with a requested FPS transitions through OPERATING before housekeeping" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );

        app.state( stateCodes::CONNECTED );
        app.m_fpsSet                   = 10.0f;
        app.m_poweredOn                = false;
        app.m_protectionResetConfirmed = 0;

        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps 10.000000 restored\n" );
        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "Gain set to 42 \n\n" );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::FAILURE );
        REQUIRE( app.m_shutdown == 1 );
        REQUIRE( g_edtStubState.serialCommands ==
                 std::vector<std::string>{ "fps", "fps 0", "synchro off", "fps 10.000000", "temp", "fps", "gain" } );
    }

    SECTION( "CONNECTED enters ERROR when getFPS fails on powered hardware" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::CONNECTED );

        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps" } );
    }

    SECTION( "CONNECTED returns quietly if getFPS fails after power is lost" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.state( stateCodes::CONNECTED );

        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::CONNECTED );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps" } );
    }

    SECTION( "CONNECTED returns quietly when setTempSetPt fails after power is lost" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.state( stateCodes::CONNECTED );
        app.m_fpsSet       = 0.0f;
        app.m_poweredOn    = true;
        app.m_ccdTempSetpt = 18.5f;

        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.m_poweredOn == false );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps", "temp 18.500000" } );
    }

    SECTION( "CONNECTED returns a software error when setTempSetPt fails on powered hardware" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::CONNECTED );
        app.m_fpsSet       = 0.0f;
        app.m_poweredOn    = true;
        app.m_ccdTempSetpt = 18.5f;

        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.m_poweredOn == false );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps", "temp 18.500000" } );
    }

    SECTION( "CONNECTED logs but continues when setSynchro fails during the initial connect" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::CONNECTED );
        app.m_fpsSet    = 0.0f;
        app.m_poweredOn = false;

        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "fps max\n" );
        queueSerialResponse( "", -1 );
        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "Gain set to 42 \n\n" );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::FAILURE );
        REQUIRE( app.m_shutdown == 1 );
        REQUIRE( g_edtStubState.serialCommands ==
                 std::vector<std::string>{ "fps", "fps 0", "synchro off", "temp", "fps", "gain" } );
    }

    SECTION( "READY expires stale protection resets and completes housekeeping before telemetry shutdown" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );

        app.state( stateCodes::READY );
        app.m_protectionResetConfirmed = 1;
        app.m_protectionResetReqTime   = mx::sys::get_curr_time() - 11.0;
        app.m_indiP_emProt             = pcf::IndiProperty( pcf::IndiProperty::Text );
        app.m_indiP_emProt.add( pcf::IndiElement( "status" ) );
        app.m_indiP_emProt["status"].set( "CONFIRM" );

        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "Gain set to 42 \n\n" );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_protectionResetConfirmed == 0 );
        REQUIRE( app.m_shutdown == 1 );
        REQUIRE( app.state() == stateCodes::FAILURE );
        REQUIRE( app.m_temps.CCD == Approx( 20.4f ) );
        REQUIRE( app.m_fps == Approx( 150.5f ) );
        REQUIRE( app.m_emGain == 42 );
    }

    SECTION( "READY returns immediately if the INDI mutex is already locked elsewhere" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );

        std::unique_lock<std::mutex> hold( app.m_indiMutex );
        fgThreadScope                fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "READY moves to ERROR when temperature polling fails on powered hardware" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );

        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( app.m_temps.CCD == Approx( -999.0f ) );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "READY returns quietly when temperature polling fails after power is lost" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.state( stateCodes::READY );

        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "READY moves to ERROR when FPS polling fails after temperatures succeed" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );

        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "fps" } );
    }

    SECTION( "READY returns quietly when FPS polling fails after power is lost" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.state( stateCodes::READY );

        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "fps" } );
    }

    SECTION( "READY returns quietly when EM gain polling fails after power is lost" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.m_powerTargetState                        = 0;
        app.state( stateCodes::READY );

        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "fps", "gain" } );
    }

    SECTION( "READY moves to ERROR when EM gain polling fails on powered hardware" )
    {
        ocam2KCtrl_test app;

        setPoweredOn( app );
        app.state( stateCodes::READY );

        queueSerialResponse( "Temperatures : CCD[20.4] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[205]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [150.5] Hz\n" );
        queueSerialResponse( "", -1 );

        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "fps", "gain" } );
    }
}

/// Verify reconfiguration reloads the requested mode and leaves the app in READY.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl reconfig reloads the next mode through edtCamera", "[ocam2KCtrl]" )
{
    ocam2KCtrl_test   app;
    dev::cameraConfig config;

    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::reconfig() );
    #endif
    // clang-format on

    config.m_configFile          = "stub.cfg";
    app.m_cameraModes["science"] = config;
    app.m_nextMode               = "science";
    app.state( stateCodes::OPERATING );

    REQUIRE( app.reconfig() == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.m_nextMode == "" );
    REQUIRE( app.m_modeName == "science" );
    REQUIRE( app.m_raw_width == 240 );
    REQUIRE( app.m_raw_height == 121 );
    REQUIRE( app.m_raw_depth == 16 );
    REQUIRE( app.m_cameraType == "stub_pdv" );
    REQUIRE( app.fps() == Approx( app.m_fps ) );
}

#endif // OCAM2KCTRL_TEST_SUPPORT_ONLY

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
