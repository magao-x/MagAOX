/** \file streamWriter_lifecycle_test.cpp
 * \brief Catch2 lifecycle and configuration tests for the streamWriter app.
 * \author OpenAI Codex
 *
 * \ingroup streamWriter_files
 */

#include "../../../tests/testXWC.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#define protected public
#include "../streamWriter.hpp"
#undef protected

using namespace MagAOX::app;

namespace libXWCTest
{

/** \addtogroup streamWriter_unit_test
 * \brief Additional lifecycle tests for the streamWriter application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `streamWriter` lifecycle unit tests.
/** \ingroup streamWriter_unit_test
 */
namespace streamWriterTest
{

namespace
{

/// Build a unique shmim name for one temporary streamWriter test stream.
std::string uniqueShmimName( const std::string &suffix )
{
    static unsigned counter = 0;

    ++counter;

    return "streamWriter_test_" + suffix + "_" + std::to_string( ::getpid() ) + "_" + std::to_string( counter );
}

/// Point ImageStreamIO shared-memory files at a writable test sandbox.
std::string ensureMilkShmDir()
{
    static const std::string shmDir = []()
    {
        const std::filesystem::path path = "/tmp/streamWriter_lifecycle_test/shm";

        std::filesystem::create_directories( path );
        return path.string();
    }();

    setenv( "MILK_SHM_DIR", shmDir.c_str(), 1 );

    return shmDir;
}

/// Configuration values used to build a deterministic streamWriter test config.
struct streamWriterConfig
{
    size_t                     m_maxCircBuffLength{ 8 };         ///< Configured circular-buffer length.
    double                     m_maxCircBuffSize{ 16.0 };        ///< Configured circular-buffer size in MB.
    size_t                     m_maxWriteChunkLength{ 4 };       ///< Configured write-chunk length.
    double                     m_maxChunkTime{ 0.5 };            ///< Configured max chunk time in seconds.
    double                     m_writeStopTimeout{ 1.0 };        ///< Configured stop-writing flush timeout.
    bool                       m_startWriting{ false };          ///< Whether writing should start armed at startup.
    int                        m_writerThreadPrio{ 0 };          ///< Configured writer thread priority.
    std::string                m_writerCpuset;                   ///< Optional writer cpuset.
    bool                       m_compress{ true };               ///< Whether XRIF compression is enabled.
    int                        m_lz4accel{ XRIF_LZ4_ACCEL_MIN }; ///< Configured LZ4 acceleration.
    std::optional<std::string> m_outName;                        ///< Optional explicit output name.
    std::optional<std::string> m_savePath;                       ///< Optional explicit save directory.
    std::string                m_shmimName{ "streamWriter_test_stream" }; ///< Shared-memory stream name.
    int                        m_semaphoreNumber{ 5 };                    ///< Shared-memory semaphore index.
    unsigned                   m_semWaitNSec{ 1000000 };                  ///< Semaphore timeout in nanoseconds.
    bool                       m_warnMissedData{ false };                 ///< Whether backlog summaries log warnings.
    int                        m_framegrabberThreadPrio{ 0 };             ///< Configured framegrabber thread priority.
    std::string                m_framegrabberCpuset;                      ///< Optional framegrabber cpuset.
    double                     m_telemeterMaxInterval{ 60.0 };            ///< Telemetry cadence in seconds.
};

/// RAII wrapper for a temporary uint16 ImageStreamIO stream used by `fgThreadExec()` tests.
class tempStream
{
  public:
    /// Create the temporary stream or throw if ImageStreamIO setup fails.
    explicit tempStream( const std::string &name,
                         uint32_t           width    = 1,
                         uint32_t           height   = 1,
                         uint32_t           depth    = 1,
                         uint8_t            dataType = _DATATYPE_UINT16 )
        : m_name( name )
    {
        uint32_t imsize[3] = { width, height, depth };

        ensureMilkShmDir();

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

        m_image.md[0].cnt0 = 0;
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

    /// Return the number of pixels in one frame.
    size_t pixelsPerFrame() const
    {
        return static_cast<size_t>( m_image.md[0].size[0] ) * static_cast<size_t>( m_image.md[0].size[1] );
    }

    /// Return the configured stream depth.
    size_t depth() const
    {
        return static_cast<size_t>( m_image.md[0].size[2] );
    }

    /// Fill one frame with a deterministic ramp, update the metadata, and post the shmim semaphore.
    void publishFrame(
        size_t frameIndex, uint64_t cnt0, uint16_t baseValue, const timespec &atime, const timespec &writetime )
    {
        uint16_t *frame = reinterpret_cast<uint16_t *>( m_image.array.raw ) + frameIndex * pixelsPerFrame();

        for( size_t n = 0; n < pixelsPerFrame(); ++n )
        {
            frame[n] = baseValue + n;
        }

        m_image.md[0].write     = 1;
        m_image.md[0].cnt0      = cnt0;
        m_image.md[0].cnt1      = frameIndex;
        m_image.md[0].atime     = atime;
        m_image.md[0].writetime = writetime;

        m_image.cntarray[frameIndex]       = cnt0;
        m_image.atimearray[frameIndex]     = atime;
        m_image.writetimearray[frameIndex] = writetime;

        m_image.md[0].write = 0;
        ImageStreamIO_sempost( &m_image, -1 );
    }

  private:
    std::string m_name;          ///< Unique shmim name for the temporary stream.
    IMAGE       m_image{};       ///< ImageStreamIO handle for the temporary stream.
    bool        m_owner{ true }; ///< Whether this wrapper should destroy the underlying shmim on teardown.
};

/// Test harness exposing path setup and shutdown cleanup helpers.
class streamWriterLifecycleTest : public streamWriter
{
  public:
    /// Prepare an isolated MagAO-X directory tree for the named test case.
    std::filesystem::path prepareSandbox( const std::string &name )
    {
        std::error_code       ec;
        std::filesystem::path root = std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / name;

        std::filesystem::remove_all( root, ec );
        std::filesystem::create_directories( root / "config" );
        std::filesystem::create_directories( root / "calib" );
        std::filesystem::create_directories( root / "logs" );
        std::filesystem::create_directories( root / "sys" );
        std::filesystem::create_directories( root / "secrets" );
        std::filesystem::create_directories( root / "telem" );
        std::filesystem::create_directories( root / "cpuset" );

        m_basePath    = root.string();
        m_configDir   = ( root / "config" ).string();
        m_calibDir    = ( root / "calib" ).string();
        m_sysPath     = ( root / "sys" ).string();
        m_secretsPath = ( root / "secrets" ).string();
        m_cpusetPath  = ( root / "cpuset" ).string();

        m_configName = name;
        m_outName    = name;

        m_log.logPath( ( root / "logs" ).string() );

        return root;
    }

    /// Shut the worker threads down after a successful startup sequence.
    int shutdownStartedApp()
    {
        m_shutdown = 1;
        return appShutdown();
    }

    /// Prepare the direct `fgThreadExec()` harness resources without starting the writer thread.
    int initializeFgHarness()
    {
        if( sem_init( &m_swSemaphore, 0, 0 ) < 0 )
        {
            return -1;
        }

        m_swSemaphoreInitialized = true;

        return initialize_xrif();
    }

    /// Launch the production `fgThreadExec()` loop in the existing framegrabber thread slot.
    void startFgHarnessThread()
    {
        m_fgThreadInit = false;
        m_shutdown     = 0;
        m_restart      = false;

        m_fgThread = std::thread( &streamWriterLifecycleTest::fgThreadEntry, this );
    }

    /// Stop the direct `fgThreadExec()` harness and release any resources it owns.
    void stopFgHarness()
    {
        m_writing      = NOT_WRITING;
        m_writePending = false;
        m_restart      = false;
        m_shutdown     = 1;

        if( m_fgThread.joinable() )
        {
            m_fgThread.join();
        }

        release_circbufs();

        if( m_swSemaphoreInitialized )
        {
            sem_destroy( &m_swSemaphore );
            m_swSemaphoreInitialized = false;
        }

        if( m_xrif )
        {
            xrif_delete( m_xrif );
            m_xrif = nullptr;
        }

        if( m_xrif_timing )
        {
            xrif_delete( m_xrif_timing );
            m_xrif_timing = nullptr;
        }
    }

    /// Return the current writer-semaphore value for assertions.
    int writerSemaphoreValue()
    {
        int value = 0;
        sem_getvalue( &m_swSemaphore, &value );
        return value;
    }

    /// Drain the writer semaphore and return the number of queued posts removed.
    int drainWriterSemaphore()
    {
        int drained = 0;

        while( sem_trywait( &m_swSemaphore ) == 0 )
        {
            ++drained;
        }

        return drained;
    }

  private:
    /// Thread entry trampoline for the direct framegrabber harness.
    static void fgThreadEntry( streamWriterLifecycleTest *app )
    {
        app->fgThreadExec();
    }

    bool m_swSemaphoreInitialized{ false }; ///< Whether the direct harness initialized `m_swSemaphore`.
};

/// Cleanup helper that only shuts the app down if startup completed.
class startupScope
{
  public:
    /// Track a started `streamWriter` instance for cleanup.
    explicit startupScope( streamWriterLifecycleTest &app ) : m_app( app )
    {
    }

    /// Mark whether `appStartup()` succeeded.
    void markStarted( bool started )
    {
        m_started = started;
    }

    /// Stop tracking once explicit shutdown has already been performed.
    void disarm()
    {
        m_started = false;
    }

    /// Shut the app down if it was started successfully.
    ~startupScope()
    {
        if( m_started )
        {
            static_cast<void>( m_app.shutdownStartedApp() );
        }
    }

  private:
    streamWriterLifecycleTest &m_app;              ///< App instance being protected.
    bool                       m_started{ false }; ///< Whether startup completed successfully.
};

/// Cleanup helper that stops the direct framegrabber harness on scope exit.
class fgHarnessScope
{
  public:
    /// Track a direct `fgThreadExec()` harness instance for cleanup.
    explicit fgHarnessScope( streamWriterLifecycleTest &app ) : m_app( app )
    {
    }

    /// Mark whether the direct harness was initialized and needs teardown.
    void markActive( bool active )
    {
        m_active = active;
    }

    /// Stop tracking once teardown has already been handled explicitly.
    void disarm()
    {
        m_active = false;
    }

    /// Stop the harness if it is still active at scope exit.
    ~fgHarnessScope()
    {
        if( m_active )
        {
            m_app.stopFgHarness();
        }
    }

  private:
    streamWriterLifecycleTest &m_app;             ///< App instance being protected.
    bool                       m_active{ false }; ///< Whether the direct harness needs teardown.
};

/// Write and load a deterministic streamWriter config for the provided test app.
std::filesystem::path loadConfig( streamWriterLifecycleTest &app,
                                  const std::string         &name,
                                  const streamWriterConfig  &cfg = streamWriterConfig() )
{
    const std::filesystem::path root       = app.prepareSandbox( name );
    const std::filesystem::path configPath = root / "config" / ( name + ".conf" );

    app.setupConfig();

    std::vector<std::string> sections{ "writer",
                                       "writer",
                                       "writer",
                                       "writer",
                                       "writer",
                                       "writer",
                                       "writer",
                                       "writer",
                                       "writer",
                                       "framegrabber",
                                       "framegrabber",
                                       "framegrabber",
                                       "framegrabber",
                                       "framegrabber",
                                       "telemeter" };

    std::vector<std::string> keys{ "maxCircBuffLength",
                                   "maxCircBuffSize",
                                   "maxWriteChunkLength",
                                   "maxChunkTime",
                                   "stopTimeout",
                                   "startWriting",
                                   "threadPrio",
                                   "compress",
                                   "lz4accel",
                                   "shmimName",
                                   "semaphoreNumber",
                                   "semWait",
                                   "warnMissedData",
                                   "threadPrio",
                                   "maxInterval" };

    std::vector<std::string> values{ std::to_string( cfg.m_maxCircBuffLength ),
                                     std::to_string( cfg.m_maxCircBuffSize ),
                                     std::to_string( cfg.m_maxWriteChunkLength ),
                                     std::to_string( cfg.m_maxChunkTime ),
                                     std::to_string( cfg.m_writeStopTimeout ),
                                     cfg.m_startWriting ? "1" : "0",
                                     std::to_string( cfg.m_writerThreadPrio ),
                                     cfg.m_compress ? "1" : "0",
                                     std::to_string( cfg.m_lz4accel ),
                                     cfg.m_shmimName,
                                     std::to_string( cfg.m_semaphoreNumber ),
                                     std::to_string( cfg.m_semWaitNSec ),
                                     cfg.m_warnMissedData ? "1" : "0",
                                     std::to_string( cfg.m_framegrabberThreadPrio ),
                                     std::to_string( cfg.m_telemeterMaxInterval ) };

    if( !cfg.m_writerCpuset.empty() )
    {
        sections.push_back( "writer" );
        keys.push_back( "cpuset" );
        values.push_back( cfg.m_writerCpuset );
    }

    if( !cfg.m_framegrabberCpuset.empty() )
    {
        sections.push_back( "framegrabber" );
        keys.push_back( "cpuset" );
        values.push_back( cfg.m_framegrabberCpuset );
    }

    if( cfg.m_outName )
    {
        sections.push_back( "writer" );
        keys.push_back( "outName" );
        values.push_back( *cfg.m_outName );
    }

    if( cfg.m_savePath )
    {
        sections.push_back( "writer" );
        keys.push_back( "savePath" );
        values.push_back( *cfg.m_savePath );
    }

    mx::app::writeConfigFile( configPath.string(), sections, keys, values );

    app.config.readConfig( configPath.string() );
    app.loadConfig();

    return root;
}

/// Wait for a test predicate to become true within a fixed timeout.
template <typename PredicateT>
bool waitFor( PredicateT predicate, int timeoutMs = 2000 )
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds( timeoutMs );

    while( std::chrono::steady_clock::now() < deadline )
    {
        if( predicate() )
        {
            return true;
        }

        std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
    }

    return predicate();
}

/// Return the current realtime clock value for timestamp-driven framegrabber tests.
timespec currentRealtime()
{
    timespec ts{ 0, 0 };

    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        throw std::runtime_error( "clock_gettime failed while preparing streamWriter test timestamps" );
    }

    return ts;
}

/// Return a timestamp offset by the requested nanoseconds.
timespec offsetTimespec( const timespec &base, long long nanoseconds )
{
    timespec shifted = base;

    shifted.tv_sec += nanoseconds / 1000000000LL;
    shifted.tv_nsec += nanoseconds % 1000000000LL;

    while( shifted.tv_nsec >= 1000000000L )
    {
        shifted.tv_nsec -= 1000000000L;
        ++shifted.tv_sec;
    }

    while( shifted.tv_nsec < 0 )
    {
        shifted.tv_nsec += 1000000000L;
        --shifted.tv_sec;
    }

    return shifted;
}

/// Return one uint16 pixel from the frame stored in the streamWriter circular buffer.
uint16_t rawFrameWord( const streamWriterLifecycleTest &app, size_t frameIndex, size_t pixelIndex )
{
    return reinterpret_cast<uint16_t *>( app.m_rawImageCircBuff )[frameIndex * app.m_width * app.m_height + pixelIndex];
}

/// Return one timing word from the streamWriter timing circular buffer.
uint64_t timingWord( const streamWriterLifecycleTest &app, size_t frameIndex, size_t timingIndex )
{
    return app.m_timingCircBuff[frameIndex * 5 + timingIndex];
}

/// Copy the currently scheduled raw-image save window for assertions before a restart frees the buffers.
std::vector<uint16_t> copyPendingRawFrames( const streamWriterLifecycleTest &app )
{
    const size_t pixelsPerFrame = app.m_width * app.m_height;
    const size_t nFrames        = app.m_currSaveStop - app.m_currSaveStart;
    auto *rawStart = reinterpret_cast<uint16_t *>( app.m_rawImageCircBuff ) + app.m_currSaveStart * pixelsPerFrame;

    return std::vector<uint16_t>( rawStart, rawStart + nFrames * pixelsPerFrame );
}

/// Copy the currently scheduled timing save window for assertions before a restart frees the buffers.
std::vector<uint64_t> copyPendingTimingFrames( const streamWriterLifecycleTest &app )
{
    const size_t nFrames     = app.m_currSaveStop - app.m_currSaveStart;
    auto        *timingStart = app.m_timingCircBuff + app.m_currSaveStart * 5;

    return std::vector<uint64_t>( timingStart, timingStart + nFrames * 5 );
}

} // namespace

/// Verify `setupConfig()` and `loadConfig()` preserve defaults, overrides, and clamp invalid accelerations.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter configuration loads defaults and overrides", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::setupConfig() );
    XWCTEST_DOXYGEN_REF( streamWriter::loadConfig() );
    #endif
    // clang-format on

    SECTION( "default-derived paths and names come from the shared-memory stream" )
    {
        streamWriterLifecycleTest app;
        streamWriterConfig        cfg;

        cfg.m_lz4accel = XRIF_LZ4_ACCEL_MIN - 5;

        const std::filesystem::path root = loadConfig( app, "load_defaults", cfg );

        std::string expectedRawRel = mx::sys::getEnv( MAGAOX_env_rawimage );
        if( expectedRawRel.empty() )
        {
            expectedRawRel = MAGAOX_rawimageRelPath;
        }

        REQUIRE( app.m_maxCircBuffLength == 8 );
        REQUIRE( app.m_maxCircBuffSize == Approx( 16.0 ) );
        REQUIRE( app.m_maxWriteChunkLength == 4 );
        REQUIRE( app.m_maxChunkTime == Approx( 0.5 ) );
        REQUIRE( app.m_writeStopTimeout == Approx( 1.0 ) );
        REQUIRE( app.m_startWriting == false );
        REQUIRE( app.m_shmimName == "streamWriter_test_stream" );
        REQUIRE( app.m_outName == app.m_shmimName );
        REQUIRE( app.m_rawimageDir == ( root / expectedRawRel ).string() );
        REQUIRE( app.m_semaphoreNumber == 5 );
        REQUIRE( app.m_semWaitNSec == 1000000 );
        REQUIRE( app.m_warnMissedData == false );
        REQUIRE( app.m_swThreadPrio == 0 );
        REQUIRE( app.m_fgThreadPrio == 0 );
        REQUIRE( app.m_lz4accel == XRIF_LZ4_ACCEL_MIN );
        REQUIRE( app.m_shutdown == 0 );
    }

    SECTION( "explicit overrides replace the defaults and high accelerations clamp to the XRIF limit" )
    {
        streamWriterLifecycleTest app;
        streamWriterConfig        cfg;

        cfg.m_outName  = "camsci";
        cfg.m_savePath = ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test_override" ) / "science" ).string();
        cfg.m_compress = false;
        cfg.m_lz4accel = XRIF_LZ4_ACCEL_MAX + 17;
        cfg.m_startWriting           = true;
        cfg.m_writeStopTimeout       = 0.25;
        cfg.m_writerThreadPrio       = 3;
        cfg.m_framegrabberThreadPrio = 2;

        loadConfig( app, "load_override", cfg );

        REQUIRE( app.m_outName == "camsci" );
        REQUIRE( app.m_rawimageDir == *cfg.m_savePath );
        REQUIRE( app.m_compress == false );
        REQUIRE( app.m_lz4accel == XRIF_LZ4_ACCEL_MAX );
        REQUIRE( app.m_startWriting == true );
        REQUIRE( app.m_writeStopTimeout == Approx( 0.25 ) );
        REQUIRE( app.m_swThreadPrio == 3 );
        REQUIRE( app.m_fgThreadPrio == 2 );
    }
}

/// Verify `appStartup()`, `appLogic()`, and `appShutdown()` cover the basic streamWriter lifecycle.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter lifecycle handles startup validation and nominal shutdown", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::appStartup() );
    XWCTEST_DOXYGEN_REF( streamWriter::appLogic() );
    XWCTEST_DOXYGEN_REF( streamWriter::appShutdown() );
    #endif
    // clang-format on

    SECTION( "appStartup fails when the save directory cannot be created" )
    {
        streamWriterLifecycleTest app;

        const std::filesystem::path root    = loadConfig( app, "startup_bad_save_path" );
        const std::filesystem::path blocker = root / "save_parent_file";

        std::ofstream blockerOut( blocker.string() );
        blockerOut << "block";
        blockerOut.close();

        app.m_rawimageDir = ( blocker / "child" ).string();

        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup fails when the write chunk length is not a divisor of the circular buffer length" )
    {
        streamWriterLifecycleTest app;

        loadConfig( app, "startup_bad_chunk_divisor" );
        app.m_maxCircBuffLength   = 10;
        app.m_maxWriteChunkLength = 4;

        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup, appLogic, and appShutdown complete a nominal idle lifecycle" )
    {
        streamWriterLifecycleTest app;
        startupScope              startup( app );
        streamWriterConfig        cfg;

        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "startup_success" / "raw" ).string();

        loadConfig( app, "startup_success", cfg );

        const int startupRv = app.appStartup();
        startup.markStarted( startupRv == 0 );

        REQUIRE( startupRv == 0 );
        REQUIRE( std::filesystem::exists( *cfg.m_savePath ) );
        REQUIRE( app.m_fgThread.joinable() );
        REQUIRE( app.m_swThread.joinable() );
        REQUIRE( app.m_xrif != nullptr );
        REQUIRE( app.m_xrif_timing != nullptr );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );

        app.m_writing = WRITING;

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::OPERATING );

        REQUIRE( app.shutdownStartedApp() == 0 );
        startup.disarm();

        REQUIRE( app.m_xrif == nullptr );
        REQUIRE( app.m_xrif_timing == nullptr );
        REQUIRE( app.m_fgThread.joinable() == false );
        REQUIRE( app.m_swThread.joinable() == false );
        REQUIRE( app.m_writing == NOT_WRITING );
    }

    SECTION( "appStartup arms writing immediately when configured to start writing" )
    {
        streamWriterLifecycleTest app;
        startupScope              startup( app );
        streamWriterConfig        cfg;

        cfg.m_startWriting = true;
        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "startup_start_writing" / "raw" ).string();

        loadConfig( app, "startup_start_writing", cfg );

        const int startupRv = app.appStartup();
        startup.markStarted( startupRv == 0 );

        REQUIRE( startupRv == 0 );
        REQUIRE( app.m_writing == START_WRITING );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::OPERATING );
    }
}

/// Verify streamWriter publishes backlog summaries, INDI status, and save telemetry from the main loop.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter appLogic reports backlog and save status", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::updateINDI() );
    XWCTEST_DOXYGEN_REF( streamWriter::checkRecordTimes() );
    XWCTEST_DOXYGEN_REF( streamWriter::recordTelem( (const telem_saving_state *)nullptr ) );
    XWCTEST_DOXYGEN_REF( streamWriter::recordSavingState( true ) );
    XWCTEST_DOXYGEN_REF( streamWriter::recordSavingStats( true ) );
    #endif
    // clang-format on

    SECTION( "appLogic doubles the backlog summary interval while skips persist and resets when idle" )
    {
        streamWriterLifecycleTest app;
        startupScope              startup( app );
        streamWriterConfig        cfg;

        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "backlog_summary" / "raw" ).string();

        loadConfig( app, "backlog_summary", cfg );

        const int startupRv = app.appStartup();
        startup.markStarted( startupRv == 0 );
        REQUIRE( startupRv == 0 );

        app.m_skipSummaryIntervalSec = 10.0;
        app.m_nextSkipSummaryTime    = 0;

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_skipSummaryIntervalSec == Approx( 10.0 ) );
        REQUIRE( app.m_nextSkipSummaryTime > mx::sys::get_curr_time() );

        app.m_skippedFrameCount.store( 4 );
        app.m_repeatSemaphoreCount.store( 2 );
        app.m_nextSkipSummaryTime = mx::sys::get_curr_time() - 1.0;

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_skippedFrameCount.load() == 0 );
        REQUIRE( app.m_repeatSemaphoreCount.load() == 0 );
        REQUIRE( app.m_skipSummaryIntervalSec == Approx( 20.0 ) );
        REQUIRE( app.m_nextSkipSummaryTime > mx::sys::get_curr_time() );

        app.m_nextSkipSummaryTime = mx::sys::get_curr_time() - 1.0;

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_skipSummaryIntervalSec == Approx( 10.0 ) );
    }

    SECTION( "save telemetry helpers accept idle and active writing statistics" )
    {
        streamWriterLifecycleTest app;
        startupScope              startup( app );
        streamWriterConfig        cfg;

        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "indi_reporting" / "raw" ).string();

        loadConfig( app, "indi_reporting", cfg );

        const int startupRv = app.appStartup();
        startup.markStarted( startupRv == 0 );
        REQUIRE( startupRv == 0 );

        app.m_writing = NOT_WRITING;
        app.updateINDI();
        REQUIRE( app.recordSavingState( true ) == 0 );

        app.m_width    = 100;
        app.m_height   = 100;
        app.m_typeSize = 2;

        app.m_xrif->compression_ratio = 2.5;
        app.m_xrif->encode_rate       = 200000.0;
        app.m_xrif->difference_rate   = 100000.0;
        app.m_xrif->reorder_rate      = 60000.0;
        app.m_xrif->compress_rate     = 40000.0;

        app.m_writing = WRITING;
        app.updateINDI();

        app.m_currSaveStart = 7;

        REQUIRE( app.recordSavingState( true ) == 0 );
        REQUIRE( app.recordSavingStats( true ) == 0 );
        REQUIRE( app.recordTelem( nullptr ) == 0 );
        REQUIRE( app.checkRecordTimes() == 0 );
    }
}

/// Verify `fgThreadExec()` ingests shmim frames, tracks gaps, and schedules save work.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter fgThreadExec ingests stream data and manages write scheduling", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::fgThreadExec() );
    XWCTEST_DOXYGEN_REF( streamWriter::allocate_circbufs() );
    XWCTEST_DOXYGEN_REF( streamWriter::allocate_xrif() );
    #endif
    // clang-format on

    SECTION( "cube streams populate frame arrays, skipped-frame counters, and missing timestamps" )
    {
        streamWriterLifecycleTest app;
        fgHarnessScope            fgScope( app );
        streamWriterConfig        cfg;

        cfg.m_shmimName           = uniqueShmimName( "cube_ingest" );
        cfg.m_maxCircBuffLength   = 8;
        cfg.m_maxWriteChunkLength = 4;
        cfg.m_maxChunkTime        = 0.25;
        cfg.m_semWaitNSec         = 1000000;
        cfg.m_savePath = ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "fg_cube" / "raw" ).string();

        tempStream source( cfg.m_shmimName, 2, 2, 8 );

        loadConfig( app, "fg_cube_ingest", cfg );
        REQUIRE( app.initializeFgHarness() == 0 );
        fgScope.markActive( true );

        app.startFgHarnessThread();

        REQUIRE( waitFor( [&app]() { return app.m_rawImageCircBuff != nullptr && app.m_timingCircBuff != nullptr; } ) );
        REQUIRE( app.m_width == 2 );
        REQUIRE( app.m_height == 2 );
        REQUIRE( app.m_dataType == _DATATYPE_UINT16 );
        REQUIRE( app.m_typeSize == sizeof( uint16_t ) );
        REQUIRE( app.m_circBuffLength == 8 );
        REQUIRE( app.m_writeChunkLength == 4 );

        const timespec atime0{ 111, 222 };
        const timespec wtime0{ 333, 444 };
        source.publishFrame( 0, 1, 100, atime0, wtime0 );

        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        REQUIRE( rawFrameWord( app, 0, 0 ) == 100 );
        REQUIRE( rawFrameWord( app, 0, 3 ) == 103 );
        REQUIRE( timingWord( app, 0, 0 ) == 1 );
        REQUIRE( timingWord( app, 0, 1 ) == static_cast<uint64_t>( atime0.tv_sec ) );
        REQUIRE( timingWord( app, 0, 2 ) == static_cast<uint64_t>( atime0.tv_nsec ) );
        REQUIRE( timingWord( app, 0, 3 ) == static_cast<uint64_t>( wtime0.tv_sec ) );
        REQUIRE( timingWord( app, 0, 4 ) == static_cast<uint64_t>( wtime0.tv_nsec ) );

        source.publishFrame( 0, 1, 200, atime0, wtime0 );

        REQUIRE( waitFor( [&app]() { return app.m_repeatSemaphoreCount.load() == 1; } ) );
        REQUIRE( app.m_currImage == 1 );
        REQUIRE( rawFrameWord( app, 0, 0 ) == 100 );

        const timespec missingTime{ 0, 0 };
        source.publishFrame( 1, 3, 300, missingTime, missingTime );

        REQUIRE( waitFor( [&app]() { return app.m_currImage == 2; } ) );
        REQUIRE( app.m_skippedFrameCount.load() == 1 );
        REQUIRE( rawFrameWord( app, 1, 0 ) == 300 );
        REQUIRE( rawFrameWord( app, 1, 3 ) == 303 );
        REQUIRE( timingWord( app, 1, 0 ) == 3 );
        REQUIRE( timingWord( app, 1, 1 ) != 0 );
        REQUIRE( timingWord( app, 1, 3 ) == timingWord( app, 1, 1 ) );
        REQUIRE( timingWord( app, 1, 4 ) == timingWord( app, 1, 2 ) );
        REQUIRE( app.m_currImageTime != 0 );

        app.stopFgHarness();
        fgScope.disarm();
    }

    SECTION( "cube streams post save work on chunk boundaries, timeout flushes, and stop requests" )
    {
        streamWriterLifecycleTest app;
        fgHarnessScope            fgScope( app );
        streamWriterConfig        cfg;

        cfg.m_shmimName           = uniqueShmimName( "cube_writing" );
        cfg.m_maxCircBuffLength   = 8;
        cfg.m_maxWriteChunkLength = 2;
        cfg.m_maxChunkTime        = 0.2;
        cfg.m_writeStopTimeout    = 0.05;
        cfg.m_semWaitNSec         = 1000000;
        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "fg_writing" / "raw" ).string();

        tempStream source( cfg.m_shmimName, 2, 2, 8 );

        loadConfig( app, "fg_cube_writing", cfg );
        REQUIRE( app.initializeFgHarness() == 0 );
        fgScope.markActive( true );

        app.startFgHarnessThread();

        REQUIRE( waitFor( [&app]() { return app.m_rawImageCircBuff != nullptr; } ) );

        const timespec baseAtime = currentRealtime();
        const timespec baseWtime = offsetTimespec( baseAtime, 1000 );

        app.m_writing = START_WRITING;
        source.publishFrame( 0, 1, 10, baseAtime, baseWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        REQUIRE( app.m_writing == WRITING );

        source.publishFrame( 1, 2, 20, offsetTimespec( baseAtime, 1000000 ), offsetTimespec( baseWtime, 1000000 ) );
        REQUIRE( waitFor( [&app]() { return app.writerSemaphoreValue() > 0; } ) );
        REQUIRE( app.m_currSaveStart == 0 );
        REQUIRE( app.m_currSaveStop == 2 );
        REQUIRE( app.m_currSaveStopFrameNo == 2 );
        REQUIRE( app.drainWriterSemaphore() >= 1 );

        const timespec stopAtime = currentRealtime();
        const timespec stopWtime = offsetTimespec( stopAtime, 1000 );
        app.m_writing            = START_WRITING;
        source.publishFrame( 2, 3, 30, stopAtime, stopWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 3; } ) );
        REQUIRE( app.m_writing == WRITING );

        app.m_writing           = STOP_WRITING;
        app.m_stopWriteDeadline = mx::sys::get_curr_time() + cfg.m_writeStopTimeout;
        source.publishFrame( 3, 4, 40, offsetTimespec( stopAtime, 1000000 ), offsetTimespec( stopWtime, 1000000 ) );
        REQUIRE( waitFor( [&app]() { return app.writerSemaphoreValue() > 0; } ) );
        REQUIRE( app.m_currSaveStart == 2 );
        REQUIRE( app.m_currSaveStop == 4 );
        REQUIRE( app.m_currSaveStopFrameNo == 4 );
        REQUIRE( app.drainWriterSemaphore() >= 1 );

        const timespec timeoutStopAtime = currentRealtime();
        const timespec timeoutStopWtime = offsetTimespec( timeoutStopAtime, 1000 );
        app.m_writing                   = START_WRITING;
        source.publishFrame( 4, 5, 50, timeoutStopAtime, timeoutStopWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 5; } ) );
        REQUIRE( app.m_writing == WRITING );

        app.m_writing           = STOP_WRITING;
        app.m_stopWriteDeadline = mx::sys::get_curr_time() + cfg.m_writeStopTimeout;
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        REQUIRE( app.writerSemaphoreValue() == 0 );
        REQUIRE( waitFor( [&app]() { return app.writerSemaphoreValue() > 0; } ) );
        REQUIRE( app.m_currSaveStart == 4 );
        REQUIRE( app.m_currSaveStop == 5 );
        REQUIRE( app.m_currSaveStopFrameNo == 5 );
        REQUIRE( app.drainWriterSemaphore() >= 1 );

        app.m_writing = NOT_WRITING;
        app.stopFgHarness();
        fgScope.disarm();
    }

    SECTION( "cube streams flush partial chunks on timeout without dropping the queued frame" )
    {
        streamWriterLifecycleTest app;
        fgHarnessScope            fgScope( app );
        streamWriterConfig        cfg;

        cfg.m_shmimName           = uniqueShmimName( "cube_timeout" );
        cfg.m_maxCircBuffLength   = 8;
        cfg.m_maxWriteChunkLength = 4;
        cfg.m_maxChunkTime        = 0.2;
        cfg.m_semWaitNSec         = 1000000;
        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "fg_timeout" / "raw" ).string();

        tempStream source( cfg.m_shmimName, 2, 2, 8 );

        loadConfig( app, "fg_cube_timeout", cfg );
        REQUIRE( app.initializeFgHarness() == 0 );
        fgScope.markActive( true );

        app.startFgHarnessThread();

        REQUIRE( waitFor( [&app]() { return app.m_rawImageCircBuff != nullptr; } ) );

        const timespec timeoutAtime = currentRealtime();
        const timespec timeoutWtime = offsetTimespec( timeoutAtime, 1000 );

        app.m_writing = START_WRITING;
        source.publishFrame( 0, 1, 10, timeoutAtime, timeoutWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        REQUIRE( app.m_writing == WRITING );

        REQUIRE( waitFor( [&app]() { return app.writerSemaphoreValue() == 1 && app.m_writing == START_WRITING; } ) );
        REQUIRE( app.m_currSaveStart == 0 );
        REQUIRE( app.m_currSaveStop == 1 );
        REQUIRE( app.m_currSaveStopFrameNo == 1 );
        REQUIRE( rawFrameWord( app, 0, 0 ) == 10 );
        REQUIRE( rawFrameWord( app, 0, 3 ) == 13 );
        REQUIRE( app.drainWriterSemaphore() == 1 );

        app.m_writing = NOT_WRITING;
        app.stopFgHarness();
        fgScope.disarm();
    }

    SECTION( "replaced shmims flush the pending chunk and reconnect ready to keep writing" )
    {
        streamWriterLifecycleTest   app;
        fgHarnessScope              fgScope( app );
        streamWriterConfig          cfg;
        std::unique_ptr<tempStream> source;

        cfg.m_shmimName           = uniqueShmimName( "cube_restart" );
        cfg.m_maxCircBuffLength   = 8;
        cfg.m_maxWriteChunkLength = 4;
        cfg.m_maxChunkTime        = 10.0;
        cfg.m_semWaitNSec         = 1000000;
        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "fg_restart" / "raw" ).string();

        source = std::make_unique<tempStream>( cfg.m_shmimName, 2, 2, 8 );

        loadConfig( app, "fg_cube_restart", cfg );
        REQUIRE( app.initializeFgHarness() == 0 );
        fgScope.markActive( true );

        app.startFgHarnessThread();

        REQUIRE( waitFor( [&app]()
                          { return app.m_rawImageCircBuff != nullptr && app.m_width == 2 && app.m_height == 2; } ) );

        const timespec baseAtime = currentRealtime();
        const timespec baseWtime = offsetTimespec( baseAtime, 1000 );
        const timespec nextAtime = offsetTimespec( baseAtime, 1000000 );
        const timespec nextWtime = offsetTimespec( baseWtime, 1000000 );

        app.m_writing = START_WRITING;
        source->publishFrame( 0, 1, 100, baseAtime, baseWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        REQUIRE( app.m_writing == WRITING );

        source->publishFrame( 1, 2, 200, nextAtime, nextWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 2; } ) );
        REQUIRE( app.writerSemaphoreValue() == 0 );

        source.reset();
        source = std::make_unique<tempStream>( cfg.m_shmimName, 3, 1, 8 );

        REQUIRE(
            waitFor( [&app]() { return app.writerSemaphoreValue() > 0 && app.m_writing == STOP_WRITING; }, 3000 ) );
        REQUIRE( app.m_currSaveStart == 0 );
        REQUIRE( app.m_currSaveStop == 2 );
        REQUIRE( app.m_currSaveStopFrameNo == 2 );

        const std::vector<uint16_t> pendingRaw    = copyPendingRawFrames( app );
        const std::vector<uint64_t> pendingTiming = copyPendingTimingFrames( app );

        REQUIRE( pendingRaw == std::vector<uint16_t>{ 100, 101, 102, 103, 200, 201, 202, 203 } );
        REQUIRE( pendingTiming == std::vector<uint64_t>{ 1,
                                                         static_cast<uint64_t>( baseAtime.tv_sec ),
                                                         static_cast<uint64_t>( baseAtime.tv_nsec ),
                                                         static_cast<uint64_t>( baseWtime.tv_sec ),
                                                         static_cast<uint64_t>( baseWtime.tv_nsec ),
                                                         2,
                                                         static_cast<uint64_t>( nextAtime.tv_sec ),
                                                         static_cast<uint64_t>( nextAtime.tv_nsec ),
                                                         static_cast<uint64_t>( nextWtime.tv_sec ),
                                                         static_cast<uint64_t>( nextWtime.tv_nsec ) } );

        app.m_writing              = START_WRITING;
        app.m_resumeAfterReconnect = true;
        app.m_writePending         = false;
        REQUIRE( app.drainWriterSemaphore() >= 1 );

        REQUIRE( waitFor( [&app]() { return app.m_width == 3 && app.m_height == 1 && app.m_writing == START_WRITING; },
                          3000 ) );
        REQUIRE( app.writerSemaphoreValue() == 0 );

        const timespec reconnectedAtime = currentRealtime();
        const timespec reconnectedWtime = offsetTimespec( reconnectedAtime, 1000 );
        source->publishFrame( 0, 101, 500, reconnectedAtime, reconnectedWtime );

        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        REQUIRE( app.m_writing == WRITING );
        REQUIRE( rawFrameWord( app, 0, 0 ) == 500 );
        REQUIRE( rawFrameWord( app, 0, 2 ) == 502 );
        REQUIRE( timingWord( app, 0, 0 ) == 101 );
        REQUIRE( app.writerSemaphoreValue() == 0 );

        app.stopFgHarness();
        fgScope.disarm();
    }

    SECTION( "restart cleanup times out instead of hanging when no writer thread drains the queued flush" )
    {
        streamWriterLifecycleTest   app;
        fgHarnessScope              fgScope( app );
        streamWriterConfig          cfg;
        std::unique_ptr<tempStream> source;

        cfg.m_shmimName           = uniqueShmimName( "cube_restart_timeout" );
        cfg.m_maxCircBuffLength   = 8;
        cfg.m_maxWriteChunkLength = 4;
        cfg.m_maxChunkTime        = 10.0;
        cfg.m_writeStopTimeout    = 0.1;
        cfg.m_semWaitNSec         = 1000000;
        cfg.m_savePath =
            ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "fg_restart_timeout" / "raw" ).string();

        source = std::make_unique<tempStream>( cfg.m_shmimName, 2, 2, 8 );

        loadConfig( app, "fg_cube_restart_timeout", cfg );
        REQUIRE( app.initializeFgHarness() == 0 );
        fgScope.markActive( true );
        app.m_writeCompletionTimeout = 0.1;

        app.startFgHarnessThread();

        REQUIRE( waitFor( [&app]()
                          { return app.m_rawImageCircBuff != nullptr && app.m_width == 2 && app.m_height == 2; } ) );

        const timespec baseAtime = currentRealtime();
        const timespec baseWtime = offsetTimespec( baseAtime, 1000 );

        app.m_writing = START_WRITING;
        source->publishFrame( 0, 1, 100, baseAtime, baseWtime );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        source->publishFrame( 1, 2, 200, offsetTimespec( baseAtime, 1000000 ), offsetTimespec( baseWtime, 1000000 ) );
        REQUIRE( waitFor( [&app]() { return app.m_currImage == 2; } ) );
        REQUIRE( app.m_writing == WRITING );
        REQUIRE( app.writerSemaphoreValue() == 0 );

        source.reset();
        source = std::make_unique<tempStream>( cfg.m_shmimName, 3, 1, 8 );

        REQUIRE( waitFor( [&app]() { return app.m_shutdown != 0; }, 3000 ) );
        REQUIRE( app.m_writing == STOP_WRITING );
        REQUIRE( app.m_writePending == true );
        REQUIRE( app.writerSemaphoreValue() > 0 );

        app.stopFgHarness();
        fgScope.disarm();
    }

    SECTION( "two-dimensional streams ingest metadata from the shared image header" )
    {
        streamWriterLifecycleTest app;
        fgHarnessScope            fgScope( app );
        streamWriterConfig        cfg;

        cfg.m_shmimName           = uniqueShmimName( "image2d" );
        cfg.m_maxCircBuffLength   = 8;
        cfg.m_maxWriteChunkLength = 4;
        cfg.m_maxChunkTime        = 0.25;
        cfg.m_semWaitNSec         = 1000000;
        cfg.m_savePath = ( std::filesystem::path( "/tmp/streamWriter_lifecycle_test" ) / "fg_2d" / "raw" ).string();

        tempStream source( cfg.m_shmimName, 3, 2, 1 );

        loadConfig( app, "fg_2d_stream", cfg );
        REQUIRE( app.initializeFgHarness() == 0 );
        fgScope.markActive( true );

        app.startFgHarnessThread();

        REQUIRE( waitFor( [&app]() { return app.m_rawImageCircBuff != nullptr; } ) );

        const timespec atime{ 900, 1234 };
        const timespec writetime{ 901, 5678 };
        source.publishFrame( 0, 21, 1000, atime, writetime );

        REQUIRE( waitFor( [&app]() { return app.m_currImage == 1; } ) );
        REQUIRE( app.m_width == 3 );
        REQUIRE( app.m_height == 2 );
        REQUIRE( rawFrameWord( app, 0, 0 ) == 1000 );
        REQUIRE( rawFrameWord( app, 0, 5 ) == 1005 );
        REQUIRE( timingWord( app, 0, 0 ) == 21 );
        REQUIRE( timingWord( app, 0, 1 ) == static_cast<uint64_t>( atime.tv_sec ) );
        REQUIRE( timingWord( app, 0, 2 ) == static_cast<uint64_t>( atime.tv_nsec ) );
        REQUIRE( timingWord( app, 0, 3 ) == static_cast<uint64_t>( writetime.tv_sec ) );
        REQUIRE( timingWord( app, 0, 4 ) == static_cast<uint64_t>( writetime.tv_nsec ) );
        REQUIRE( app.m_currImageTime == static_cast<uint64_t>( writetime.tv_sec ) * 1000000000ULL +
                                            static_cast<uint64_t>( writetime.tv_nsec ) );

        app.stopFgHarness();
        fgScope.disarm();
    }
}

} // namespace streamWriterTest

} // namespace libXWCTest
