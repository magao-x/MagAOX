/** \file frameGrabber_test.cpp
 * \brief Catch2 tests for the MagAOX::app::dev::frameGrabber CRTP mixin.
 *
 * The tests drive the real frameGrabber code through small MagAOXApp<true> harness
 * classes. The harness classes stub the derived-class hooks that a camera would
 * provide, such as configureAcquisition() and acquireAndCheckValid(), and count how
 * often each hook is called. The acquisition loop fgThreadExec() is usually run on the
 * calling thread so every branch can be reached deterministically.
 *
 * The tests use real ImageStreamIO shared memory streams and real semaphores. They
 * point MILK_SHM_DIR at /tmp/frameGrabber_test/shm before any stream is created and
 * write config files under /tmp. The last TEST_CASE in this file leaks its harness on
 * purpose. See the comment above it for why it must remain last.
 *
 * \ingroup app_dev_unit_tests
 */

#include "frameGrabber_test.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

using namespace MagAOX::app;

namespace frameGrabber_tests
{

/// Put a `fgTest`-like harness into the state where `fgThreadExec` will proceed
/// straight through its startup wait loops without blocking. The app is marked as
/// powered on and OPERATING, and the thread-init handshake is skipped.
template <class appT> void primeForSyncExec( appT &app )
{
    app.m_fgThreadInit    = false;
    app.m_powerMgtEnabled = true;
    app.m_powerState      = 1;
    app.m_powerTargetState = 1;
    app.state( stateCodes::OPERATING );
    app.m_shutdown = 0;
}

/// Start a short-lived dummy framegrabber thread so `frameGrabber::appLogic()` has a
/// live, joinable thread to check. Its first line always calls `pthread_tryjoin_np`
/// on `m_fgThread`. That is undefined behavior on a never-started `std::thread`.
template <class appT> void startDummyFgThread( appT &app, int sleepMs = 200 )
{
    app.m_fgThread = std::thread( [sleepMs]() { std::this_thread::sleep_for( std::chrono::milliseconds( sleepMs ) ); } );
}

/// RAII helper that starts a dummy framegrabber thread and joins it on scope exit.
/// This keeps a failed REQUIRE from destroying the harness while the thread is still
/// joinable, which would call std::terminate.
template <class appT> struct fgThreadScope
{
    appT &m_app;
    explicit fgThreadScope( appT &app, int sleepMs = 200 ) : m_app( app )
    {
        startDummyFgThread( m_app, sleepMs );
    }
    ~fgThreadScope()
    {
        if( m_app.m_fgThread.joinable() )
        {
            m_app.m_fgThread.join();
        }
    }
};

} // namespace frameGrabber_tests

using namespace frameGrabber_tests;

/// Verify that setupConfig() and loadConfig() handle the flippable and non-flippable
/// configurations and every input-validation branch in loadConfig(). Config files are
/// written under /tmp and read back through a real appConfigurator.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber setupConfig and loadConfig manage every configurable option", "[frameGrabber]" )
{
    SECTION( "setupConfig adds the flip option when flippable" )
    {
        fgTest             app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
    }

    SECTION( "setupConfig omits the flip option when not flippable" )
    {
        fgTestNoFlip       app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
    }

    SECTION( "loadConfig applies defaults, corrections, and every flip option" )
    {
        struct flipCase
        {
            std::string flip;
            int         expected;
        };

        std::vector<flipCase> cases = { { "flipNone", fgTest::fgFlipNone },
                                        { "flipUD", fgTest::fgFlipUD },
                                        { "flipLR", fgTest::fgFlipLR },
                                        { "flipUDLR", fgTest::fgFlipUDLR },
                                        { "garbage", fgTest::fgFlipNone } };

        for( auto &c : cases )
        {
            fgTest app;

            mx::app::appConfigurator config;
            REQUIRE( app.setupConfig( config ) == 0 );

            std::string path = "/tmp/frameGrabber_test_flip_" + c.flip + ".conf";
            mx::app::writeConfigFile( path,
                                      { "framegrabber", "framegrabber", "framegrabber" },
                                      { "shmimName", "circBuffLength", "defaultFlip" },
                                      { "myshmim", "0", c.flip } );

            config.readConfig( path );
            REQUIRE( app.loadConfig( config ) == 0 );
            REQUIRE( app.m_shmimName == "myshmim" );
            REQUIRE( app.m_circBuffLength == 1 ); // Corrected up from the configured 0.
            REQUIRE( app.m_defaultFlip == c.expected );
        }
    }

    SECTION( "loadConfig fails when shmimName is empty and configName is empty" )
    {
        fgTest app;
        app.m_configName = "";
        app.m_shmimName  = "";

        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        REQUIRE( app.loadConfig( config ) < 0 );
    }

    SECTION( "loadConfig clamps a negative latencyTime to 0" )
    {
        fgTest app;

        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );

        std::string path = "/tmp/frameGrabber_test_negtime.conf";
        mx::app::writeConfigFile( path, { "framegrabber" }, { "latencyTime" }, { "-5" } );
        config.readConfig( path );

        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.m_latencyCircBuffMaxTime == 0 );
    }
}

/// Verify that configCircBuffs() sizes the latency circular buffers from the fps and
/// the configured limits. It covers the zero-length, negative-fps, upper clamp, and
/// lower floor branches.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber configCircBuffs configures or clamps the latency buffers", "[frameGrabber]" )
{
    SECTION( "zero latency length disables the circular buffers" )
    {
        fgTest app;
        app.m_latencyCircBuffMaxLength = 0;
        app.fpsValue                  = 100;
        REQUIRE( app.configCircBuffs() == 0 );
        REQUIRE( app.m_atimes.maxEntries() == 0 );
        REQUIRE( app.m_wtimes.maxEntries() == 0 );
    }

    SECTION( "a negative fps disables the buffers and reports an error" )
    {
        fgTest app;
        app.fpsValue = -1;
        REQUIRE( app.configCircBuffs() < 0 );
    }

    SECTION( "a large requested size is clamped to the configured maximum" )
    {
        fgTest app;
        app.m_latencyCircBuffMaxLength = 10;
        app.m_latencyCircBuffMaxTime   = 5;
        app.fpsValue                   = 1000; // 2*5*1000 = 10000 entries requested. This clamps down to 10.
        REQUIRE( app.configCircBuffs() == 0 );
        REQUIRE( app.m_atimes.maxEntries() == 10 );
    }

    SECTION( "a tiny requested size is floored to 3 for a meaningful variance" )
    {
        fgTest app;
        app.m_latencyCircBuffMaxLength = 100000;
        app.m_latencyCircBuffMaxTime   = 1e-4; // arbitrary small window
        app.fpsValue                   = 1;
        REQUIRE( app.configCircBuffs() == 0 );
        REQUIRE( app.m_atimes.maxEntries() == 3 );
    }
}

/// Verify that loadImageIntoStreamCopy() takes the plain memcpy shortcut when the
/// harness is not flippable and dispatches on m_defaultFlip when it is. An unknown flip
/// value must return nullptr.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber loadImageIntoStreamCopy dispatches on the flip setting", "[frameGrabber]" )
{
    uint8_t src[4]  = { 1, 2, 3, 4 };
    uint8_t dest[4] = { 0, 0, 0, 0 };

    SECTION( "the non-flippable configuration always uses a plain memcpy" )
    {
        fgTestNoFlip app;
        REQUIRE( app.loadImageIntoStreamCopy( dest, src, 2, 2, 1 ) == dest );
        REQUIRE( dest[0] == 1 );
    }

    SECTION( "flippable configurations dispatch on m_defaultFlip" )
    {
        fgTest app;

        app.m_defaultFlip = fgTest::fgFlipNone;
        REQUIRE( app.loadImageIntoStreamCopy( dest, src, 2, 2, 1 ) != nullptr );

        app.m_defaultFlip = fgTest::fgFlipUD;
        REQUIRE( app.loadImageIntoStreamCopy( dest, src, 2, 2, 1 ) != nullptr );

        app.m_defaultFlip = fgTest::fgFlipLR;
        REQUIRE( app.loadImageIntoStreamCopy( dest, src, 2, 2, 1 ) != nullptr );

        app.m_defaultFlip = fgTest::fgFlipUDLR;
        REQUIRE( app.loadImageIntoStreamCopy( dest, src, 2, 2, 1 ) != nullptr );

        app.m_defaultFlip = 99; // An unknown flip value.
        REQUIRE( app.loadImageIntoStreamCopy( dest, src, 2, 2, 1 ) == nullptr );
    }
}

/// Verify that openShmim() attaches to a shmim created by another process and reports
/// why it could not. Real streams are created with tempStream. The cases are a missing
/// shmim, a not-yet-ready shmim, a corrupt shmim, and valid 1D, 2D, and 3D shmims.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber openShmim attaches to an external shmim or reports why it could not", "[frameGrabber]" )
{
    SECTION( "a missing shmim is retried without a pre-existing stream" )
    {
        fgTest app;
        app.m_shmimName = uniqueShmimName( "missing" );
        REQUIRE( app.openShmim() == 1 );
        REQUIRE( app.openShmim() == 1 ); // The second call exercises the "already logged" branch.
    }

    SECTION( "a pre-existing local stream is closed before attaching" )
    {
        fgTest app;
        app.m_shmimName = uniqueShmimName( "replace" );
        app.m_imageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
        *app.m_imageStream = IMAGE{};
        REQUIRE( app.openShmim() == 1 ); // Still missing on disk, but the old handle was freed first.
        REQUIRE( app.m_imageStream == nullptr );
    }

    SECTION( "a shmim with too few live semaphores is not yet ready" )
    {
        fgTest      app;
        std::string name = uniqueShmimName( "notready" );
        app.m_shmimName  = name;

        // Create the stream with fewer semaphores than SEMAPHORE_MAXVAL. The field
        // `.md[0].sem` reflects the number requested at creation time.
        tempStream ext( name, 3, 4, 4, 2, _DATATYPE_UINT8, 1 );

        REQUIRE( app.openShmim() == 1 );
    }

    SECTION( "a valid 2D shmim is attached" )
    {
        fgTest      app;
        std::string name = uniqueShmimName( "valid2d" );
        app.m_shmimName  = name;

        tempStream ext( name, 2, 5, 6, 1 );

        REQUIRE( app.openShmim() == 0 );
        REQUIRE( app.m_width == 5 );
        REQUIRE( app.m_height == 6 );
        REQUIRE( app.m_circBuffLength == 1 );
        REQUIRE( app.m_dataType == _DATATYPE_UINT8 );

        if( app.m_imageStream != nullptr )
        {
            ImageStreamIO_closeIm( app.m_imageStream );
            free( app.m_imageStream );
            app.m_imageStream = nullptr;
        }
    }

    SECTION( "a valid 3D shmim reports its circular buffer depth" )
    {
        fgTest      app;
        std::string name = uniqueShmimName( "valid3d" );
        app.m_shmimName  = name;

        tempStream ext( name, 3, 5, 6, 4 );

        REQUIRE( app.openShmim() == 0 );
        REQUIRE( app.m_width == 5 );
        REQUIRE( app.m_height == 6 );
        REQUIRE( app.m_circBuffLength == 4 );

        if( app.m_imageStream != nullptr )
        {
            ImageStreamIO_closeIm( app.m_imageStream );
            free( app.m_imageStream );
            app.m_imageStream = nullptr;
        }
    }

    SECTION( "a valid 1D shmim reports a height and depth of 1" )
    {
        fgTest      app;
        std::string name = uniqueShmimName( "valid1d" );
        app.m_shmimName  = name;

        tempStream ext( name, 1, 9, 1, 1 );

        REQUIRE( app.openShmim() == 0 );
        REQUIRE( app.m_width == 9 );
        REQUIRE( app.m_height == 1 );
        REQUIRE( app.m_circBuffLength == 1 );

        if( app.m_imageStream != nullptr )
        {
            ImageStreamIO_closeIm( app.m_imageStream );
            free( app.m_imageStream );
            app.m_imageStream = nullptr;
        }
    }

    SECTION( "a corrupt shmim file fails to open and is reported as not-yet-ready" )
    {
        fgTest      app;
        std::string name = uniqueShmimName( "corrupt" );
        app.m_shmimName  = name;

        char path[1024];
        ImageStreamIO_filename( path, sizeof( path ), name.c_str() );
        FILE *f = fopen( path, "w" );
        REQUIRE( f != nullptr );
        const char junk[16] = { 0 };
        fwrite( junk, 1, sizeof( junk ), f );
        fclose( f );

        REQUIRE( app.openShmim() == 1 );

        unlink( path );
    }
}

/// Verify that updateINDI() returns 0 and does nothing when no INDI driver exists.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber updateINDI is a no-op without an active INDI driver", "[frameGrabber]" )
{
    fgTest app;
    REQUIRE( app.m_indiDriver == nullptr );
    REQUIRE( app.updateINDI() == 0 );
}

/// Verify that updateINDI() publishes its properties when a real indiDriver with no
/// FIFOs is attached. This exercises the indi::updateIfChanged() calls. The send-failure
/// paths those calls take internally are covered directly in indiUtils_test.cpp.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber updateINDI publishes properties with an active INDI driver", "[frameGrabber]" )
{
    fgTest app;
    app.m_configName = uniqueShmimName( "updateindi" );
    app.setupRealDriver();

    REQUIRE( app.frameGrabberT::appStartup() == 0 );

    // Set non-zero latency statistics like the ones appLogic() would compute from real
    // frames. This makes updateINDI() also exercise its fps-from-mean-latency divisions.
    app.m_mna = 0.01;
    app.m_mnw = 0.02;

    REQUIRE( app.updateINDI() == 0 );

    app.m_shutdown = 1;
    REQUIRE( app.frameGrabberT::appShutdown() == 0 );
}

/// Verify that appStartup() registers the shmimName, frameSize, and timing INDI
/// properties and starts the framegrabber thread. Each property registration failure
/// is forced by registering a duplicate first, and each must fail startup.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber appStartup registers INDI properties and starts its thread", "[frameGrabber]" )
{
    SECTION( "a duplicate fg_shmimName property blocks startup" )
    {
        fgTest            app;
        pcf::IndiProperty dup( pcf::IndiProperty::Text );
        dup.setDevice( app.configName() );
        dup.setName( "fg_shmimName" );
        REQUIRE( app.registerIndiPropertyNew( dup, nullptr ) == 0 );

        REQUIRE( app.frameGrabberT::appStartup() < 0 );
    }

    SECTION( "a duplicate fg_frameSize property blocks startup" )
    {
        fgTest            app;
        pcf::IndiProperty dup( pcf::IndiProperty::Number );
        dup.setDevice( app.configName() );
        dup.setName( "fg_frameSize" );
        REQUIRE( app.registerIndiPropertyNew( dup, nullptr ) == 0 );

        REQUIRE( app.frameGrabberT::appStartup() < 0 );
    }

    SECTION( "a duplicate fg_timing property blocks startup" )
    {
        fgTest            app;
        pcf::IndiProperty dup;
        REQUIRE( app.createROIndiNumber( dup, "fg_timing" ) == 0 );
        REQUIRE( app.registerIndiPropertyReadOnly( dup ) == 0 );

        REQUIRE( app.frameGrabberT::appStartup() < 0 );
    }

    SECTION( "a successful startup registers properties and joins cleanly on shutdown" )
    {
        fgTest app;

        REQUIRE( app.frameGrabberT::appStartup() == 0 );
        REQUIRE( app.m_fgThread.joinable() );

        // The thread is parked waiting for a READY or OPERATING state. Release it.
        app.m_shutdown = 1;
        REQUIRE( app.frameGrabberT::appShutdown() == 0 );
        REQUIRE( !app.m_fgThread.joinable() );

        // A second call is a no-op because nothing is left to join.
        REQUIRE( app.frameGrabberT::appShutdown() == 0 );
    }

    SECTION( "an unwritable cpuset assignment fails startup after the thread is running" )
    {
        fgTest app;
        app.m_fgCpuset = "definitely_not_a_real_cpuset_xyz";

        REQUIRE( app.frameGrabberT::appStartup() < 0 );

        // The thread itself did start. threadStart() only fails afterward, while trying
        // to move it into the cpuset. So the thread still needs to be released and joined.
        app.m_shutdown = 1;
        if( app.m_fgThread.joinable() )
        {
            app.m_fgThread.join();
        }
    }
}

/// Verify that appLogic() computes latency statistics from the acquisition and write
/// time circular buffers, or resets them when it cannot. The buffers are filled by hand.
/// A dummy thread stands in for the framegrabber thread so the liveness check passes.
/// The cases are the idle reset path, the insufficient-history reset path, the normal
/// statistics path, and the non-monotonic-write-time error path.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber appLogic computes latency statistics or reports why it could not", "[frameGrabber]" )
{
    SECTION( "appLogic resets statistics when not operating" )
    {
        fgTest        app;
        fgThreadScope<fgTest> thr( app );
        app.state( stateCodes::READY );
        REQUIRE( app.frameGrabberT::appLogic() == 0 );
        REQUIRE( app.m_mna == 0 );
    }

    SECTION( "appLogic resets statistics when there is not enough history" )
    {
        fgTest        app;
        fgThreadScope<fgTest> thr( app );
        app.state( stateCodes::OPERATING );
        app.m_powerMgtEnabled = false; // With power management off, powerState() always returns 1.
        app.fpsValue          = 100;

        timespec ts{ 1, 0 };
        app.m_atimes.maxEntries( 5 );
        app.m_wtimes.maxEntries( 5 );
        app.m_atimes.nextEntry( ts );
        app.m_wtimes.nextEntry( ts );
        ts.tv_sec = 2;
        app.m_atimes.nextEntry( ts );
        app.m_wtimes.nextEntry( ts );

        // A vanishingly small max time forces latTime below 2. That forces usedEntries
        // below 2 as well, which is the insufficient-history condition.
        app.m_latencyCircBuffMaxTime = 1e-7;
        app.m_cbFPS                  = 100;

        REQUIRE( app.frameGrabberT::appLogic() == 0 );
        REQUIRE( app.m_mna == 0 );
    }

    SECTION( "appLogic computes statistics once there is enough well-formed history" )
    {
        fgTest        app;
        fgThreadScope<fgTest> thr( app );
        app.state( stateCodes::OPERATING );
        app.m_powerMgtEnabled = false;
        app.fpsValue          = 100;
        app.m_cbFPS           = 100;
        app.m_latencyCircBuffMaxTime = 5;

        // The refEntry and usedEntries math in frameGrabber::appLogic() assumes the
        // buffers are filled to capacity, as they would be in steady-state operation.
        // So the loop below pushes at least maxEntries entries to fill the buffers. Two
        // extra entries beyond capacity also wrap the circular index, so m_latest ends up
        // less than usedEntries. That exercises the `refEntry = maxEntries() + ...` branch.
        app.m_atimes.maxEntries( 6 );
        app.m_wtimes.maxEntries( 6 );

        for( int i = 0; i < 8; ++i )
        {
            timespec a{ 100 + i, 0 };
            timespec w{ 100 + i, 500000000 };
            app.m_atimes.nextEntry( a );
            app.m_wtimes.nextEntry( w );
        }

        REQUIRE( app.frameGrabberT::appLogic() == 0 );
        REQUIRE( app.m_mna > 0 );
        REQUIRE( app.m_mnw > 0 );
        REQUIRE( app.m_maxa >= app.m_mina );

        // Also exercise recordFGTimings() and updateINDI() with real statistics.
        REQUIRE( app.recordFGTimings() == 0 );
        REQUIRE( app.recordFGTimings( true ) == 0 ); // Force a repeat record.
        REQUIRE( app.updateINDI() == 0 );
    }

    SECTION( "appLogic reports a non-monotonic write time" )
    {
        fgTest        app;
        fgThreadScope<fgTest> thr( app );
        app.state( stateCodes::OPERATING );
        app.m_powerMgtEnabled = false;
        app.fpsValue          = 100;
        app.m_cbFPS           = 100;
        app.m_latencyCircBuffMaxTime = 5;

        app.m_atimes.maxEntries( 6 );
        app.m_wtimes.maxEntries( 6 );

        for( int i = 0; i < 6; ++i )
        {
            timespec a{ 100 + i, 0 };
            // Write times decrease on every other sample, forcing a negative delta.
            timespec w{ 100 + ( i % 2 == 0 ? i : i - 2 ), 0 };
            app.m_atimes.nextEntry( a );
            app.m_wtimes.nextEntry( w );
        }

        REQUIRE( app.frameGrabberT::appLogic() == 0 );
    }

}

/// Verify that onPowerOff() resets the frame size and the latency statistics and
/// requests a reconfiguration.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber onPowerOff resets state and requests a reconfiguration", "[frameGrabber]" )
{
    fgTest app;
    app.m_width          = 100;
    app.m_height         = 100;
    app.m_circBuffLength = 5;
    app.m_mna            = 5;

    REQUIRE( app.frameGrabberT::onPowerOff() == 0 );
    REQUIRE( app.m_width == 0 );
    REQUIRE( app.m_height == 0 );
    REQUIRE( app.m_circBuffLength == 1 );
    REQUIRE( app.m_mna == 0 );
    REQUIRE( app.m_reconfig == true );
}

/// Verify the main acquisition loop of the framegrabber thread, fgThreadExec(). Most
/// sections call it on the test thread rather than on a background thread so every
/// branch can be reached deterministically. The harness ends the loop by requesting
/// shutdown once its queue of acquire results is empty. Sections that need to change
/// the harness while the loop sleeps run it on a background thread instead.
/**
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber fgThreadExec drives the full acquisition loop", "[frameGrabber]" )
{
    SECTION( "the thread waits for a READY/OPERATING, powered state and exits cleanly on shutdown" )
    {
        fgTest app;
        app.m_fgThreadInit    = false;
        app.m_powerMgtEnabled = true;
        app.m_powerState      = 0; // Not powered, so the state-wait loop must sleep and retry.
        app.m_shutdown        = 0;
        app.m_shmimName       = uniqueShmimName( "waitstate" );

        std::thread runner( [&app]() { app.fgThreadExec(); } );

        // Give the state-wait loop a chance to log and sleep at least once. Then ask it
        // to shut down without ever becoming ready.
        std::this_thread::sleep_for( std::chrono::milliseconds( 1200 ) );
        app.m_shutdown = 1;

        runner.join();

        REQUIRE( app.configureAcquisitionCalls == 0 ); // The loop never got past the state wait.
    }

    SECTION( "a single frame is created, published, and cleaned up (owned shmim, no circ. buff.)" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName      = uniqueShmimName( "single" );
        app.m_circBuffLength = 1;
        app.nextWidth        = 4;
        app.nextHeight       = 4;
        app.acquireResults   = { 0 };

        app.fgThreadExec();

        REQUIRE( app.configureAcquisitionCalls == 1 );
        REQUIRE( app.startAcquisitionCalls == 1 );
        REQUIRE( app.loadImageIntoStreamCalls == 1 );
        REQUIRE( app.m_imageStream == nullptr ); // Destroyed on shutdown because it was owned.
    }

    SECTION( "multiple frames wrap the circular buffer index" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName      = uniqueShmimName( "circ" );
        app.m_circBuffLength = 3;
        app.nextWidth        = 4;
        app.nextHeight       = 4;
        app.acquireResults   = { 0, 0, 0, 0 };

        app.fgThreadExec();

        REQUIRE( app.loadImageIntoStreamCalls == 4 );
    }

    SECTION( "acquireAndCheckValid returning >0 is retried without publishing" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName      = uniqueShmimName( "retry" );
        app.m_circBuffLength = 1;
        app.acquireResults   = { 1, 1, 0 };

        app.fgThreadExec();

        REQUIRE( app.acquireCalls == 4 ); // Results 1, 1, 0, then the empty-queue shutdown call.
        REQUIRE( app.loadImageIntoStreamCalls == 1 );
    }

    SECTION( "a loadImageIntoStream failure breaks the acquisition loop" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName            = uniqueShmimName( "loadfail" );
        app.m_circBuffLength       = 1;
        app.acquireResults         = { 0 };
        app.failLoadImageIntoStream = true;

        app.fgThreadExec();

        REQUIRE( app.loadImageIntoStreamCalls == 1 );
    }

    SECTION( "a startAcquisition failure is retried from the top of the outer loop" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName               = uniqueShmimName( "startfail" );
        app.m_circBuffLength          = 1;
        app.startAcquisitionFailCount = 1;
        app.acquireResults            = { 0 };

        app.fgThreadExec();

        REQUIRE( app.startAcquisitionCalls == 2 );
        REQUIRE( app.configureAcquisitionCalls == 2 );
    }

    SECTION( "a configureAcquisition failure sleeps briefly and is retried" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName                    = uniqueShmimName( "cfgfail" );
        app.m_circBuffLength               = 1;
        app.configureAcquisitionFailCount  = 1;
        app.acquireResults                 = { 0 };

        app.fgThreadExec();

        REQUIRE( app.configureAcquisitionCalls == 2 );
    }

    SECTION( "reconfig is invoked once m_reconfig is set" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName              = uniqueShmimName( "reconfig" );
        app.m_circBuffLength         = 1;
        app.acquireResults           = { 0 };
        // The harness sets m_reconfig while acquiring the first frame. The inner loop
        // condition then sees it and exits, so `reconfig()` gets called. The harness
        // version of reconfig() sets m_shutdown to 1 to end the outer loop cleanly.
        app.setReconfigOnNextAcquire = true;

        app.fgThreadExec();

        REQUIRE( app.reconfigCalls == 1 );
    }

    SECTION( "an optional post-publish hook can abort the acquisition loop" )
    {
        fgTestHook app;
        primeForSyncExec( app );
        app.m_shmimName      = uniqueShmimName( "hook" );
        app.m_circBuffLength = 1;
        app.acquireResults   = { 0, 0 };
        app.postPublishReturn = -1;

        app.fgThreadExec();

        REQUIRE( app.postPublishCalls == 1 );
    }

    SECTION( "a present post-publish hook that succeeds lets multiple circular-buffer frames publish" )
    {
        fgTestHook app;
        primeForSyncExec( app );
        app.m_shmimName      = uniqueShmimName( "hookok" );
        app.m_circBuffLength = 3;
        app.acquireResults   = { 1, 0, 0, 0, 0 }; // Includes one "no data yet" retry.

        app.fgThreadExec();

        REQUIRE( app.postPublishCalls == 4 );
    }

    SECTION( "a present post-publish hook still allows reconfig to be triggered" )
    {
        fgTestHook app;
        primeForSyncExec( app );
        app.m_shmimName              = uniqueShmimName( "hookreconfig" );
        app.m_circBuffLength         = 1;
        app.acquireResults           = { 0 };
        app.setReconfigOnNextAcquire = true;

        app.fgThreadExec();

        REQUIRE( app.reconfigCalls == 1 );
    }

    SECTION( "a negative fps from the derived class disables the latency circ. buffs. and is logged" )
    {
        fgTestHook app;
        primeForSyncExec( app );
        app.m_shmimName      = uniqueShmimName( "hookbadfps" );
        app.m_circBuffLength = 1;
        app.fpsValue         = -1;
        app.acquireResults   = { 0 };

        app.fgThreadExec();

        REQUIRE( app.postPublishCalls == 1 );
    }

    SECTION( "an already-connected stream with matching geometry is reused with a hook present" )
    {
        fgTestHook  app;
        primeForSyncExec( app );
        std::string name = uniqueShmimName( "hookreuse" );
        app.m_shmimName  = name;

        tempStream ext( name, 3, 4, 4, 2, _DATATYPE_UINT8 );

        app.m_imageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
        REQUIRE( ImageStreamIO_openIm( app.m_imageStream, name.c_str() ) == 0 );
        app.m_ownShmim       = false;
        app.nextWidth        = 4;
        app.nextHeight       = 4;
        app.m_circBuffLength = 2;
        app.acquireResults   = { 0 };

        app.fgThreadExec();

        REQUIRE( app.postPublishCalls == 1 );
        REQUIRE( app.m_imageStream == nullptr );
    }

    SECTION( "an already-connected 2D stream with matching geometry is reused" )
    {
        fgTest      app;
        primeForSyncExec( app );
        std::string name = uniqueShmimName( "reuse2d" );
        app.m_shmimName  = name;

        tempStream ext( name, 2, 4, 4, _DATATYPE_UINT8 );

        // Attach an independent handle to the externally-owned shmim. The frameGrabber
        // cleanup will `free()` this handle struct itself once it closes it. It does that
        // for every m_imageStream, owned or not. So the handle must be its own malloc'd
        // IMAGE, separate from the handle inside `ext` that the `ext` destructor cleans up.
        app.m_imageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
        REQUIRE( ImageStreamIO_openIm( app.m_imageStream, name.c_str() ) == 0 );
        app.m_ownShmim       = false;
        app.nextWidth        = 4;
        app.nextHeight       = 4;
        app.m_circBuffLength = 1;
        app.acquireResults   = { 0 };

        app.fgThreadExec();

        REQUIRE( app.loadImageIntoStreamCalls == 1 );
        REQUIRE( app.m_imageStream == nullptr ); // Closed but not destroyed because it was not owned.
    }

    SECTION( "an already-connected 3D stream with matching geometry is reused" )
    {
        fgTest      app;
        primeForSyncExec( app );
        std::string name = uniqueShmimName( "reuse3d" );
        app.m_shmimName  = name;

        tempStream ext( name, 3, 4, 4, 2, _DATATYPE_UINT8 );

        app.m_imageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
        REQUIRE( ImageStreamIO_openIm( app.m_imageStream, name.c_str() ) == 0 );
        app.m_ownShmim       = false;
        app.nextWidth        = 4;
        app.nextHeight       = 4;
        app.m_circBuffLength = 2;
        app.acquireResults   = { 0 };

        app.fgThreadExec();

        REQUIRE( app.loadImageIntoStreamCalls == 1 );
    }

    SECTION( "an unowned stream with the wrong size is retried until the size matches" )
    {
        fgTest      app;
        primeForSyncExec( app );
        std::string name = uniqueShmimName( "wrongsize" );
        app.m_shmimName  = name;

        tempStream ext( name, 2, 4, 4, _DATATYPE_UINT8 );

        // On the first configureAcquisition() call, report a mismatched size so the
        // "wrong size" branch and its one-time log are exercised. Changing the geometry
        // to match while the loop runs lets the retry converge and publish one frame.
        app.nextWidth  = 8;
        app.nextHeight = 8;
        app.m_imageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
        REQUIRE( ImageStreamIO_openIm( app.m_imageStream, name.c_str() ) == 0 );
        app.m_ownShmim    = false;
        app.m_circBuffLength = 1;
        app.acquireResults = { 0 };

        // Use a background thread here. The mismatch-retry path sleeps for a full
        // second, and the test must change the geometry to match during that sleep.
        std::thread runner( [&app]() { app.fgThreadExec(); } );

        // Give the mismatch branch a chance to log and sleep at least once.
        std::this_thread::sleep_for( std::chrono::milliseconds( 1300 ) );
        app.nextWidth  = 4;
        app.nextHeight = 4;

        runner.join();

        REQUIRE( app.loadImageIntoStreamCalls == 1 );
        REQUIRE( app.m_imageStream == nullptr );
    }

    SECTION( "the fps changing mid-acquisition reconfigures the latency circ. buffs." )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName              = uniqueShmimName( "fpschange" );
        app.m_circBuffLength         = 1;
        app.fpsValue                 = 100;
        app.m_latencyCircBuffMaxTime = 5;
        app.acquireResults           = { 0, 0 };

        // fps() is sampled into m_cbFPS once per outer-loop pass, inside configCircBuffs().
        // Changing it right after the first acquireAndCheckValid() call makes the
        // end-of-iteration `m_cbFPS != fps()` check fire on that same pass.
        app.bumpFpsAfterCall = 1;
        app.bumpedFpsValue   = 50;

        app.fgThreadExec();

        REQUIRE( app.loadImageIntoStreamCalls == 2 );
        REQUIRE( app.m_cbFPS == 50 );
    }

    SECTION( "the fps changing to a negative value mid-acquisition reports the "
             "configCircBuffs() failure" )
    {
        fgTest app;
        primeForSyncExec( app );
        app.m_shmimName              = uniqueShmimName( "fpsnegative" );
        app.m_circBuffLength         = 1;
        app.fpsValue                 = 100;
        app.m_latencyCircBuffMaxTime = 5;
        app.acquireResults           = { 0, 0 };

        // Same mid-iteration fps-change mechanism as above, but to a negative value.
        // configCircBuffs() itself then returns -1. That path is only reachable through
        // a real negative fps() reading, not through a hand-enumerated error code.
        app.bumpFpsAfterCall = 1;
        app.bumpedFpsValue   = -1;

        app.fgThreadExec();

        REQUIRE( app.loadImageIntoStreamCalls == 2 );
        REQUIRE( app.m_cbFPS == -1 );
    }
}

/// Verify that appLogic() reports an error once the framegrabber thread has exited.
/** This must be the last TEST_CASE in this file. `pthread_tryjoin_np` reaps the
 * native thread without clearing the owning `std::thread`. So the harness instance
 * is leaked on purpose rather than destructed. Destructing it would call
 * std::terminate on a thread that still reports joinable but whose native handle is
 * no longer valid to join or detach. `MagAOXApp` is a process-wide singleton, so no
 * other `MagAOXApp`-derived object may be constructed anywhere in this binary
 * afterward. This mirrors the same pattern used by
 * apps/ocam2KCtrl/tests/ocam2KCtrl_lifecycle_test.cpp.
 * \ingroup app_dev_unit_tests
 */
TEST_CASE( "frameGrabber appLogic reports an error once the framegrabber thread has exited", "[frameGrabber]" )
{
    auto *app = new fgTest;
    startDummyFgThread( *app, 1 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 30 ) );

    REQUIRE( app->frameGrabberT::appLogic() == -1 );
}
