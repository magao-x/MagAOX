//#define CATCH_CONFIG_MAIN
/** \file shmimMonitor_test.cpp
  * \brief Catch2 tests for the MagAOX::app::dev::shmimMonitor device mixin.
  *
  * The tests drive the real shmimMonitor code through the smTest harness declared in
  * shmimMonitor_test.hpp. It records every allocate() and processImage() call and can
  * make either one fail.
  * The monitoring thread smThreadExec() runs on a real background thread against real
  * ImageStreamIO shared memory streams and real semaphores. Frames are written into the
  * streams by hand to stand in for an upstream source process.
  *
  * The tests point MILK_SHM_DIR at /tmp/shmimMonitor_test/shm and wipe that directory
  * at startup. They write config files under /tmp. A no-op SIGUSR1 handler is installed
  * so the monitor thread can be interrupted out of a blocking semaphore wait.
  *
  * \ingroup app_dev_unit_tests
  */
#include "../../../../tests/catch2/catch.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <new>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/sys/timeUtils.hpp>

#include "shmimMonitor_test.hpp"

using namespace MagAOX::app;

/** \defgroup shmimMonitor_tests libXWC::app::dev::shmimMonitor Unit Tests
 * \ingroup app_dev_unit_tests
 */
namespace shmimMonitor_tests
{

// A single shared memory directory for the whole test binary. ImageStreamIO caches
// the shared memory directory the first time it is queried. The cache is a static
// local in ImageStreamIO_shmdirname. So MILK_SHM_DIR must be set before any
// ImageStreamIO call happens anywhere in this process. Doing this in a namespace-scope
// static initializer guarantees it runs before any TEST_CASE body, because all global
// constructors run before main().
const std::string g_shmDir = "/tmp/shmimMonitor_test/shm";

/// Wipe and recreate the shared memory directory, then point MILK_SHM_DIR at it.
int setupShmDir()
{
    // Remove any shmim files left behind by a prior run of this binary. Stale
    // semaphores and inodes under the same names would otherwise make the tests
    // non-idempotent across repeated invocations.
    std::filesystem::remove_all( g_shmDir );
    mx::ioutils::createDirectories( g_shmDir );
    setenv( "MILK_SHM_DIR", g_shmDir.c_str(), 1 );
    return 0;
}
static int g_shmDirSetup = setupShmDir();

// A no-op handler for SIGUSR1. With it installed, sending SIGUSR1 to a thread
// interrupts a blocking syscall with EINTR. Without it the default action would
// terminate the process.
void noopUsr1Handler( int )
{
}

/// Install noopUsr1Handler for SIGUSR1 from a static initializer.
int setupUsr1Handler()
{
    struct sigaction act;
    memset( &act, 0, sizeof( act ) );
    act.sa_handler = &noopUsr1Handler;
    sigemptyset( &act.sa_mask );
    act.sa_flags = 0;
    sigaction( SIGUSR1, &act, 0 );
    return 0;
}
static int g_usr1Setup = setupUsr1Handler();

/// Return the path of the shmim file for a stream name under g_shmDir.
std::string shmimPath( const std::string &name )
{
    return g_shmDir + "/" + name + ".im.shm";
}

/// Directly create a float IMAGE with full control over naxis, size, and nbsem. This
/// bypasses both milkImage and shmimMonitor::create(), which always use naxis=3, so
/// that the naxis==1 and naxis==2 branches in shmimMonitor can be exercised.
int rawCreate( IMAGE &image,
               const std::string &name,
               long naxis,
               uint32_t sz0,
               uint32_t sz1,
               uint32_t sz2,
               int nbsem = IMAGE_NB_SEMAPHORE )
{
    uint32_t imsize[3] = { sz0, sz1, sz2 };
    errno_t rv = ImageStreamIO_createIm_gpu(
        &image, name.c_str(), naxis, imsize, IMAGESTRUCT_FLOAT, -1, 1, nbsem, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0 );
    if( rv != IMAGESTREAMIO_SUCCESS )
        return -1;

    image.md->cnt1 = 0;
    return 0;
}

/// Write a constant-valued frame into slice `sliceIndex` of a float image, set cnt1 to
/// that slice, and post all semaphores. The image may be a circular buffer. This mimics
/// what a real upstream source process does on every new frame.
void writeFrame( IMAGE &image, uint32_t width, uint32_t height, uint32_t sliceIndex, float value )
{
    float *data = (float *)image.array.raw;
    size_t frameSize = (size_t)width * height;
    for( size_t i = 0; i < frameSize; ++i )
        data[sliceIndex * frameSize + i] = value;
    image.md->cnt1 = sliceIndex;
    ImageStreamIO_sempost( &image, -1 );
}

/// Mark every semaphore slot as already owned by another process that is always alive,
/// so that ImageStreamIO_getsemwaitindex() cannot find or adopt any of them.
void exhaustSemaphores( IMAGE &image )
{
    for( int i = 0; i < IMAGE_NB_SEMAPHORE; ++i )
        image.semReadPID[i] = 1; // Process 1 is init, which is always alive.
}

/// Poll a condition until it is true or a timeout elapses. Returns false on timeout.
template <typename Pred>
bool waitFor( Pred pred, int timeoutMs = 3000, int stepMs = 10 )
{
    int waited = 0;
    while( !pred() )
    {
        if( waited >= timeoutMs )
            return false;
        std::this_thread::sleep_for( std::chrono::milliseconds( stepMs ) );
        waited += stepMs;
    }
    return true;
}

} // namespace shmimMonitor_tests

using namespace shmimMonitor_tests;

/// Verify that setupConfig() and loadConfig() apply the defaults and then every
/// [shmimMonitor] config key. Config files are written under /tmp and read back
/// through a real appConfigurator.
/**
 * \ingroup shmimMonitor_tests
 */
SCENARIO( "shmimMonitor Configuration", "[dev::shmimMonitor]" )
{
    GIVEN( "a config file with no [shmimMonitor] section, loading defaults" )
    {
        mx::app::writeConfigFile( "/tmp/shmimMonitor_test.conf", { "none" }, { "nada" }, { "0" } ); // the placeholder entry every test uses so the config file is not empty

        mx::app::appConfigurator config;

        smTest pdt;

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/shmimMonitor_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        // setupConfig() sets m_shmimName to configName() by default.
        REQUIRE( pdt.shmimName() == "shmimMonitorTest" );
        REQUIRE( pdt.width() == 0 );
        REQUIRE( pdt.height() == 0 );
        REQUIRE( pdt.depth() == 0 );
        REQUIRE( pdt.dataType() == 0 );
        REQUIRE( pdt.typeSize() == 0 );
    }

    GIVEN( "a config file with a [shmimMonitor] section changing everything" )
    {
        std::vector<std::string> s, k, v;

        s.push_back( "shmimMonitor" );
        k.push_back( "threadPrio" );
        v.push_back( "23" );

        s.push_back( "shmimMonitor" );
        k.push_back( "cpuset" );
        v.push_back( "myset" );

        s.push_back( "shmimMonitor" );
        k.push_back( "shmimName" );
        v.push_back( "customShmim" );

        s.push_back( "shmimMonitor" );
        k.push_back( "getExistingFirst" );
        v.push_back( "true" );

        mx::app::writeConfigFile( "/tmp/shmimMonitor_test.conf", s, k, v );

        mx::app::appConfigurator config;

        smTest pdt;

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/shmimMonitor_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        REQUIRE( pdt.shmimName() == "customShmim" );
    }
}

/// Verify the appStartup(), appLogic(), and appShutdown() lifecycle. appStartup() must
/// start the monitor thread and must fail when INDI registration or threadStart()
/// fails. appLogic() and appShutdown() are then run against hand-built threads so the
/// thread-alive, thread-exited, and blocked-in-syscall cases can each be forced.
/**
 * \ingroup shmimMonitor_tests
 */
SCENARIO( "shmimMonitor appStartup, appLogic, appShutdown", "[dev::shmimMonitor]" )
{
    GIVEN( "a default-configured shmimMonitor" )
    {
        WHEN( "appStartup succeeds" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smStartupOk" );

            int rv = pdt.appStartup();
            REQUIRE( rv == 0 );
            REQUIRE( pdt.smThreadJoinable() == true );

            // The real monitor thread is now blocked waiting for state() to become
            // OPERATING, which never happens here. appShutdown() only signals and joins
            // the thread. It does not set the shutdown flag itself, because that is the
            // job of the main app loop. So the flag must be set first or the join below
            // would hang forever.
            pdt.setShutdownFlag( 1 );

            rv = pdt.appShutdown();
            REQUIRE( rv == 0 );
            REQUIRE( pdt.smThreadJoinable() == false );

            // Calling appShutdown() again on an already-joined thread is a no-op.
            rv = pdt.appShutdown();
            REQUIRE( rv == 0 );
        }

        WHEN( "registerIndiPropertyNew fails for the shmimName property" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smStartupFailReg1" );
            pdt.m_regFailAt = 1; // sm_shmimName is the first property appStartup() registers.

            int rv = pdt.appStartup();
            REQUIRE( rv == -1 );
            REQUIRE( pdt.smThreadJoinable() == false );
        }

        WHEN( "registerIndiPropertyNew fails for the frameSize property" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smStartupFailReg2" );
            pdt.m_regFailAt = 2; // sm_frameSize is the second property appStartup() registers.

            int rv = pdt.appStartup();
            REQUIRE( rv == -1 );
            REQUIRE( pdt.smThreadJoinable() == false );
        }

        WHEN( "threadStart fails" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smStartupFailThread" );
            pdt.m_failThreadStart = true;

            int rv = pdt.appStartup();
            REQUIRE( rv == -1 );
            REQUIRE( pdt.smThreadJoinable() == false );
        }
    }

    GIVEN( "appLogic is called directly against a controlled m_smThread" )
    {
        WHEN( "the thread is still running" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            std::atomic<bool> stayAlive{ true };
            pdt.setSmThread( std::thread(
                [&stayAlive]()
                {
                    while( stayAlive )
                        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                } ) );

            int rv = pdt.appLogic();
            REQUIRE( rv == 0 );

            stayAlive = false;
            pdt.joinMonitorThread();
        }

        WHEN( "the thread has already exited" )
        {
            smTest pdt;

            pdt.setSmThread( std::thread( [](){} ) );

            // Give the thread time to finish running, not just to be started.
            REQUIRE( waitFor(
                [&pdt]()
                {
                    // appLogic() itself performs the destructive tryjoin check, so the
                    // test cannot poll for thread exit without consuming it. The
                    // predicate is always true and the sleep below gives it a moment.
                    return true;
                },
                100 ) );
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

            int rv = pdt.appLogic();
            REQUIRE( rv == -1 );

            // The pthread_tryjoin_np() call inside appLogic() has already reaped the OS
            // thread. std::thread does not know that, so it must be neutralized without
            // a real join() or detach(). See the abandonSmThread() comment.
            pdt.abandonSmThread();
            REQUIRE( pdt.smThreadJoinable() == false );
        }
    }

    GIVEN( "appShutdown is called directly" )
    {
        WHEN( "no thread was ever started" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            int rv = pdt.appShutdown();
            REQUIRE( rv == 0 );
        }

        WHEN( "a joinable thread is blocked and gets signaled" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            // The thread must actually be blocked in pause() before appShutdown() sends
            // SIGUSR1. If the signal arrives while the thread is still starting up and
            // has not reached pause(), the no-op handler consumes it right there. pause()
            // then blocks forever because no second signal ever comes. Synchronize on a
            // flag set immediately before the pause() call to make that race negligible.
            std::atomic<bool> aboutToPause{ false };
            pdt.setSmThread( std::thread(
                [&aboutToPause]()
                {
                    aboutToPause = true;
                    pause();
                } ) );
            REQUIRE( waitFor( [&aboutToPause]() { return aboutToPause.load(); }, 1000 ) );
            std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );

            int rv = pdt.appShutdown();
            REQUIRE( rv == 0 );
            REQUIRE( pdt.smThreadJoinable() == false );
        }
    }
}

/// Verify that shmimMonitor::create() makes a new stream, fills it with initial data
/// when given, replaces an existing stream, and fails on a corrupt file or an invalid
/// data type. The results are checked by opening the stream with milkImage.
/**
 * \ingroup shmimMonitor_tests
 */
SCENARIO( "shmimMonitor create()", "[dev::shmimMonitor]" )
{
    GIVEN( "a shmimMonitor configured with a shmim name" )
    {
        smTest pdt;
        pdt.setShmimName( "smCreateTest" );

        WHEN( "creating a fresh image with no initial data" )
        {
            int rv = pdt.doCreate( 12, 8, 3, IMAGESTRUCT_FLOAT );
            REQUIRE( rv == 0 );

            mx::improc::milkImage<float> chk;
            bool opened = false;
            try
            {
                chk.open( "smCreateTest" );
                opened = true;
            }
            catch( ... )
            {
            }
            REQUIRE( opened == true );
            REQUIRE( chk.rows() == 12 );
            REQUIRE( chk.cols() == 8 );
        }

        WHEN( "creating with initial data" )
        {
            std::vector<float> initData( 12 * 8, 3.5f );
            int rv = pdt.doCreate( 12, 8, 3, IMAGESTRUCT_FLOAT, initData.data() );
            REQUIRE( rv == 0 );

            mx::improc::milkImage<float> chk;
            chk.open( "smCreateTest" );
            REQUIRE( chk( 0, 0 ) == 3.5f );
        }

        WHEN( "creating again over an existing image (destroy-then-recreate path)" )
        {
            int rv = pdt.doCreate( 12, 8, 3, IMAGESTRUCT_FLOAT );
            REQUIRE( rv == 0 );

            rv = pdt.doCreate( 20, 5, 2, IMAGESTRUCT_FLOAT );
            REQUIRE( rv == 0 );

            mx::improc::milkImage<float> chk;
            chk.open( "smCreateTest" );
            REQUIRE( chk.rows() == 20 );
            REQUIRE( chk.cols() == 5 );
        }

        WHEN( "the existing file can't be opened by ImageStreamIO (corrupt)" )
        {
            // Put a too-small junk file where the shmim would be. A raw open() succeeds
            // because it is a real file, but ImageStreamIO_openIm fails because the file
            // is smaller than IMAGE_METADATA.
            std::string path = shmimPath( "smCreateCorrupt" );
            FILE *f = fopen( path.c_str(), "w" );
            REQUIRE( f != nullptr );
            fputs( "x", f );
            fclose( f );

            pdt.setShmimName( "smCreateCorrupt" );
            int rv = pdt.doCreate( 4, 4, 2, IMAGESTRUCT_FLOAT );
            REQUIRE( rv == -1 );
        }

        WHEN( "ImageStreamIO_createIm_gpu itself fails (invalid datatype code)" )
        {
            pdt.setShmimName( "smCreateBadType" );
            int rv = pdt.doCreate( 4, 4, 2, 255 ); // 255 is not a valid ImageStreamIO datatype code.
            REQUIRE( rv == -1 );
        }
    }
}

/// Verify that updateINDI() returns 0 with no INDI driver and also publishes its
/// properties when a real indiDriver with no FIFOs is attached.
/**
 * \ingroup shmimMonitor_tests
 */
SCENARIO( "shmimMonitor updateINDI", "[dev::shmimMonitor]" )
{
    GIVEN( "a shmimMonitor with no INDI driver connected (MagAOXApp<false>)" )
    {
        smTest pdt;

        int rv = pdt.updateINDI();
        REQUIRE( rv == 0 );
    }

    GIVEN( "a shmimMonitor with a real (FIFO-less) indiDriver connected" )
    {
        // indi::updateIfChanged() catches its own send failures internally. Those are
        // covered directly in indiUtils_test.cpp. So a driver with no FIFOs is enough to
        // exercise the publishing branch of updateINDI() without a live INDI server.
        smTest pdt;
        ThreadGuard guard( pdt );
        pdt.setShmimName( "smUpdateIndi" );
        pdt.m_configName = "smUpdateIndiTest";
        pdt.setupRealDriver();

        REQUIRE( pdt.appStartup() == 0 );

        int rv = pdt.updateINDI();
        REQUIRE( rv == 0 );
    }
}

/// Verify the monitoring thread smThreadExec() against real ImageStreamIO shared
/// memory streams. Each case starts the thread, creates or damages a stream, writes
/// frames by hand, and polls the harness counters and state with waitFor(). The cases
/// cover connecting, every naxis value, corrupt and under-provisioned streams, restart
/// and state interruptions, allocate() and processImage() failures, metadata changes,
/// a vanished or replaced stream, and semaphore exhaustion. Several cases sleep past
/// the one second semaphore timeout inside the monitor loop.
/**
 * \ingroup shmimMonitor_tests
 */
SCENARIO( "shmimMonitor thread lifecycle", "[dev::shmimMonitor]" )
{
    GIVEN( "a shmimMonitor waiting for a shmim that does not exist yet" )
    {
        WHEN( "the shmim appears, frames are processed, and shutdown is clean via EINTR" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeA" );
            pdt.state( stateCodes::OPERATING );
            pdt.startMonitorThread();

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::notfound; } ) );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeA", 3, 6, 4, 2 ) == 0 ); // naxis=3, 6x4, depth 2.

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_allocateCount == 1; } ) );
            REQUIRE( pdt.width() == 6 );
            REQUIRE( pdt.height() == 4 );
            REQUIRE( pdt.depth() == 2 );

            writeFrame( img, 6, 4, 0, 42.0f );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() == 1; } ) );

            // Clean shutdown. SIGUSR1 interrupts the blocked sem_timedwait, which is the EINTR path.
            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            REQUIRE( pdt.smThreadJoinable() == false );

            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "images with different axis counts" )
    {
        WHEN( "naxis==1: height and depth both collapse to 1" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeNaxis1" );
            pdt.state( stateCodes::OPERATING );
            pdt.setGetExistingFirst( true );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeNaxis1", 1, 10, 0, 0 ) == 0 );
            writeFrame( img, 10, 1, 0, 1.0f ); // A pre-existing frame, delivered through getExistingFirst.

            pdt.startMonitorThread();

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );
            REQUIRE( pdt.height() == 1 );
            REQUIRE( pdt.depth() == 1 );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() >= 1; } ) );

            writeFrame( img, 10, 1, 0, 2.0f ); // A live frame, delivered through the main loop.
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() >= 2; } ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }

        WHEN( "naxis==2: depth collapses to 1" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeNaxis2" );
            pdt.state( stateCodes::OPERATING );
            pdt.setGetExistingFirst( true );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeNaxis2", 2, 8, 5, 0 ) == 0 );
            writeFrame( img, 8, 5, 0, 1.0f );

            pdt.startMonitorThread();

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );
            REQUIRE( pdt.width() == 8 );
            REQUIRE( pdt.height() == 5 );
            REQUIRE( pdt.depth() == 1 );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() >= 1; } ) );

            writeFrame( img, 8, 5, 0, 2.0f );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() >= 2; } ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a corrupt file where the shmim should be" )
    {
        WHEN( "ImageStreamIO_openIm fails and retries until a valid image replaces it" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeCorrupt" );
            pdt.state( stateCodes::OPERATING );

            std::string path = shmimPath( "smLifeCorrupt" );
            FILE *f = fopen( path.c_str(), "w" );
            REQUIRE( f != nullptr );
            fputs( "x", f );
            fclose( f );

            pdt.startMonitorThread();

            // Give the thread time to hit the retry where raw open() succeeds but
            // ImageStreamIO_openIm fails.
            std::this_thread::sleep_for( std::chrono::milliseconds( 1200 ) );
            REQUIRE( pdt.smState() != MAPPNS::shmimMonitorState::connected );

            remove( path.c_str() );
            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeCorrupt", 3, 4, 4, 1 ) == 0 );

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; }, 5000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "an image created with too few semaphores" )
    {
        WHEN( "it retries until destroyed and recreated with a valid semaphore count" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeSemWait" );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeSemWait", 3, 4, 4, 1, 1 ) == 0 ); // nbsem=1 is less than SEMAPHORE_MAXVAL.

            pdt.startMonitorThread();

            std::this_thread::sleep_for( std::chrono::milliseconds( 1200 ) );
            REQUIRE( pdt.smState() != MAPPNS::shmimMonitorState::connected );

            ImageStreamIO_destroyIm( &img );
            REQUIRE( rawCreate( img, "smLifeSemWait", 3, 4, 4, 1 ) == 0 ); // The default nbsem is IMAGE_NB_SEMAPHORE.

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; }, 5000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a shmimMonitor searching for a shmim that never appears" )
    {
        WHEN( "m_restart interrupts the search, then a state change interrupts it again" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeRestartSearch" );
            pdt.state( stateCodes::OPERATING );
            pdt.startMonitorThread();

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::notfound; } ) );

            // First, trip the "if (m_restart) continue;" check while searching.
            pdt.setRestart( true );
            REQUIRE( waitFor( [&pdt]() { return pdt.getRestart() == false; } ) ); // Reset at the top of the next outer pass.

            // Second, trip the "if (state()!=target) continue;" check while searching.
            pdt.state( stateCodes::READY );
            std::this_thread::sleep_for( std::chrono::milliseconds( 1200 ) );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeRestartSearch", 3, 4, 4, 1 ) == 0 );
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; }, 5000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }

        WHEN( "shutdown is requested while waiting for the target state" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeShutdownWait" );
            pdt.state( stateCodes::READY ); // Not the target state, so the thread spins in the "wait for state" loop.
            pdt.startMonitorThread();

            std::this_thread::sleep_for( std::chrono::milliseconds( 300 ) );
            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            REQUIRE( pdt.smThreadJoinable() == false );
        }

        WHEN( "shutdown is requested while searching for the shmim (never opened)" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeShutdownSearch" );
            pdt.state( stateCodes::OPERATING );
            pdt.startMonitorThread();

            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::notfound; } ) );
            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            REQUIRE( pdt.smThreadJoinable() == false );
        }
    }

    GIVEN( "a connectable shmim and a derived class that fails allocate()" )
    {
        WHEN( "allocate() failing ends the monitoring thread" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeAllocFail" );
            pdt.state( stateCodes::OPERATING );
            pdt.m_failAllocate = true;

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeAllocFail", 3, 4, 4, 1 ) == 0 );

            pdt.startMonitorThread();

            REQUIRE( waitFor( [&pdt]() { return pdt.m_allocateCount == 1; } ) );
            pdt.joinMonitorThread(); // The allocate() failure breaks out and the thread ends on its own.
            REQUIRE( pdt.smThreadJoinable() == false );

            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a connectable shmim and a derived class that fails processImage()" )
    {
        WHEN( "processImage() failing just logs and the loop keeps running" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeProcFail" );
            pdt.state( stateCodes::OPERATING );
            pdt.m_failProcessImage = true;

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeProcFail", 3, 4, 4, 2 ) == 0 );

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );

            writeFrame( img, 4, 4, 0, 1.0f );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() == 1; } ) );

            writeFrame( img, 4, 4, 1, 2.0f );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() == 2; } ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }

        WHEN( "processImage() failing on the getExistingFirst frame just logs" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeProcFailExisting" );
            pdt.state( stateCodes::OPERATING );
            pdt.setGetExistingFirst( true );
            pdt.m_failProcessImage = true;

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeProcFailExisting", 3, 4, 4, 2 ) == 0 );
            writeFrame( img, 4, 4, 0, 1.0f ); // A pre-existing frame, delivered through getExistingFirst.

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() == 1; } ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a connected shmim and a pending shutdown request" )
    {
        WHEN( "shutdown is noticed right after a successful semaphore wait, before processImage()" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifePreProcessBreak" );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifePreProcessBreak", 3, 4, 4, 2 ) == 0 );

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );

            writeFrame( img, 4, 4, 0, 1.0f );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() == 1; } ) );

            // Request shutdown, then deliver one more frame. sem_timedwait succeeds and
            // the size check passes, but the shutdown, restart, and state check right
            // after it now trips. That breaks out before processImage() is called.
            pdt.setShutdownFlag( 1 );
            writeFrame( img, 4, 4, 1, 2.0f );

            pdt.joinMonitorThread();
            REQUIRE( pdt.m_processImageCount.load() == 1 ); // The second frame was never handed to processImage().

            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a connected shmim whose metadata is mutated mid-stream" )
    {
        WHEN( "a size mismatch detected in the main loop triggers a full reconnect" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeMismatch" );
            pdt.state( stateCodes::OPERATING );
            pdt.m_mutateAtProcessCount = 1; // Mutate the size and repost right after the first processImage() call.

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeMismatch", 3, 4, 4, 2 ) == 0 );

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );

            writeFrame( img, 4, 4, 0, 1.0f );
            REQUIRE( waitFor( [&pdt]() { return pdt.m_allocateCount == 2; }, 5000 ) ); // The mismatch forces a reconnect.

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }

        WHEN( "getExistingFirst reads a stale frame whose metadata was mutated during allocate()" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeExistingMismatch" );
            pdt.state( stateCodes::OPERATING );
            pdt.setGetExistingFirst( true );
            pdt.m_mutateOnAllocate = true;

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeExistingMismatch", 3, 4, 4, 2 ) == 0 );
            writeFrame( img, 4, 4, 0, 9.0f ); // A pre-existing frame written before the monitor ever starts.

            pdt.startMonitorThread();

            // The first allocate() mutates size[0]. The getExistingFirst size check then
            // fails, the loop continues, the monitor reconnects, and allocate() runs again.
            REQUIRE( waitFor( [&pdt]() { return pdt.m_allocateCount >= 2; }, 5000 ) );
            // After reconnecting with consistent metadata, the still-present frame is delivered.
            REQUIRE( waitFor( [&pdt]() { return pdt.m_processImageCount.load() >= 1; }, 3000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a connected shmim whose source cleans up" )
    {
        WHEN( "sem<=0 triggers a break, and reconnecting waits for a fresh, valid image" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeSemZero" );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeSemZero", 3, 4, 4, 1 ) == 0 );

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );

            pdt.corruptSemCount(); // Simulates the source process having cleaned up.

            // Wait out the roughly one second sem_timedwait timeout so the "sem<=0" break is hit.
            std::this_thread::sleep_for( std::chrono::milliseconds( 1500 ) );

            // The reconnect attempt now finds sem below SEMAPHORE_MAXVAL forever, because
            // that field cannot be fixed in place. So destroy and recreate the stream properly.
            ImageStreamIO_destroyIm( &img );
            REQUIRE( rawCreate( img, "smLifeSemZero", 3, 4, 4, 1 ) == 0 );

            REQUIRE( waitFor(
                [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected && pdt.m_allocateCount == 2; },
                5000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "a connected shmim that disappears or is replaced" )
    {
        WHEN( "the shmim file is deleted mid-connection" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeFileGone" );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeFileGone", 3, 4, 4, 1 ) == 0 );

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );

            remove( shmimPath( "smLifeFileGone" ).c_str() );

            // Wait out the roughly one second sem_timedwait timeout so restart detection trips.
            std::this_thread::sleep_for( std::chrono::milliseconds( 1500 ) );
            REQUIRE( pdt.smState() != MAPPNS::shmimMonitorState::connected );

            REQUIRE( rawCreate( img, "smLifeFileGone", 3, 4, 4, 1 ) == 0 );
            REQUIRE( waitFor(
                [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected && pdt.m_allocateCount == 2; },
                5000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }

        WHEN( "the shmim is replaced with a new inode under the same name" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeInodeChange" );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeInodeChange", 3, 4, 4, 1 ) == 0 );

            pdt.startMonitorThread();
            REQUIRE( waitFor( [&pdt]() { return pdt.smState() == MAPPNS::shmimMonitorState::connected; } ) );

            // Replace the file with a fresh one that has a new inode right away. By the
            // time the roughly one second timeout check runs, both open() and stat()
            // succeed against the new file. This isolates the inode-only restart branch
            // from the file-missing one.
            ImageStreamIO_destroyIm( &img );
            REQUIRE( rawCreate( img, "smLifeInodeChange", 3, 4, 4, 1 ) == 0 );

            REQUIRE( waitFor( [&pdt]() { return pdt.m_allocateCount == 2; }, 5000 ) );

            pdt.setShutdownFlag( 1 );
            pdt.killMonitorThread();
            pdt.joinMonitorThread();
            ImageStreamIO_closeIm( &img );
        }
    }

    GIVEN( "an image whose semaphores are all held by another process" )
    {
        WHEN( "getsemwaitindex finds none available, and the thread exits" )
        {
            smTest pdt;
            ThreadGuard guard( pdt );

            pdt.setShmimName( "smLifeNoSem" );
            pdt.state( stateCodes::OPERATING );

            IMAGE img;
            REQUIRE( rawCreate( img, "smLifeNoSem", 3, 4, 4, 1 ) == 0 );
            exhaustSemaphores( img );

            pdt.startMonitorThread();
            pdt.joinMonitorThread(); // getsemwaitindex fails, the thread logs a critical error, and it returns on its own.
            REQUIRE( pdt.smThreadJoinable() == false );
            REQUIRE( pdt.smState() != MAPPNS::shmimMonitorState::connected );

            ImageStreamIO_closeIm( &img );
        }
    }
}
