/** \file dmRecon_test.cpp
 * \brief Offline dmRecon calibration, snapshot, reconstruction, publication and timing regressions.
 * \ingroup dmRecon_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../dmReconMath.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#include <sched.h>

/// \cond DMRECON_APP_TEST_HARNESS
namespace dmReconAppHarness
{
std::mutex              inverseMutex;           ///< Protects the test-only inverse gate and completion notifications.
std::condition_variable inverseCondition;       ///< Wakes blocked test inversions and observers.
std::atomic<uint64_t>   inverseCalls{ 0 };      ///< Number of actual inverse attempts, including rejected input.
std::atomic<int>        inverseScheduler{ -1 }; ///< Scheduling policy observed by the latest inverse caller.
bool                    inversePaused{ false }; ///< Test-only pause before the real math helper, under inverseMutex.

/// Release an inverse even if a test assertion aborts while the worker is paused.
class InversePause
{
  public:
    /// Hold subsequent inverse attempts until explicitly released or destroyed.
    InversePause();

    /// Release any blocked worker before its owning app is destroyed.
    ~InversePause();

    /// Release the test gate and wake a blocked inverse.
    void release();
};

InversePause::InversePause()
{
    std::lock_guard<std::mutex> lock( inverseMutex );
    inversePaused = true;
}

InversePause::~InversePause()
{
    release();
}

void InversePause::release()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( inverseMutex );
        inversePaused = false;
    }

    inverseCondition.notify_all();
}

/// Wait at most two seconds for a specified number of inverse attempts.
bool waitForInverse( uint64_t count /**< [in] Minimum total call count. */ )
{
    std::unique_lock<std::mutex> lock( inverseMutex );
    return inverseCondition.wait_for( lock,
                                      std::chrono::seconds( 2 ),
                                      [&]
                                      {
                                          return inverseCalls.load() >= count;
                                      } );
}
} // namespace dmReconAppHarness

namespace MagAOX
{
namespace app
{
namespace dmReconMath
{
/// Observe and optionally pause the app's inverse call, then run the unmodified production math helper.
inline int
dmReconTestBuildInverse( mx::improc::eigenImage<float>       &inverse /**< [out] Validated inverse. */,
                         float                               &condition /**< [out] Condition estimate. */,
                         int                                 &rejected /**< [out] Rejected mode count. */,
                         const mx::improc::eigenCube<float>  &modes /**< [in] Private mode cube. */,
                         const std::vector<size_t>           &indices /**< [in] Active pixel indices. */,
                         const mx::improc::eigenImage<float> &response /**< [in] Optional response matrix. */,
                         int          inverseNumModes /**< [in] Singular modes retained, or zero for all. */,
                         std::string &error /**< [out] Validation or decomposition failure. */ )
{
    { //mutex scope
        std::unique_lock<std::mutex> lock( dmReconAppHarness::inverseMutex );
        dmReconAppHarness::inverseScheduler = sched_getscheduler( 0 );
        ++dmReconAppHarness::inverseCalls;
        dmReconAppHarness::inverseCondition.notify_all();
        dmReconAppHarness::inverseCondition.wait( lock,
                                                  []
                                                  {
                                                      return !dmReconAppHarness::inversePaused;
                                                  } );
    }

    return buildInverse( inverse, condition, rejected, modes, indices, response, inverseNumModes, error );
}
} // namespace dmReconMath
} // namespace app
} // namespace MagAOX
/// \endcond

#define buildInverse dmReconTestBuildInverse
#include "../dmRecon.hpp"
#undef buildInverse

/// \cond DMRECON_APP_TEST_HARNESS
namespace MagAOX
{
namespace app
{

/// Expose real callbacks and an optional local calibration worker without starting instrument services.
class dmRecon_test : public dmRecon
{
  public:
    using dmRecon::buildCalibration;
    using dmRecon::FGTimingSource;
    using dmRecon::Frame;
    using dmRecon::m_beforeNext;
    using dmRecon::m_calibrationExited;
    using dmRecon::m_calibrationGeneration;
    using dmRecon::m_calibrationUsable;
    using dmRecon::m_coalesced;
    using dmRecon::m_commandPending;
    using dmRecon::m_currentCommand;
    using dmRecon::m_dryTimingStats;
    using dmRecon::m_fgTimingSource;
    using dmRecon::m_inputGeneration;
    using dmRecon::m_late;
    using dmRecon::m_latestTrigger;
    using dmRecon::m_maskIDX;
    using dmRecon::m_maskRetry;
    using dmRecon::m_maskSnapshot;
    using dmRecon::m_maskValid;
    using dmRecon::m_modesRetry;
    using dmRecon::m_modesSnapshot;
    using dmRecon::m_modesValid;
    using dmRecon::m_modevals;
    using dmRecon::m_outputChanged;
    using dmRecon::m_pendingCommand;
    using dmRecon::m_pendingMask;
    using dmRecon::m_pendingModes;
    using dmRecon::m_PInv;
    using dmRecon::m_publicationLock;
    using dmRecon::m_rejected;
    using dmRecon::m_setupStage;
    using dmRecon::m_smSemaphore;
    using dmRecon::m_triggerClock;
    using dmRecon::m_triggerClockHigh;
    using dmRecon::m_triggerClockLow;
    using dmRecon::m_triggerClockValid;
    using dmRecon::m_unknown;
    using dmRecon::m_workerGeneration;
    using dmRecon::m_writeDMf;
    using dmRecon::outputCompatible;
    using dmRecon::publishDiagnostics;
    using dmRecon::reconstruct;
    using dmRecon::recordCompletion;
    using dmRecon::recordMonitorTiming;
    using dmRecon::setCalibrationStage;
    using dmRecon::snapshot;
    using dmRecon::SnapshotStatus;
    using dmRecon::takeCommand;
    using frameGrabberT::m_fgTiming;

    /// Build synchronously for existing fixtures, then consume through the production acquisition method.
    int prepareCalibration();

    /// Invoke acquisition preparation alone, which must never compute an inverse.
    int prepareOnly();

    /// Start the production calibration entry point using an ordinary process-local thread.
    void startCalibrationWorker( bool release = true /**< [in] Release the normal startup gate immediately. */ );

    /// Wait for the latest valid calibration result without consuming or computing it.
    bool waitForCalibration();

    /// Wake the worker without replacing its current calibration generation.
    void wakeCalibrationWorker();

    /// Inspect the calibration generation most recently attempted under snapshot ownership.
    uint64_t attemptedCalibration();

    /// Stop and join the production worker without touching any stream or app service.
    using dmRecon::stopCalibrationThread;

    /// Initialize only the local wake semaphore and callback property.
    dmRecon_test();

    /// Release retained publication ownership and detach all borrowed images.
    ~dmRecon_test() noexcept;

    /// Attach a local complete mode cube to the real monitor callback.
    void modesSource( const IMAGE &source /**< [in] Borrowed fixture image. */ );

    /// Attach a local mask to the real monitor callback.
    void maskSource( const IMAGE &source /**< [in] Borrowed fixture image. */ );

    /// Attach a local command to the real monitor callback.
    void commandSource( const IMAGE &source /**< [in] Borrowed fixture image. */ );

    /// Attach local metadata to the real loop-trigger observer without opening a stream.
    void triggerSource( const IMAGE &source /**< [in] Borrowed trigger fixture image. */ );

    /// Accumulate deterministic dry timing updates under the production snapshot lock.
    void applyDryUpdate( const dmReconTiming::Update &update /**< [in] Completed timing classifications. */ );

    /// Supply borrowed output metadata for the real final publication checks.
    void output( IMAGE *source /**< [in] Borrowed fixture image. */ );

    /// Select the normal staged copy or direct final copy for local publication tests.
    void staged( bool value /**< [in] Whether the frame grabber stages its output. */ );

    /// Invoke the real operator-enable callback without registering or connecting INDI.
    int enable( bool value /**< [in] Requested enable state. */ );

    /// Inspect the retry request belonging to one calibration monitor.
    bool
    calibrationRestart( bool modes /**< [in] True selects the mode-cube monitor; false selects the mask. */ ) const;

    /// Simulate the monitor consuming its reconnect flag and allocating against the same borrowed source.
    int restartCalibrationMonitor( bool modes /**< [in] True selects modes; false selects the mask. */ );
};

dmRecon_test::dmRecon_test()
{
    REQUIRE( sem_init( &m_smSemaphore, 0, 0 ) == 0 );
    m_semaphoreInitialized = true;
    m_configName           = "offline-dmRecon";
    createStandardIndiToggleSw( m_indiP_writeDMf, "writeDMf" );
}

dmRecon_test::~dmRecon_test() noexcept
{
    stopCalibrationThread();
    reconfig();
    dmModesSMT::m_imageStream     = {};
    dmMaskSMT::m_imageStream      = {};
    dmCommandSMT::m_imageStream   = {};
    loopTriggerSMT::m_imageStream = {};
    frameGrabberT::m_imageStream  = nullptr;
}

int dmRecon_test::prepareCalibration()
{
    buildCalibration();
    return dmRecon::prepareCalibration();
}

int dmRecon_test::prepareOnly()
{
    return dmRecon::prepareCalibration();
}

void dmRecon_test::startCalibrationWorker( bool release )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );
        m_calibrationStart  = release;
        m_calibrationStop   = false;
        m_calibrationExited = false;
    }

    m_calibrationThread = std::thread( calibrationThreadStart, this );
    m_calibrationCondition.notify_all();
}

bool dmRecon_test::waitForCalibration()
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );

    while( std::chrono::steady_clock::now() < deadline )
    {
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_snapshotMutex );

            if( m_calibrationResult && m_calibrationResult->generation == m_calibrationGeneration )
            {
                return true;
            }
        }

        std::this_thread::yield();
    }

    return false;
}

void dmRecon_test::wakeCalibrationWorker()
{
    m_calibrationCondition.notify_all();
}

uint64_t dmRecon_test::attemptedCalibration()
{
    std::lock_guard<std::mutex> lock( m_snapshotMutex );
    return m_calibrationAttempted;
}

void dmRecon_test::modesSource( const IMAGE &source )
{
    dmModesSMT::m_imageStream = source;
    dmModesSMT::m_depth       = source.md->naxis == 3 ? source.md->size[2] : 1;
}

void dmRecon_test::maskSource( const IMAGE &source )
{
    dmMaskSMT::m_imageStream = source;
    dmMaskSMT::m_depth       = source.md->naxis == 3 ? source.md->size[2] : 1;
}

void dmRecon_test::commandSource( const IMAGE &source )
{
    dmCommandSMT::m_imageStream = source;
    dmCommandSMT::m_depth       = source.md->naxis == 3 ? source.md->size[2] : 1;
}

void dmRecon_test::triggerSource( const IMAGE &source )
{
    loopTriggerSMT::m_imageStream = source;
}

void dmRecon_test::applyDryUpdate( const dmReconTiming::Update &update )
{
    std::lock_guard<std::mutex> lock( m_snapshotMutex );
    dryTimingUpdate( update );
}

void dmRecon_test::output( IMAGE *source )
{
    frameGrabberT::m_imageStream = source;
}

void dmRecon_test::staged( bool value )
{
    frameGrabberT::m_staged = value;
}

int dmRecon_test::enable( bool value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch );
    property.setDevice( m_configName );
    property.setName( "writeDMf" );
    property.add( pcf::IndiElement( "toggle" ) );
    property["toggle"].setSwitchState( value ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return newCallBack_m_indiP_writeDMf( property );
}

bool dmRecon_test::calibrationRestart( bool modes ) const
{
    return modes ? dmModesSMT::m_restart : dmMaskSMT::m_restart;
}

int dmRecon_test::restartCalibrationMonitor( bool modes )
{
    if( modes )
    {
        dmModesSMT::m_restart = false;
        return allocate( dmModesShmimT{} );
    }

    dmMaskSMT::m_restart = false;
    return allocate( dmMaskShmimT{} );
}

} // namespace app
} // namespace MagAOX

namespace dmReconAppHarness
{
using MagAOX::app::dmCommandShmimT;
using MagAOX::app::dmLoopTriggerShmimT;
using MagAOX::app::dmMaskShmimT;
using MagAOX::app::dmModesShmimT;
using MagAOX::app::dmRecon_test;
using MagAOX::app::dmReconTiming;

/// Local metadata and vector storage only: never opened, created, posted or destroyed through ImageStreamIO.
struct Image
{
    IMAGE_METADATA     metadata{}; ///< Borrowed metadata for the duration of each test.
    std::vector<float> pixels;     ///< Private backing pixels.
    IMAGE              image{};    ///< Non-owning image handle.

    /// Initialize a private float image with a valid publication identity.
    Image( uint32_t width /**< [in] First image dimension. */,
           uint32_t height /**< [in] Second image dimension. */,
           uint32_t depth = 1 /**< [in] Number of planes. */ );
};

Image::Image( uint32_t width, uint32_t height, uint32_t depth ) : pixels( size_t( width ) * height * depth )
{
    metadata.datatype      = _DATATYPE_FLOAT;
    metadata.naxis         = depth == 1 ? 2 : 3;
    metadata.location      = -1;
    metadata.size[0]       = width;
    metadata.size[1]       = height;
    metadata.size[2]       = depth == 1 ? 0 : depth;
    metadata.nelement      = pixels.size();
    metadata.imdatamemsize = pixels.size() * sizeof( float );
    metadata.cnt0          = 1;
    metadata.inode         = 42;
    metadata.writetime     = { 10, 100 };
    image.md               = &metadata;
    image.array.raw        = pixels.data();
}

/// Own a two-mode CPU reconstruction and all borrowed input/output image storage.
struct Fixture
{
    Image        modes{ 2, 2, 2 };    ///< Two independent modes.
    Image        mask{ 2, 2 };        ///< Asymmetric active-pixel selection.
    Image        command{ 2, 2 };     ///< Applied shape with known mode amplitudes.
    Image        destination{ 2, 1 }; ///< Private feedback-vector destination.
    Image        trigger{ 1, 1 };     ///< Borrowed loop-trigger metadata for timing tests.
    dmRecon_test app;                 ///< Unstarted app with no external resources.

    /// Stage real mode/mask callbacks, prepare the inverse, and enable local publication.
    Fixture();

    /// Capture the current command using the production callback.
    int receive();
};

Fixture::Fixture()
{
    modes.pixels[0]         = 2;
    modes.pixels[6]         = 4;
    mask.pixels[0]          = 1;
    mask.pixels[2]          = 1;
    command.pixels          = { 6, 99, 20, 88 };
    command.image.array.raw = command.pixels.data();
    app.modesSource( modes.image );
    app.maskSource( mask.image );
    app.commandSource( command.image );
    app.output( &destination.image );
    REQUIRE( app.allocate( dmModesShmimT{} ) == 0 );
    REQUIRE( app.allocate( dmMaskShmimT{} ) == 0 );
    REQUIRE( app.processImage( modes.pixels.data() + 4, dmModesShmimT{} ) == 0 );
    REQUIRE( app.processImage( mask.pixels.data(), dmMaskShmimT{} ) == 0 );
    REQUIRE( app.prepareCalibration() == 0 );
    REQUIRE( app.allocate( dmCommandShmimT{} ) == 0 );
    REQUIRE( app.enable( true ) == 0 );
}

int Fixture::receive()
{
    return app.processImage( command.pixels.data(), dmCommandShmimT{} );
}

/// Shift a valid clock sample for deterministic ordering of local fixture timestamps.
timespec shiftedTime( const timespec &stamp /**< [in] Nonzero timestamp in the publication clock. */,
                      int64_t         delta /**< [in] Signed nanosecond shift. */ )
{
    const int64_t total = int64_t( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec + delta;
    return { total / 1000000000, total % 1000000000 };
}

/// Observe a local baseline preceding the fixture command in the actual publication clock.
void prepareTiming( Fixture &fixture /**< [in,out] Local app and borrowed trigger/command images. */ )
{
    timespec now{};
    REQUIRE( clock_gettime( CLOCK_ISIO, &now ) == 0 );
    fixture.trigger.metadata.cnt0      = 10;
    fixture.trigger.metadata.writetime = shiftedTime( now, -2000000 );
    fixture.command.metadata.writetime = shiftedTime( now, -1000000 );
    fixture.app.triggerSource( fixture.trigger.image );
    REQUIRE( fixture.app.allocate( dmLoopTriggerShmimT{} ) == 0 );
    REQUIRE( fixture.app.processImage( nullptr, dmLoopTriggerShmimT{} ) == 0 );
}

/// Advance only local trigger metadata and invoke the production observer after command completion.
void observeNextTrigger( Fixture &fixture /**< [in,out] Borrowed trigger source and receiving app. */,
                         uint64_t step = 1 /**< [in] Counter increment, allowing an intentional observation gap. */ )
{
    REQUIRE( clock_gettime( CLOCK_ISIO, &fixture.trigger.metadata.writetime ) == 0 );
    fixture.trigger.metadata.cnt0 += step;
    REQUIRE( fixture.app.processImage( nullptr, dmLoopTriggerShmimT{} ) == 0 );
}

/// Give the next local command a distinct identity and a current source timestamp.
void advanceTimedCommand( Fixture &fixture /**< [in,out] Local command metadata. */ )
{
    ++fixture.command.metadata.cnt0;
    REQUIRE( clock_gettime( CLOCK_ISIO, &fixture.command.metadata.writetime ) == 0 );
}

} // namespace dmReconAppHarness
/// \endcond

namespace libXWCTest
{
namespace dmReconTest
{
using namespace dmReconAppHarness;

/// Copy only valid float payload ranges and preserve source publication identity in private storage.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon snapshots validate image metadata and payload bounds", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::snapshot;
    #endif
    // clang-format on

    Image source( 2, 2, 2 );
    source.pixels          = { 1, 2, 3, 4, 5, 6, 7, 8 };
    source.image.array.raw = source.pixels.data();
    dmRecon_test::Frame frame;

    SECTION( "whole cube and selected frame are independent private copies" )
    {
        REQUIRE( dmRecon_test::snapshot( frame, source.image, source.pixels.data(), 2 ) );
        CHECK( frame.pixels == source.pixels );
        CHECK( frame.depth == 2 );
        CHECK( frame.counter == 1 );
        CHECK( frame.inode == 42 );
        source.pixels[0] = 99;
        CHECK( frame.pixels[0] == 1 );
        REQUIRE( dmRecon_test::snapshot( frame, source.image, source.pixels.data() + 4, 1 ) );
        CHECK( frame.pixels == std::vector<float>{ 5, 6, 7, 8 } );
        CHECK_FALSE( dmRecon_test::snapshot( frame, source.image, source.pixels.data() + 4, 2 ) );
        return;
    }

    SECTION( "wrong pixel type" )
    {
        source.metadata.datatype = _DATATYPE_UINT16;
    }

    SECTION( "GPU source" )
    {
        source.metadata.location = 0;
    }

    SECTION( "producer write in progress" )
    {
        source.metadata.write = 1;
    }

    SECTION( "truncated mapped bytes" )
    {
        source.metadata.imdatamemsize -= sizeof( float );
    }

    SECTION( "truncated element count" )
    {
        --source.metadata.nelement;
    }

    SECTION( "zero dimension" )
    {
        source.metadata.size[0] = 0;
    }

    CHECK_FALSE( dmRecon_test::snapshot( frame, source.image, source.pixels.data(), 2 ) );
}

/// Explain rejected source states so calibration waiters can distinguish incomplete writes from invalid schemas.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon snapshots report calibration rejection reasons", "[dmRecon][app][calibration]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::snapshot;
    #endif
    // clang-format on

    using Status = dmRecon_test::SnapshotStatus;
    Image               source( 2, 2 );
    dmRecon_test::Frame frame;
    Status              status = Status::unavailable;
    REQUIRE( dmRecon_test::snapshot( frame, source.image, source.pixels.data(), 1, &status ) );
    REQUIRE( status == Status::ready );
    Status expected = Status::unavailable;

    SECTION( "unavailable mapping" )
    {
        source.image.md = nullptr;
    }

    SECTION( "invalid pixel format" )
    {
        source.metadata.datatype = _DATATYPE_UINT16;
        expected                 = Status::invalidFormat;
    }

    SECTION( "invalid dimensions" )
    {
        source.metadata.size[0] = 0;
        expected                = Status::invalidShape;
    }

    SECTION( "truncated payload capacity" )
    {
        --source.metadata.imdatamemsize;
        expected = Status::invalidCapacity;
    }

    SECTION( "incomplete producer write" )
    {
        __atomic_store_n( &source.metadata.write, 1, __ATOMIC_RELEASE );
        expected = Status::busy;
    }

    CHECK_FALSE( dmRecon_test::snapshot( frame, source.image, source.pixels.data(), 1, &status ) );
    CHECK( status == expected );
}

/// Admit only a complete host-float modal vector with the exact reconstructed length.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon validates feedback destination schemas", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::outputCompatible;
    #endif
    // clang-format on

    Image output( 2, 1 );
    REQUIRE( dmRecon_test::outputCompatible( output.metadata, 2 ) );
    output.metadata.naxis   = 3;
    output.metadata.size[2] = 1;
    REQUIRE( dmRecon_test::outputCompatible( output.metadata, 2 ) );

    SECTION( "wrong datatype" )
    {
        output.metadata.datatype = _DATATYPE_UINT16;
    }

    SECTION( "nonhost data" )
    {
        output.metadata.location = 0;
    }

    SECTION( "oversized vector" )
    {
        output.metadata.size[0]       = 3;
        output.metadata.nelement      = 3;
        output.metadata.imdatamemsize = 3 * sizeof( float );
    }

    SECTION( "short vector" )
    {
        output.metadata.size[0]       = 1;
        output.metadata.nelement      = 1;
        output.metadata.imdatamemsize = sizeof( float );
    }

    SECTION( "wrong depth" )
    {
        output.metadata.size[2] = 2;
    }

    SECTION( "short byte range" )
    {
        --output.metadata.imdatamemsize;
    }

    SECTION( "short element range" )
    {
        --output.metadata.nelement;
    }

    CHECK_FALSE( dmRecon_test::outputCompatible( output.metadata, 2 ) );
    CHECK_FALSE( dmRecon_test::outputCompatible( output.metadata, 0 ) );
}

/// Reconstruct an asymmetric basis after copying the cube from its base even when the callback points to its last
/// plane.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon prepares complete private calibration generations", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    MagAOX::app::dmRecon::reconstruct;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    CHECK( app.m_pendingModes.pixels == fixture.modes.pixels );
    CHECK( app.m_maskIDX == std::vector<size_t>{ 0, 2 } );
    CHECK( app.m_calibrationUsable );
    const auto generation = app.m_workerGeneration;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    CHECK( app.m_calibrationGeneration == generation );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );
    REQUIRE( app.m_modevals.rows() == 2 );
    CHECK( app.m_modevals( 0, 0 ) == Approx( 3 ) );
    CHECK( app.m_modevals( 1, 0 ) == Approx( 5 ) );

    fixture.mask.metadata.cnt0++;
    std::fill( fixture.mask.pixels.begin(), fixture.mask.pixels.end(), 0 );
    REQUIRE( app.processImage( fixture.mask.pixels.data(), dmMaskShmimT{} ) == 0 );
    REQUIRE( app.prepareCalibration() == 1 );
    CHECK_FALSE( app.m_calibrationUsable );
    CHECK( app.attemptedCalibration() > generation );
}

/// Consume completed inverse results on acquisition without performing a new inverse there.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon acquisition only consumes completed calibration", "[dmRecon][app][calibration][worker]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    #endif
    // clang-format on

    Fixture    fixture;
    auto      &app          = fixture.app;
    const auto calls        = inverseCalls.load();
    fixture.modes.pixels[0] = 4;
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    CHECK( app.prepareOnly() == 1 );
    CHECK( inverseCalls.load() == calls );
    REQUIRE( app.buildCalibration() == 0 );
    CHECK( inverseCalls.load() == calls + 1 );
    REQUIRE( app.prepareOnly() == 0 );
    CHECK( inverseCalls.load() == calls + 1 );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );
    CHECK( app.m_modevals( 0, 0 ) == Approx( 1.5F ) );
}

/// Refuse a completed result after a newer source generation arrives before acquisition consumes it.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon rejects obsolete completed calibration results", "[dmRecon][app][calibration][worker]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::setCalibrationStage;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app             = fixture.app;
    fixture.modes.pixels[0] = 4;
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    REQUIRE( app.buildCalibration() == 0 );
    const auto obsolete     = app.attemptedCalibration();
    fixture.modes.pixels[0] = 8;
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    REQUIRE( app.setCalibrationStage( app.m_calibrationGeneration, "building_inverse" ) );
    CHECK_FALSE( app.setCalibrationStage( obsolete, "acquiring" ) );
    CHECK( std::string( app.m_setupStage.load() ) == "building_inverse" );
    CHECK( app.prepareOnly() == 1 );
    CHECK( app.m_workerGeneration != obsolete );
    REQUIRE( app.buildCalibration() == 0 );
    REQUIRE( app.prepareOnly() == 0 );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );
    CHECK( app.m_modevals( 0, 0 ) == Approx( 0.75F ) );
}

/// Discard an inverse superseded while it is running and build the latest complete calibration on the worker.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon worker replaces an obsolete in-flight inverse", "[dmRecon][app][calibration][worker]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::calibrationThreadStart;
    MagAOX::app::dmRecon::calibrationThreadExec;
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    MagAOX::app::dmRecon::stopCalibrationThread;
    #endif
    // clang-format on

    Fixture    fixture;
    auto      &app          = fixture.app;
    const auto calls        = inverseCalls.load();
    fixture.modes.pixels[0] = 4;
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    InversePause pause;
    app.startCalibrationWorker();
    REQUIRE( waitForInverse( calls + 1 ) );
    CHECK( inverseScheduler.load() == SCHED_OTHER );
    CHECK( app.prepareOnly() == 1 );
    fixture.modes.pixels[0] = 8;
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    pause.release();
    REQUIRE( app.waitForCalibration() );
    CHECK( inverseCalls.load() == calls + 2 );
    REQUIRE( app.prepareOnly() == 0 );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );
    CHECK( app.m_modevals( 0, 0 ) == Approx( 0.75F ) );
    app.stopCalibrationThread();
}

/// Attempt a numerically invalid generation once, then recover when a new valid snapshot arrives.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon worker waits for new data after inverse rejection", "[dmRecon][app][calibration][worker]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::calibrationThreadExec;
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    MagAOX::app::dmRecon::stopCalibrationThread;
    #endif
    // clang-format on

    Fixture    fixture;
    auto      &app          = fixture.app;
    const auto calls        = inverseCalls.load();
    fixture.modes.pixels[0] = std::numeric_limits<float>::quiet_NaN();
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    const auto invalid = app.m_calibrationGeneration;
    app.startCalibrationWorker();
    REQUIRE( waitForInverse( calls + 1 ) );
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );

    while( std::string( app.m_setupStage.load() ) != "calibration_error" &&
           std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::yield();
    }

    REQUIRE( std::string( app.m_setupStage.load() ) == "calibration_error" );
    REQUIRE( app.attemptedCalibration() == invalid );

    for( int attempt = 0; attempt < 10; ++attempt )
    {
        app.wakeCalibrationWorker();
        CHECK( app.prepareOnly() == 1 );
    }

    app.stopCalibrationThread();
    CHECK( inverseCalls.load() == calls + 1 );
    fixture.modes.pixels[0] = 4;
    ++fixture.modes.metadata.cnt0;
    REQUIRE( app.processImage( fixture.modes.pixels.data(), dmModesShmimT{} ) == 0 );
    app.startCalibrationWorker();
    REQUIRE( app.waitForCalibration() );
    CHECK( inverseCalls.load() == calls + 2 );
    REQUIRE( app.prepareOnly() == 0 );
    app.stopCalibrationThread();
}

/// Wake and join an idle worker both before its startup gate opens and after startup with no calibration.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon stops an idle calibration worker", "[dmRecon][app][calibration][worker]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::calibrationThreadStart;
    MagAOX::app::dmRecon::calibrationThreadExec;
    MagAOX::app::dmRecon::stopCalibrationThread;
    #endif
    // clang-format on

    const bool   release = GENERATE( true, false );
    dmRecon_test app;
    const auto   calls = inverseCalls.load();
    app.startCalibrationWorker( release );
    app.stopCalibrationThread();
    CHECK( inverseCalls.load() == calls );
    CHECK( app.m_calibrationExited.load() );
    CHECK( app.prepareOnly() == 1 );
    app.stopCalibrationThread();
}

/// Recover the static calibration after its only notification precedes completion of the producer write.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon retries calibration until an early-notified write completes", "[dmRecon][app][calibration]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::captureCalibration;
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::allocate;
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    #endif
    // clang-format on

    const bool modes = GENERATE( true, false );
    Fixture    fixture;
    auto      &app    = fixture.app;
    auto      &source = modes ? fixture.modes : fixture.mask;
    auto      &status = modes ? app.m_modesSnapshot : app.m_maskSnapshot;
    __atomic_store_n( &source.metadata.write, 1, __ATOMIC_RELEASE );
    auto       callback = std::async( std::launch::async,
                                [&]
                                {
                                    return modes ? app.processImage( source.pixels.data() + 4, dmModesShmimT{} )
                                                       : app.processImage( source.pixels.data(), dmMaskShmimT{} );
                                } );
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );

    while( status.load() != dmRecon_test::SnapshotStatus::busy && std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::yield();
    }

    const bool observedBusy = status.load() == dmRecon_test::SnapshotStatus::busy;

    if( modes )
    {
        source.pixels[0] = 4;
    }
    else
    {
        source.pixels[3] = 0.5F;
    }

    __atomic_add_fetch( &source.metadata.cnt0, 1, __ATOMIC_RELEASE );
    __atomic_store_n( &source.metadata.write, 0, __ATOMIC_RELEASE );
    const int result = callback.get();
    REQUIRE( observedBusy );
    REQUIRE( result == 0 );

    if( status.load() != dmRecon_test::SnapshotStatus::ready )
    {
        // A descheduled test thread can exceed the short retry; the monitor then revisits its existing image.
        REQUIRE( app.calibrationRestart( modes ) );
        REQUIRE( app.restartCalibrationMonitor( modes ) == 0 );
    }

    CHECK( status.load() == dmRecon_test::SnapshotStatus::ready );
    CHECK( ( modes ? app.m_modesValid : app.m_maskValid ) );
    CHECK_FALSE( app.calibrationRestart( modes ) );
    REQUIRE( app.prepareCalibration() == 0 );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );
    CHECK( app.m_modevals( 0, 0 ) == Approx( modes ? 1.5F : 3.0F ) );
}

/// Leave the callback after a bounded retry and ask its monitor to revisit a persistently invalid calibration.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon invalid calibration requests a monitor-owned retry", "[dmRecon][app][calibration]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::captureCalibration;
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::allocate;
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    #endif
    // clang-format on

    const bool modes = GENERATE( true, false );
    Fixture    fixture;
    auto      &app           = fixture.app;
    auto      &source        = modes ? fixture.modes : fixture.mask;
    auto      &status        = modes ? app.m_modesSnapshot : app.m_maskSnapshot;
    source.metadata.datatype = _DATATYPE_UINT16;
    const int result         = modes ? app.processImage( source.pixels.data(), dmModesShmimT{} )
                                     : app.processImage( source.pixels.data(), dmMaskShmimT{} );
    REQUIRE( result == 0 );
    CHECK( status.load() == dmRecon_test::SnapshotStatus::invalidFormat );
    CHECK_FALSE( ( modes ? app.m_modesValid : app.m_maskValid ) );
    CHECK( app.calibrationRestart( modes ) );
    CHECK( ( modes ? app.m_modesRetry : app.m_maskRetry ) );
    CHECK( app.prepareCalibration() == 1 );

    // A reconnect must capture an existing corrected image without waiting for another producer notification.
    source.metadata.datatype = _DATATYPE_FLOAT;
    REQUIRE( app.restartCalibrationMonitor( modes ) == 0 );
    CHECK( status.load() == dmRecon_test::SnapshotStatus::ready );
    CHECK( ( modes ? app.m_modesValid : app.m_maskValid ) );
    CHECK_FALSE( ( modes ? app.m_modesRetry : app.m_maskRetry ) );
    CHECK_FALSE( app.calibrationRestart( modes ) );
    CHECK( app.prepareCalibration() == 0 );
}

/// Distinguish missing mode and mask snapshots before any inverse or GPU work is attempted.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon reports which calibration snapshot setup is waiting for", "[dmRecon][app][calibration]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    MagAOX::app::dmRecon::allocate;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    REQUIRE( std::string( app.m_setupStage.load() ) == "calibration_ready" );
    std::string expected;

    SECTION( "mode cube missing" )
    {
        REQUIRE( app.allocate( dmModesShmimT{} ) == 0 );
        expected = "waiting_for_modes";
    }

    SECTION( "mask missing" )
    {
        REQUIRE( app.allocate( dmMaskShmimT{} ) == 0 );
        expected = "waiting_for_mask";
    }

    SECTION( "both snapshots missing" )
    {
        REQUIRE( app.allocate( dmModesShmimT{} ) == 0 );
        REQUIRE( app.allocate( dmMaskShmimT{} ) == 0 );
        expected = "waiting_for_modes_and_mask";
    }

    CHECK( app.prepareCalibration() == 1 );
    CHECK( std::string( app.m_setupStage.load() ) == expected );
}

/// Refuse inconsistent private payload sizes before allocating a cube or indexing its pixels.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon calibration rejects malformed private frame lengths", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::buildCalibration;
    MagAOX::app::dmRecon::prepareCalibration;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    ++app.m_calibrationGeneration;

    SECTION( "short mode cube" )
    {
        app.m_pendingModes.pixels.pop_back();
    }

    SECTION( "oversized mode cube" )
    {
        app.m_pendingModes.pixels.push_back( 0 );
    }

    SECTION( "short mask" )
    {
        app.m_pendingMask.pixels.pop_back();
    }

    SECTION( "oversized mask" )
    {
        app.m_pendingMask.pixels.push_back( 0 );
    }

    CHECK( app.prepareCalibration() == 1 );
    CHECK_FALSE( app.m_calibrationUsable );
}

/// Coalesce pending commands once, suppress duplicate identities and reject obsolete input attachments.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon command mailbox keeps only the latest distinct input", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::allocate;
    MagAOX::app::dmRecon::takeCommand;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( fixture.receive() == 0 );
    CHECK( app.m_coalesced.load() == 0 );
    fixture.command.metadata.cnt0++;
    fixture.command.pixels[0] = 10;
    REQUIRE( fixture.receive() == 0 );
    CHECK( app.m_coalesced.load() == 1 );
    int wakeCount = 0;
    REQUIRE( sem_getvalue( &app.m_smSemaphore, &wakeCount ) == 0 );
    CHECK( wakeCount == 1 );
    REQUIRE( app.takeCommand() );
    CHECK( app.m_currentCommand.counter == 2 );
    CHECK( app.m_currentCommand.pixels[0] == 10 );
    CHECK_FALSE( app.takeCommand() );
    REQUIRE( fixture.receive() == 0 );
    CHECK_FALSE( app.m_commandPending );
    const auto generation = app.m_inputGeneration;
    REQUIRE( app.allocate( dmCommandShmimT{} ) == 0 );
    CHECK( app.m_inputGeneration == generation + 1 );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    CHECK( app.m_currentCommand.inputGeneration == app.m_inputGeneration );
}

/// Maintain complete private payloads when the input monitor overtakes the consuming worker.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon command mailbox preserves concurrent snapshot ownership", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::takeCommand;
    #endif
    // clang-format on

    Fixture           fixture;
    std::atomic<bool> done{ false };
    std::atomic<int>  failures{ 0 };
    std::thread       producer(
        [&]
        {
            for( uint64_t counter = 1; counter <= 2000; ++counter )
            {
                fixture.command.metadata.cnt0 = counter;

                for( size_t index = 0; index < fixture.command.pixels.size(); ++index )
                {
                    fixture.command.pixels[index] = float( counter * 4 + index );
                }

                if( fixture.receive() != 0 )
                {
                    ++failures;
                }
            }

            done = true;
        } );
    uint64_t last     = 0;
    bool     coherent = true;

    while( true )
    {
        bool taken = fixture.app.takeCommand();

        if( !taken && done.load() )
        {
            taken = fixture.app.takeCommand();

            if( !taken )
            {
                break;
            }
        }

        if( taken )
        {
            const auto &frame = fixture.app.m_currentCommand;
            coherent          = coherent && frame.counter > last && frame.pixels.size() == 4;

            for( size_t index = 0; index < frame.pixels.size(); ++index )
            {
                coherent = coherent && frame.pixels[index] == float( frame.counter * 4 + index );
            }

            last = frame.counter;
        }
        else
        {
            std::this_thread::yield();
        }
    }

    producer.join();
    CHECK( coherent );
    CHECK( last == 2000 );
    CHECK( failures.load() == 0 );
}

/// Leave timing identity unknown when trigger causality or the source clock cannot be established.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon refuses unqualified trigger identities at command capture", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::processImage;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app                    = fixture.app;
    app.m_latestTrigger.generation = 7;
    app.m_latestTrigger.counter    = 22;
    app.m_latestTrigger.inode      = 44;
    app.m_latestTrigger.writeTime  = { 9, 100 };
    app.m_triggerClockLow          = std::numeric_limits<int64_t>::min() / 2;
    app.m_triggerClockHigh         = std::numeric_limits<int64_t>::max() / 2;

    SECTION( "trigger clock has not been qualified" )
    {
        app.m_triggerClockValid = false;
    }

    SECTION( "trigger timestamp is newer than the command" )
    {
        app.m_triggerClockValid       = true;
        app.m_latestTrigger.writeTime = { 11, 0 };
    }

    SECTION( "configured clock differs from ImageStreamIO publication clock" )
    {
        app.m_triggerClockValid = true;
        app.m_triggerClock      = CLOCK_MONOTONIC;
    }

    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    CHECK( app.m_currentCommand.trigger.generation == 0 );
    CHECK( app.reconstruct() == 0 );
}

/// Classify disabled reconstruction after its exact successor without changing feedback bytes or metadata.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon disabled acquisition records dry timing without publication", "[dmRecon][app][dryTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::acquireAndCheckValid;
    MagAOX::app::dmRecon::recordCompletion;
    MagAOX::app::dmRecon::processImage;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    prepareTiming( fixture );
    fixture.destination.pixels         = { -91, -92 };
    fixture.destination.metadata.cnt0  = 77;
    fixture.destination.metadata.cnt1  = 13;
    fixture.destination.metadata.write = 1;
    const auto stamp                   = fixture.destination.metadata.writetime;
    REQUIRE( app.enable( false ) == 0 );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.m_currentCommand.trigger.generation != 0 );
    CHECK( app.m_modevals( 0, 0 ) == Approx( 3 ) );
    CHECK( app.m_modevals( 1, 0 ) == Approx( 5 ) );
    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 0 );
    observeNextTrigger( fixture );
    CHECK( app.m_dryTimingStats.beforeNext == 1 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 0 );
    CHECK( app.m_dryTimingStats.marginNs > 0 );
    CHECK( app.m_dryTimingStats.minMarginNs == app.m_dryTimingStats.marginNs );
    CHECK( app.m_dryTimingStats.maxMarginNs == app.m_dryTimingStats.marginNs );
    CHECK( app.m_beforeNext.load() == 0 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
    CHECK( fixture.destination.pixels == std::vector<float>{ -91, -92 } );
    CHECK( fixture.destination.metadata.cnt0 == 77 );
    CHECK( fixture.destination.metadata.cnt1 == 13 );
    CHECK( fixture.destination.metadata.write == 1 );
    CHECK( fixture.destination.metadata.writetime.tv_sec == stamp.tv_sec );
    CHECK( fixture.destination.metadata.writetime.tv_nsec == stamp.tv_nsec );
    CHECK_FALSE( app.m_publicationLock.owns_lock() );
}

/// Preserve each completion's dry or live origin when an enable toggle occurs before its successor is observed.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon toggles preserve pending dry and live timing origins", "[dmRecon][app][dryTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::acquireAndCheckValid;
    MagAOX::app::dmRecon::frameGrabberPostPublish;
    MagAOX::app::dmRecon::recordCompletion;
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::newCallBack_m_indiP_writeDMf;
    #endif
    // clang-format on

    const bool firstDry = GENERATE( true, false );
    Fixture    fixture;
    auto      &app = fixture.app;
    prepareTiming( fixture );

    for( bool dry : { firstDry, !firstDry } )
    {
        REQUIRE( app.enable( !dry ) == 0 );
        REQUIRE( fixture.receive() == 0 );
        REQUIRE( app.acquireAndCheckValid() == ( dry ? 1 : 0 ) );
        REQUIRE( app.m_currentCommand.trigger.generation != 0 );

        if( !dry )
        {
            REQUIRE( app.loadImageIntoStream( fixture.destination.pixels.data() ) == 0 );
            REQUIRE( app.frameGrabberPostPublish( &fixture.destination.image ) == 0 );
        }

        advanceTimedCommand( fixture );
    }

    CHECK( app.m_beforeNext.load() == 0 );
    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    REQUIRE( app.enable( firstDry ) == 0 );
    observeNextTrigger( fixture );
    CHECK( app.m_beforeNext.load() == 1 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
    CHECK( app.m_dryTimingStats.beforeNext == 1 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 0 );
}

/// Keep dry timing unknown when clocks, exact successors, or trigger attachment continuity are unavailable.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon dry timing does not infer missing clock or trigger evidence", "[dmRecon][app][dryTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::acquireAndCheckValid;
    MagAOX::app::dmRecon::recordCompletion;
    MagAOX::app::dmRecon::allocate;
    MagAOX::app::dmRecon::processImage;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    prepareTiming( fixture );
    REQUIRE( app.enable( false ) == 0 );
    REQUIRE( fixture.receive() == 0 );

    SECTION( "command clock qualification failed" )
    {
        app.m_pendingCommand.clockValid = false;
        REQUIRE( app.acquireAndCheckValid() == 1 );
    }

    SECTION( "command has no qualified baseline trigger" )
    {
        app.m_pendingCommand.trigger = {};
        REQUIRE( app.acquireAndCheckValid() == 1 );
    }

    SECTION( "missing exact successor remains pending until reset" )
    {
        REQUIRE( app.acquireAndCheckValid() == 1 );
        observeNextTrigger( fixture, 2 );
        CHECK( app.m_dryTimingStats.beforeNext == 0 );
        CHECK( app.m_dryTimingStats.late == 0 );
        CHECK( app.m_dryTimingStats.unknown == 0 );
        REQUIRE( app.allocate( dmLoopTriggerShmimT{} ) == 0 );
    }

    SECTION( "trigger attachment is replaced" )
    {
        REQUIRE( app.acquireAndCheckValid() == 1 );
        REQUIRE( app.allocate( dmLoopTriggerShmimT{} ) == 0 );
    }

    SECTION( "trigger clock sample fails" )
    {
        REQUIRE( app.acquireAndCheckValid() == 1 );
        app.m_triggerClock = static_cast<clockid_t>( -12345 );
        REQUIRE( app.processImage( nullptr, dmLoopTriggerShmimT{} ) == 0 );
    }

    SECTION( "trigger counter resets" )
    {
        REQUIRE( app.acquireAndCheckValid() == 1 );
        fixture.trigger.metadata.cnt0 = 0;
        observeNextTrigger( fixture, 0 );
    }

    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 1 );
    CHECK( app.m_dryTimingStats.marginNs == 0 );
    CHECK( app.m_dryTimingStats.minMarginNs == 0 );
    CHECK( app.m_dryTimingStats.maxMarginNs == 0 );
    CHECK( app.m_beforeNext.load() == 0 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
}

/// A dry clock failure retires only dry observations, preserving a live publication awaiting its successor.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon dry clock failure preserves pending live timing", "[dmRecon][app][dryTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::acquireAndCheckValid;
    MagAOX::app::dmRecon::frameGrabberPostPublish;
    MagAOX::app::dmRecon::recordCompletion;
    MagAOX::app::dmRecon::processImage;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    prepareTiming( fixture );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.loadImageIntoStream( fixture.destination.pixels.data() ) == 0 );
    REQUIRE( app.frameGrabberPostPublish( &fixture.destination.image ) == 0 );
    advanceTimedCommand( fixture );
    REQUIRE( app.enable( false ) == 0 );
    REQUIRE( fixture.receive() == 0 );
    app.m_pendingCommand.clockValid = false;
    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.m_dryTimingStats.unknown == 1 );
    observeNextTrigger( fixture );
    CHECK( app.m_beforeNext.load() == 1 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 1 );
}

/// Reject state superseded between reconstruction and dry completion without counting a timing observation.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon dry completion rejects superseded reconstruction state", "[dmRecon][app][dryTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::recordCompletion;
    MagAOX::app::dmRecon::processImage;
    MagAOX::app::dmRecon::allocate;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    prepareTiming( fixture );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );

    SECTION( "calibration generation changed" )
    {
        ++app.m_calibrationGeneration;
    }

    SECTION( "command attachment changed" )
    {
        ++app.m_inputGeneration;
    }

    SECTION( "mode cube invalidated" )
    {
        app.m_modesValid = false;
    }

    SECTION( "mask invalidated" )
    {
        app.m_maskValid = false;
    }

    SECTION( "calibration unusable" )
    {
        app.m_calibrationUsable = false;
    }

    SECTION( "feedback output replaced" )
    {
        app.m_outputChanged = true;
    }

    app.recordCompletion( true );
    CHECK( app.m_rejected.load() == 1 );
    observeNextTrigger( fixture );
    REQUIRE( app.allocate( dmLoopTriggerShmimT{} ) == 0 );
    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 0 );
    CHECK( app.m_beforeNext.load() == 0 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
}

/// Accumulate signed margins only from resolved classifications and preserve extrema across unknown updates.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon dry timing retains cumulative signed margin extrema", "[dmRecon][app][dryTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::dryTimingUpdate;
    #endif
    // clang-format on

    const int64_t first = GENERATE( int64_t( 30 ), int64_t( 0 ), int64_t( -30 ) );
    dmRecon_test  app;
    app.applyDryUpdate( { 0, 0, 3, 999 } );
    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 3 );
    CHECK( app.m_dryTimingStats.marginNs == 0 );
    CHECK( app.m_dryTimingStats.minMarginNs == 0 );
    CHECK( app.m_dryTimingStats.maxMarginNs == 0 );
    app.applyDryUpdate( { unsigned( first > 0 ), unsigned( first <= 0 ), 0, first } );
    CHECK( app.m_dryTimingStats.beforeNext + app.m_dryTimingStats.late == 1 );
    CHECK( app.m_dryTimingStats.marginNs == first );
    CHECK( app.m_dryTimingStats.minMarginNs == first );
    CHECK( app.m_dryTimingStats.maxMarginNs == first );
    app.applyDryUpdate( { 0, 0, 1, -999 } );
    CHECK( app.m_dryTimingStats.marginNs == first );
    CHECK( app.m_dryTimingStats.minMarginNs == first );
    CHECK( app.m_dryTimingStats.maxMarginNs == first );
    app.applyDryUpdate( { 0, 1, 0, -40 } );
    app.applyDryUpdate( { 0, 1, 0, 0 } );
    app.applyDryUpdate( { 1, 0, 0, 60 } );
    CHECK( app.m_dryTimingStats.beforeNext == 1 + uint64_t( first > 0 ) );
    CHECK( app.m_dryTimingStats.late == 2 + uint64_t( first <= 0 ) );
    CHECK( app.m_dryTimingStats.unknown == 4 );
    CHECK( app.m_dryTimingStats.marginNs == 60 );
    CHECK( app.m_dryTimingStats.minMarginNs == -40 );
    CHECK( app.m_dryTimingStats.maxMarginNs == 60 );
    CHECK( app.m_beforeNext.load() == 0 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
}

/// Collect valid paired monitor samples once, excluding the endpoint transition and the base rejected finish.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon monitor timing supplies compatible paired statistics", "[dmRecon][app][fgTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::recordMonitorTiming;
    MagAOX::app::dmRecon::selectFGTimingSource;
    MagAOX::app::dev::frameGrabberTiming::published;
    MagAOX::app::dev::frameGrabberTiming::finish;
    #endif
    // clang-format on

    using namespace MagAOX::fgTiming;
    dmRecon_test app;
    app.m_fgTiming.configure( 5, 32, false );

    for( uint64_t counter = 1; counter <= 4; ++counter )
    {
        const auto acquired           = shiftedTime( { 100, 0 }, int64_t( counter ) * 1000000 );
        app.m_currentCommand.received = acquired;
        app.m_fgTiming.begin();
        app.recordMonitorTiming( shiftedTime( acquired, 200000 ), counter );
        CHECK_FALSE( app.m_fgTiming.active() );
        // The normal frame-grabber tail still sees a disabled acquisition and must not append a second event.
        app.m_fgTiming.mark( Marker::validated );
        app.m_fgTiming.finish( Outcome::rejected );
    }

    app.m_fgTiming.update( 10000000000, 101000000000 );
    const auto &summary   = app.m_fgTiming.summary();
    const auto &acquired  = summary.metrics[metricIndex( Metric::acquisitionInterval )];
    const auto &published = summary.metrics[metricIndex( Metric::publicationInterval )];
    const auto &delay     = summary.metrics[metricIndex( Metric::acquisitionToWrite )];
    CHECK( app.m_fgTimingSource == dmRecon_test::FGTimingSource::monitor );
    CHECK( summary.mode == uint32_t( dmRecon_test::FGTimingSource::monitor ) );
    CHECK( summary.total.events == 4 );
    CHECK( summary.total.published == 3 );
    CHECK( summary.total.rejected == 1 );
    REQUIRE( acquired.valid );
    REQUIRE( published.valid );
    REQUIRE( delay.valid );
    CHECK( acquired.count == 2 );
    CHECK( acquired.mean == Approx( 0.001 ) );
    CHECK( published.count == 2 );
    CHECK( published.mean == Approx( 0.001 ) );
    CHECK( delay.count == 3 );
    CHECK( delay.mean == Approx( 0.0002 ) );
    CHECK( delay.stddev == Approx( 0 ) );
    CHECK( app.m_beforeNext.load() == 0 );
    CHECK( app.m_late.load() == 0 );
    CHECK( app.m_unknown.load() == 0 );
    CHECK( app.m_dryTimingStats.beforeNext == 0 );
    CHECK( app.m_dryTimingStats.late == 0 );
    CHECK( app.m_dryTimingStats.unknown == 0 );
}

/// Preserve unavailable legacy timing when no monitor was posted or its completion timestamp is unusable.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon monitor timing excludes absent and invalid completions", "[dmRecon][app][fgTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::publishDiagnostics;
    MagAOX::app::dmRecon::recordMonitorTiming;
    #endif
    // clang-format on

    using namespace MagAOX::fgTiming;
    dmRecon_test app;
    app.m_fgTiming.configure( 5, 32, false );
    app.m_currentCommand.received = { 100, 0 };
    app.m_fgTiming.begin();

    SECTION( "monitor is unavailable" )
    {
        app.publishDiagnostics( true );
    }

    SECTION( "completion clock failed" )
    {
        app.recordMonitorTiming( {}, 1 );
    }

    SECTION( "completion nanoseconds are negative" )
    {
        app.recordMonitorTiming( { 100, -1 }, 1 );
    }

    SECTION( "completion nanoseconds are unnormalized" )
    {
        app.recordMonitorTiming( { 100, 1000000000 }, 1 );
    }

    CHECK( app.m_fgTiming.active() );
    app.m_fgTiming.finish( Outcome::rejected );
    app.m_fgTiming.update( 10000000000, 101000000000 );
    const auto &summary = app.m_fgTiming.summary();
    CHECK( app.m_fgTimingSource == dmRecon_test::FGTimingSource::unavailable );
    CHECK( summary.mode == uint32_t( dmRecon_test::FGTimingSource::unavailable ) );
    CHECK( summary.total.published == 0 );
    CHECK( summary.total.rejected == 1 );
    CHECK( summary.metrics[metricIndex( Metric::acquisitionToWrite )].count == 0 );
    CHECK( summary.metrics[metricIndex( Metric::acquisitionInterval )].count == 0 );
    CHECK( summary.metrics[metricIndex( Metric::publicationInterval )].count == 0 );
}

/// Keep each endpoint's window separate while preserving the normal live feedback publication path.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing sessions follow actual output endpoints", "[dmRecon][app][fgTiming]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::recordMonitorTiming;
    MagAOX::app::dmRecon::selectFGTimingSource;
    MagAOX::app::dmRecon::loadImageIntoStream;
    MagAOX::app::dmRecon::frameGrabberPostPublish;
    MagAOX::app::dmRecon::reconfig;
    #endif
    // clang-format on

    using namespace MagAOX::fgTiming;
    const bool firstMonitor = GENERATE( true, false );
    Fixture    fixture;
    auto      &app = fixture.app;
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );
    REQUIRE( app.reconstruct() == 0 );
    app.m_fgTiming.configure( 5, 32, false );
    uint64_t counter = 0;

    for( bool monitor : { firstMonitor, !firstMonitor } )
    {
        const int64_t latency = monitor ? 800000 : 300000;
        const auto    source = monitor ? dmRecon_test::FGTimingSource::monitor : dmRecon_test::FGTimingSource::feedback;

        for( int sample = 0; sample < 4; ++sample )
        {
            ++counter;
            const auto acquired           = shiftedTime( { 100, 0 }, int64_t( counter ) * 1000000 );
            const auto completed          = shiftedTime( acquired, latency );
            app.m_currentCommand.received = acquired;
            app.m_fgTiming.begin();

            if( monitor )
            {
                app.recordMonitorTiming( completed, counter );
                app.m_fgTiming.finish( Outcome::rejected );
            }
            else
            {
                REQUIRE( app.loadImageIntoStream( fixture.destination.pixels.data() ) == 0 );
                // Simulate the base's successful local feedback metadata/post sequence without stream operations.
                app.m_fgTiming.published( acquired, completed, counter, fixture.destination.metadata.inode );
                REQUIRE( app.frameGrabberPostPublish( &fixture.destination.image ) == 0 );
                app.m_fgTiming.finish();
                CHECK_FALSE( app.m_publicationLock.owns_lock() );
                CHECK( fixture.destination.pixels == std::vector<float>{ 3, 5 } );
            }
        }

        app.m_fgTiming.update( 10000000000, 101000000000 );
        const auto &summary = app.m_fgTiming.summary();
        const auto &delay   = summary.metrics[metricIndex( Metric::acquisitionToWrite )];
        CHECK( app.m_fgTimingSource == source );
        CHECK( summary.mode == uint32_t( source ) );
        REQUIRE( delay.valid );
        CHECK( delay.count == 3 );
        CHECK( delay.mean == Approx( double( latency ) * 1e-9 ) );
        CHECK( summary.metrics[metricIndex( Metric::acquisitionInterval )].count == 2 );
        CHECK( summary.metrics[metricIndex( Metric::publicationInterval )].count == 2 );
    }

    REQUIRE( app.reconfig() == 0 );
    app.m_fgTiming.update( 10000000000, 101000000000 );
    CHECK( app.m_fgTimingSource == dmRecon_test::FGTimingSource::unavailable );
    CHECK( app.m_fgTiming.summary().mode == uint32_t( dmRecon_test::FGTimingSource::unavailable ) );
    CHECK( app.m_fgTiming.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count == 0 );
}

/// Refuse mismatched, nonfinite and obsolete commands before publishing any reconstructed value.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon CPU reconstruction rejects invalid inputs", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::reconstruct;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.takeCommand() );

    SECTION( "shape mismatch" )
    {
        ++app.m_currentCommand.width;
    }

    SECTION( "nonfinite sample outside the selected actuator mask" )
    {
        app.m_currentCommand.pixels[1] = std::bit_cast<float>( 0x7fc00000U );
    }

    SECTION( "obsolete calibration" )
    {
        ++app.m_currentCommand.generation;
    }

    SECTION( "unusable calibration" )
    {
        app.m_calibrationUsable = false;
    }

    SECTION( "finite input overflows during reconstruction" )
    {
        app.m_PInv( 0, 0 ) = std::numeric_limits<float>::max();
    }

    SECTION( "short private input" )
    {
        app.m_currentCommand.pixels.pop_back();
    }

    SECTION( "oversized private input" )
    {
        app.m_currentCommand.pixels.push_back( 0 );
    }

    CHECK( app.reconstruct() == 1 );
    CHECK( app.m_rejected.load() == 1 );
}

/// Treat semaphore counts as wake hints rather than permission to replay the previous reconstruction.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon acquisition never replays a consumed command", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::acquireAndCheckValid;
    #endif
    // clang-format on

    Fixture fixture;
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( fixture.app.acquireAndCheckValid() == 0 );
    REQUIRE( sem_post( &fixture.app.m_smSemaphore ) == 0 );
    CHECK( fixture.app.acquireAndCheckValid() == 1 );
    fixture.command.metadata.cnt0++;
    REQUIRE( fixture.app.enable( false ) == 0 );
    REQUIRE( fixture.receive() == 0 );
    CHECK( fixture.app.acquireAndCheckValid() == 1 );
    CHECK_FALSE( fixture.app.m_commandPending );
}

/// Recheck operator state and both generations at the final copy, leaving rejected outputs untouched.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon final publication rejects disabled or replaced state", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::loadImageIntoStream;
    MagAOX::app::dmRecon::frameGrabberPostPublish;
    MagAOX::app::dmRecon::newCallBack_m_indiP_writeDMf;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    std::fill( fixture.destination.pixels.begin(), fixture.destination.pixels.end(), -99 );
    fixture.destination.metadata.write = 1;

    SECTION( "operator disabled after acquisition" )
    {
        REQUIRE( app.enable( false ) == 0 );
    }

    SECTION( "calibration replaced after acquisition" )
    {
        REQUIRE( app.allocate( dmModesShmimT{} ) == 0 );
    }

    SECTION( "command attachment replaced after acquisition" )
    {
        REQUIRE( app.allocate( dmCommandShmimT{} ) == 0 );
    }

    SECTION( "feedback destination replaced after acquisition" )
    {
        app.m_outputChanged = true;
    }

    CHECK( app.loadImageIntoStream( fixture.destination.pixels.data() ) == -1 );
    CHECK( fixture.destination.pixels == std::vector<float>{ -99, -99 } );
    CHECK( fixture.destination.metadata.write == 0 );
    CHECK_FALSE( app.m_publicationLock.owns_lock() );
}

/// Retain output ownership until publication completes so a returning disable cannot be followed by an old copy.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon serializes disable with completion of the final publication", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
    MagAOX::app::dmRecon::loadImageIntoStream;
    MagAOX::app::dmRecon::frameGrabberPostPublish;
    MagAOX::app::dmRecon::newCallBack_m_indiP_writeDMf;
    #endif
    // clang-format on

    Fixture fixture;
    auto   &app = fixture.app;
    app.staged( GENERATE( true, false ) );
    REQUIRE( fixture.receive() == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.loadImageIntoStream( fixture.destination.pixels.data() ) == 0 );
    CHECK( fixture.destination.pixels[0] == Approx( 3 ) );
    CHECK( fixture.destination.pixels[1] == Approx( 5 ) );
    CHECK( app.m_publicationLock.owns_lock() );
    std::promise<void> started;
    auto               start    = started.get_future();
    auto               disabled = std::async( std::launch::async,
                                [&app, &started]
                                {
                                    started.set_value();
                                    return app.enable( false );
                                } );
    start.wait();
    const auto beforeCompletion = disabled.wait_for( std::chrono::milliseconds( 20 ) );
    const int  completed        = app.frameGrabberPostPublish( &fixture.destination.image );
    const int  disableResult    = disabled.get();
    CHECK( beforeCompletion == std::future_status::timeout );
    CHECK( completed == 0 );
    CHECK( disableResult == 0 );
    CHECK_FALSE( app.m_publicationLock.owns_lock() );
    CHECK_FALSE( app.m_writeDMf.load() );
    CHECK( app.loadImageIntoStream( fixture.destination.pixels.data() ) == -1 );
}

} // namespace dmReconTest
} // namespace libXWCTest
