/** \file dmRecon_test.cpp
 * \brief Offline dmRecon snapshot, reconstruction and publication ownership regressions.
 * \ingroup dmRecon_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../dmRecon.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>

/// \cond DMRECON_APP_TEST_HARNESS
namespace MagAOX
{
namespace app
{

/// Expose real app callbacks using borrowed process-local images without starting any service or worker.
class dmRecon_test : public dmRecon
{
  public:
    using dmRecon::Frame;
    using dmRecon::m_calibrationGeneration;
    using dmRecon::m_calibrationUsable;
    using dmRecon::m_coalesced;
    using dmRecon::m_commandPending;
    using dmRecon::m_currentCommand;
    using dmRecon::m_inputGeneration;
    using dmRecon::m_latestTrigger;
    using dmRecon::m_maskIDX;
    using dmRecon::m_maskValid;
    using dmRecon::m_modesValid;
    using dmRecon::m_modevals;
    using dmRecon::m_outputChanged;
    using dmRecon::m_pendingCommand;
    using dmRecon::m_pendingMask;
    using dmRecon::m_pendingModes;
    using dmRecon::m_PInv;
    using dmRecon::m_publicationLock;
    using dmRecon::m_rejected;
    using dmRecon::m_smSemaphore;
    using dmRecon::m_triggerClock;
    using dmRecon::m_triggerClockHigh;
    using dmRecon::m_triggerClockLow;
    using dmRecon::m_triggerClockValid;
    using dmRecon::m_workerGeneration;
    using dmRecon::m_writeDMf;
    using dmRecon::outputCompatible;
    using dmRecon::prepareCalibration;
    using dmRecon::reconstruct;
    using dmRecon::snapshot;
    using dmRecon::takeCommand;

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

    /// Supply borrowed output metadata for the real final publication checks.
    void output( IMAGE *source /**< [in] Borrowed fixture image. */ );

    /// Select the normal staged copy or direct final copy for local publication tests.
    void staged( bool value /**< [in] Whether the frame grabber stages its output. */ );

    /// Invoke the real operator-enable callback without registering or connecting INDI.
    int enable( bool value /**< [in] Requested enable state. */ );
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
    reconfig();
    dmModesSMT::m_imageStream    = {};
    dmMaskSMT::m_imageStream     = {};
    dmCommandSMT::m_imageStream  = {};
    frameGrabberT::m_imageStream = nullptr;
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

} // namespace app
} // namespace MagAOX

namespace dmReconAppHarness
{
using MagAOX::app::dmCommandShmimT;
using MagAOX::app::dmMaskShmimT;
using MagAOX::app::dmModesShmimT;
using MagAOX::app::dmRecon_test;

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
    CHECK( app.m_workerGeneration > generation );
}

/// Refuse inconsistent private payload sizes before allocating a cube or indexing its pixels.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon calibration rejects malformed private frame lengths", "[dmRecon][app]" )
{
    // clang-format off
    #ifdef DMRECON_TEST_DOXYGEN_REF
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
