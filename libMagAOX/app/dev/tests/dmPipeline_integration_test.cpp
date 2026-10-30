/** \file dmPipeline_integration_test.cpp
 * \brief Offline ownership, provenance and output tests of detached DM pipeline stages.
 * \ingroup dmPipeline_integration_test
 */
#include <filesystem>

#include "../../../../tests/testXWC.hpp"
#include "../../MagAOXApp.hpp"
#include "../shmimMonitor.hpp"

/// \cond DM_PIPELINE_INTEGRATION_HELPERS
namespace dmPipelineIntegration
{
bool m_failPost{ false }; ///< Fail only the test-owned saturation notification.

/// Post a local semaphore or inject a deterministic final publication failure.
int semPost( sem_t *semaphore /**< [in] Workstation-local saturation semaphore. */ );

int semPost( sem_t *semaphore )
{
    if( m_failPost )
    {
        errno = EIO;
        return -1;
    }

    return ::sem_post( semaphore );
}
} // namespace dmPipelineIntegration
/// \endcond

#define sem_post dmPipelineIntegration::semPost
#include "../dm.hpp"
#undef sem_post

namespace libXWCTest
{
namespace dmPipelineIntegrationTest
{
using MagAOX::dmCompletion::qualified;
using MagAOX::dmCompletion::readersQualified;
using MagAOX::dmCompletion::State;
using MagAOX::dmTiming::Marker;
using MagAOX::dmTiming::Outcome;
using Clock = MagAOX::app::dev::dmTiming;

/// \cond DM_PIPELINE_INTEGRATION_HELPERS
/// Isolate ImageStreamIO's process-cached shared-memory directory from instrument streams.
class Directory
{
    std::string m_path; ///< Workstation-only temporary image directory.

    std::string m_previous; ///< Original directory restored at process exit.

    bool m_hadPrevious{ false }; ///< Distinguish an originally absent environment setting.

  public:
    /// Select a process-private directory before any test image is opened.
    Directory();

    /// Restore the environment and remove only this test's local images.
    ~Directory();
};

Directory::Directory()
{
    char        pattern[] = "/tmp/dm-pipeline-integration-XXXXXX";
    const auto *path      = mkdtemp( pattern );
    REQUIRE( path != nullptr );
    m_path               = path;
    const auto *previous = getenv( "MILK_SHM_DIR" );

    if( previous != nullptr )
    {
        m_hadPrevious = true;
        m_previous    = previous;
    }

    REQUIRE( setenv( "MILK_SHM_DIR", m_path.c_str(), 1 ) == 0 );
}

Directory::~Directory()
{
    if( m_hadPrevious )
    {
        setenv( "MILK_SHM_DIR", m_previous.c_str(), 1 );
    }
    else
    {
        unsetenv( "MILK_SHM_DIR" );
    }

    std::filesystem::remove_all( m_path );
}

/// Retain the directory through destruction of all image mappings.
Directory &directory();

Directory &directory()
{
    static Directory value;
    return value;
}

/// Exercise generic pipeline helpers without an app thread, INDI connection or hardware SDK.
struct Controller : MagAOX::app::MagAOXApp<false>,
                    MagAOX::app::dev::dm<Controller, float>,
                    MagAOX::app::dev::shmimMonitor<Controller>
{
    friend class MagAOX::app::dev::dm<Controller, float>;
    friend class MagAOX::app::dev::shmimMonitor<Controller>;
    using dmT = MagAOX::app::dev::dm<Controller, float>;
    using dmT::m_dmCompletion;
    using dmT::m_dmReaderRelease;
    using dmT::m_dmTiming;
    using dmT::m_instSatMap;
    using dmT::m_outputShape;
    using dmT::m_pipelineSaturation;
    using dmT::m_satSemaphore;

    IMAGE_METADATA m_metadata{}; ///< Combined metadata with no concurrent external writer.

    std::array<float, 2> m_pixels{ 1, 2 }; ///< Caller-owned simulated combined command pixels.

    std::vector<std::string> m_names; ///< Local image names removed after attachment closure.

    /// Create local geometry, channels and independent completion/release publishers.
    Controller();

    /// Close local mappings and remove the fixture's images after all jobs finish.
    ~Controller() noexcept;

    /// Satisfy the app interface without starting a device or worker.
    int appStartup() override;

    /// Satisfy the app interface without starting an INDI server.
    int appLogic() override;

    /// Satisfy the app interface without contacting hardware.
    int appShutdown() override;

    /// Advance the coherent combined input identity once.
    void next();

    /// Update the fixture's two uniform input-channel shapes.
    void channels( float flat /**< [in] Non-delta channel value. */, float delta /**< [in] Delta channel value. */ );

    /// Capture, prepare and release a genuine job using the real generic helpers.
    void prepare( PipelineCommand &command /**< [in,out] Preallocated job. */,
                  float            shape /**< [in] Private clipped shape for this command. */ );

    /// Report a successful software SDK endpoint through the real publication path.
    int complete( PipelineCommand &command /**< [in,out] Released job. */ );

    /// Read the published delta and difference at the first actuator.
    std::pair<float, float> products();
};

Controller::Controller() : MagAOXApp( "test", false )
{
    static_cast<void>( directory() );
    m_configName         = "dm-pipeline-integration";
    m_calibDir           = "/tmp/dm-pipeline-integration";
    m_flatPath           = "/tmp";
    m_testPath           = "/tmp";
    m_metadata.inode     = 1234;
    m_metadata.cnt0      = 50;
    m_metadata.writetime = { 1700000000, 500000 };
    m_imageStream        = {};
    m_imageStream.md     = &m_metadata;
    m_dmWidth            = 2;
    m_dmHeight           = 1;
    m_totalFlat.resize( 2, 1 );
    m_totalDelta.resize( 2, 1 );
    m_instSatMap.resize( 2, 1 );
    m_instSatMap.setZero();
    m_dmTiming.configure( 2, 20 );
    REQUIRE( sem_init( &m_satSemaphore, 0, 0 ) == 0 );
    m_names = { "pipeline_flat",
                "pipeline_delta",
                "pipeline_shape",
                "pipeline_out_delta",
                "pipeline_out_diff",
                "pipeline_done",
                "pipeline_read_done" };
    m_channels.push_back( new mx::improc::milkImage<float>( m_names[0], 2, 1 ) );
    m_channels.push_back( new mx::improc::milkImage<float>( m_names[1], 2, 1 ) );
    m_notDeltas     = { 0 };
    m_deltas        = { 1 };
    m_deltaChannels = { m_names[1] };
    m_outputShape.create( m_names[2], 2, 1 );
    m_outputDelta.create( m_names[3], 2, 1 );
    m_outputDiff.create( m_names[4], 2, 1 );
    m_outputShape().setZero();
    m_outputDelta().setZero();
    m_outputDiff().setZero();
    channels( 4, 9 );
    REQUIRE( m_dmCompletion.open( m_names[5] ) == 0 );
    REQUIRE( m_dmReaderRelease.open( m_names[6], MagAOX::dmCompletion::Kind::readers ) == 0 );
}

Controller::~Controller() noexcept
{
    dmPipelineIntegration::m_failPost = false;
    m_dmCompletion.close();
    m_dmReaderRelease.close();
    m_outputShape.close();
    m_outputDelta.close();
    m_outputDiff.close();

    for( auto *channel : m_channels )
    {
        delete channel;
    }

    m_channels.clear();
    m_imageStream.md = nullptr;
    sem_destroy( &m_satSemaphore );

    for( const auto &name : m_names )
    {
        IMAGE image{};

        if( ImageStreamIO_openIm( &image, name.c_str() ) == IMAGESTREAMIO_SUCCESS )
        {
            ImageStreamIO_destroyIm( &image );
        }
    }
}

int Controller::appStartup()
{
    return 0;
}

int Controller::appLogic()
{
    return 0;
}

int Controller::appShutdown()
{
    return 0;
}

void Controller::next()
{
    ++m_metadata.cnt0;
    ++m_metadata.writetime.tv_nsec;
}

void Controller::channels( float flat, float delta )
{
    ( *m_channels[0] )().setConstant( flat );
    ( *m_channels[1] )().setConstant( delta );
}

void Controller::prepare( PipelineCommand &command, float shape )
{
    pipelineShmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( capturePipelineCommand( command, m_pixels.data() ) == 0 );
    command.shape.setConstant( shape );
    pipelineMark( command, Marker::actuatorEnd );
    command.dispatchNs = Clock::monotonicNowNs();
    REQUIRE( stagePipelineCommand( command ) == 0 );
    releasePipelineCommand( command );
    REQUIRE( command.preparedStatus == 0 );
    REQUIRE( readersQualified( m_dmReaderRelease.snapshot() ) );
}

int Controller::complete( PipelineCommand &command )
{
    const auto start = Clock::monotonicNowNs();
    const auto end   = Clock::monotonicNowNs();
    return finishPipelineCommand( command, Outcome::success, 0, start, end, true );
}

std::pair<float, float> Controller::products()
{
    return { m_outputDelta()( 0, 0 ), m_outputDiff()( 0, 0 ) };
}
/// \endcond

/// Publish A's private products after B has replaced every live input used by A.
/** \ingroup dmPipeline_integration_test */
TEST_CASE( "DM pipeline preserves private products and frozen identity across newer commands", "[dev::dmPipeline]" )
{
    // clang-format off
#ifdef DM_PIPELINE_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::allocatePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::capturePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::stagePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::releasePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::finishPipelineCommand;
#endif
    // clang-format on
    Controller                  app;
    Controller::PipelineCommand first;
    Controller::PipelineCommand second;
    app.allocatePipelineCommand( first );
    app.allocatePipelineCommand( second );
    app.prepare( first, 20 );
    const auto identity = first.completion;
    CHECK( first.delta( 0, 0 ) == 16 );
    CHECK( first.diff( 0, 0 ) == -7 );
    app.next();
    app.channels( 6, 11 );
    app.m_outputShape().setConstant( 123 );
    app.prepare( second, 70 );
    CHECK( second.delta( 0, 0 ) == 64 );
    const auto release = app.m_dmReaderRelease.snapshot();
    REQUIRE( release.inputCounter != identity.inputCounter );
    REQUIRE( app.complete( first ) == 0 );
    CHECK( app.products() == std::make_pair( 16.0F, -7.0F ) );
    const auto full = app.m_dmCompletion.snapshot();
    CHECK( qualified( full ) );
    CHECK( full.inputCounter == identity.inputCounter );
    CHECK( full.publicationNsec == identity.publicationNsec );
    CHECK( full.inputCounter != app.m_metadata.cnt0 );
    CHECK( MagAOX::dmCompletion::encode( app.m_dmReaderRelease.snapshot() ) ==
           MagAOX::dmCompletion::encode( release ) );
    REQUIRE( app.complete( second ) == 0 );
    CHECK( app.products() == std::make_pair( 64.0F, -53.0F ) );
}

/// Preserve actual submission even when private preparation or later publication fails.
/** \ingroup dmPipeline_integration_test */
TEST_CASE( "DM pipeline callback errors retain successful SDK submission evidence", "[dev::dmPipeline]" )
{
    // clang-format off
#ifdef DM_PIPELINE_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::stagePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::releasePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::finishPipelineCommand;
#endif
    // clang-format on
    for( int failure = 0; failure < 3; ++failure )
    {
        Controller                  app;
        Controller::PipelineCommand command;
        app.allocatePipelineCommand( command );

        if( failure == 0 )
        {
            app.pipelineShmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
            REQUIRE( app.capturePipelineCommand( command, app.m_pixels.data() ) == 0 );
            command.shape.setConstant( 20 );
            app.pipelineMark( command, Marker::actuatorEnd );
            command.dispatchNs = Clock::monotonicNowNs();
            command.delta.resize( 0, 0 );
            REQUIRE( app.stagePipelineCommand( command ) < 0 );
            app.releasePipelineCommand( command );
            CHECK_FALSE( readersQualified( app.m_dmReaderRelease.snapshot() ) );
        }
        else
        {
            app.prepare( command, 20 );

            if( failure == 1 )
            {
                command.saturation.resize( 1 );
            }
            else
            {
                dmPipelineIntegration::m_failPost = true;
            }
        }

        REQUIRE( app.complete( command ) < 0 );
        dmPipelineIntegration::m_failPost = false;
        CHECK( command.timing.submitted );
        CHECK( command.timing.outcome == Outcome::failed );
        const auto sdkBits = ( 1u << MagAOX::dmTiming::markerIndex( Marker::sdkStart ) ) |
                             ( 1u << MagAOX::dmTiming::markerIndex( Marker::sdkEnd ) );
        CHECK( ( command.timing.valid & sdkBits ) == sdkBits );
        CHECK( app.m_dmCompletion.snapshot().state == State::error );
        CHECK( readersQualified( app.m_dmReaderRelease.snapshot() ) == ( failure != 0 ) );
        app.m_dmTiming.update();
        CHECK( app.m_dmTiming.summary().total.submitted == 1 );
        CHECK( app.m_dmTiming.summary().total.failed == 1 );
    }
}

/// Do not manufacture SDK endpoints or successful hardware submission for cancelled work.
/** \ingroup dmPipeline_integration_test */
TEST_CASE( "DM pipeline no-send jobs keep SDK populations empty", "[dev::dmPipeline]" )
{
    // clang-format off
#ifdef DM_PIPELINE_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::finishPipelineCommand;
#endif
    // clang-format on
    Controller                  app;
    Controller::PipelineCommand command;
    app.allocatePipelineCommand( command );
    app.prepare( command, 20 );
    REQUIRE( app.finishPipelineCommand(
                 command, Outcome::noSend, MagAOX::dmCompletion::unavailableResult, -1, -1, false ) < 0 );
    CHECK_FALSE( command.timing.submitted );
    CHECK( command.timing.outcome == Outcome::noSend );
    const auto sdkBits = ( 1u << MagAOX::dmTiming::markerIndex( Marker::sdkStart ) ) |
                         ( 1u << MagAOX::dmTiming::markerIndex( Marker::sdkEnd ) );
    CHECK( ( command.timing.valid & sdkBits ) == 0 );
    CHECK( app.products() == std::make_pair( 0.0F, 0.0F ) );
    int posts = -1;
    REQUIRE( sem_getvalue( &app.m_satSemaphore, &posts ) == 0 );
    CHECK( posts == 0 );
    app.m_dmTiming.update();
    CHECK( app.m_dmTiming.summary().total.noSend == 1 );
    CHECK( app.m_dmTiming.summary().total.submitted == 0 );
    CHECK( app.m_dmTiming.summary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdk )].count ==
           0 );
}

/// Finalize A without mutating the monitor's receipt event for B.
/** \ingroup dmPipeline_integration_test */
TEST_CASE( "DM pipeline finalization preserves the monitor's next active timing event", "[dev::dmPipeline]" )
{
    // clang-format off
#ifdef DM_PIPELINE_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::pipelineShmimMonitorTiming;
    MagAOX::app::dev::dm<Controller, float>::discardPipelineCommand;
#endif
    // clang-format on
    Controller                  app;
    Controller::PipelineCommand first;
    Controller::PipelineCommand second;
    app.allocatePipelineCommand( first );
    app.allocatePipelineCommand( second );
    app.prepare( first, 20 );
    app.next();
    app.pipelineShmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( app.m_dmTiming.active() );
    REQUIRE( app.complete( first ) == 0 );
    CHECK( app.m_dmTiming.active() );
    CHECK( app.m_dmTiming.outcome() == Outcome::unknown );
    REQUIRE( app.capturePipelineCommand( second, app.m_pixels.data() ) == 0 );
    CHECK( second.timing.sequence == first.timing.sequence + 1 );
    CHECK( second.timing.outcome == Outcome::unknown );
    CHECK_FALSE( second.timing.submitted );
    CHECK_FALSE( app.m_dmTiming.active() );
    app.m_dmTiming.update();
    CHECK( app.m_dmTiming.summary().total.events == 1 );
    app.discardPipelineCommand();
    CHECK_FALSE( app.m_dmTiming.active() );
}

/// Consume both successful masks once even when the saturation worker wakes only after the latest map changed.
/** \ingroup dmPipeline_integration_test */
TEST_CASE( "DM pipeline saturation payloads preserve batches rather than replaying latest mask", "[dev::dmPipeline]" )
{
    // clang-format off
#ifdef DM_PIPELINE_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::finishPipelineCommand;
    MagAOX::app::dev::dmSaturation::drain;
#endif
    // clang-format on
    Controller                  app;
    Controller::PipelineCommand first;
    Controller::PipelineCommand second;
    app.allocatePipelineCommand( first );
    app.allocatePipelineCommand( second );
    app.prepare( first, 20 );
    first.saturation     = { 1, 0 };
    first.saturatedCount = 1;
    REQUIRE( app.complete( first ) == 0 );
    app.next();
    app.prepare( second, 70 );
    second.saturation     = { 0, 1 };
    second.saturatedCount = 1;
    REQUIRE( app.complete( second ) == 0 );
    CHECK( app.m_instSatMap( 0, 0 ) == 0 );
    CHECK( app.m_instSatMap( 1, 0 ) == 1 );
    std::array<uint64_t, 2> totals{ 0, 0 };
    REQUIRE( app.m_pipelineSaturation.drain( totals.data(), totals.size() ) == 2 );
    CHECK( totals == std::array<uint64_t, 2>{ 1, 1 } );
    REQUIRE( sem_trywait( &app.m_satSemaphore ) == 0 );
    REQUIRE( sem_trywait( &app.m_satSemaphore ) == 0 );
    CHECK( app.m_pipelineSaturation.drain( totals.data(), totals.size() ) == 0 );
    CHECK( totals == std::array<uint64_t, 2>{ 1, 1 } );
}

/// Withhold stale read permission without turning newer live metadata into a failed private command.
/** \ingroup dmPipeline_integration_test */
TEST_CASE( "DM pipeline late input advancement preserves SDK success and subsequent frame timing", "[dev::dmPipeline]" )
{
    // clang-format off
#ifdef DM_PIPELINE_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::capturePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::stagePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::releasePipelineCommand;
    MagAOX::app::dev::dm<Controller, float>::finishPipelineCommand;
    MagAOX::app::dev::dmTiming::update;
    MagAOX::app::dev::dmCompletion::publishCompleted;
#endif
    // clang-format on
    for( const bool releaseEnabled : { true, false } )
    {
        Controller                  app;
        Controller::PipelineCommand first;
        app.allocatePipelineCommand( first );

        if( !releaseEnabled )
        {
            app.m_dmReaderRelease.close();
        }

        app.pipelineShmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
        REQUIRE( app.capturePipelineCommand( first, app.m_pixels.data() ) == 0 );
        first.shape.setConstant( 20 );
        app.pipelineMark( first, Marker::actuatorEnd );
        first.dispatchNs = Clock::monotonicNowNs();
        REQUIRE( app.stagePipelineCommand( first ) == 0 );
        const auto firstCounter = first.completion.inputCounter;
        REQUIRE( first.delta( 0, 0 ) == 16 );
        REQUIRE( first.diff( 0, 0 ) == -7 );

        // A's products are private before the producer publishes B; only provenance is now uncertain.
        app.next();
        app.channels( 6, 11 );
        app.releasePipelineCommand( first );
        CHECK( first.preparedStatus == 0 );
        CHECK( first.completion.state == State::unqualified );
        CHECK_FALSE( readersQualified( app.m_dmReaderRelease.snapshot() ) );

        if( releaseEnabled )
        {
            CHECK( app.m_dmReaderRelease.snapshot().state == State::unqualified );
        }

        REQUIRE( app.complete( first ) == 0 );
        CHECK( app.products() == std::make_pair( 16.0F, -7.0F ) );
        CHECK( first.timing.submitted );
        CHECK( first.timing.outcome == Outcome::success );
        CHECK( app.m_dmCompletion.snapshot().state == State::unqualified );
        CHECK( app.m_dmCompletion.snapshot().inputCounter == firstCounter );
        CHECK_FALSE( qualified( app.m_dmCompletion.snapshot() ) );

        for( int index = 0; index < 2; ++index )
        {
            Controller::PipelineCommand command;
            app.allocatePipelineCommand( command );

            if( index != 0 )
            {
                app.next();
            }

            app.pipelineShmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
            REQUIRE( app.capturePipelineCommand( command, app.m_pixels.data() ) == 0 );
            command.shape.setConstant( 70 + index );
            app.pipelineMark( command, Marker::actuatorEnd );
            command.dispatchNs = Clock::monotonicNowNs();
            REQUIRE( app.stagePipelineCommand( command ) == 0 );
            app.releasePipelineCommand( command );
            REQUIRE( command.preparedStatus == 0 );

            if( releaseEnabled )
            {
                REQUIRE( readersQualified( app.m_dmReaderRelease.snapshot() ) );
            }

            REQUIRE( app.complete( command ) == 0 );
            REQUIRE( qualified( app.m_dmCompletion.snapshot() ) );
            CHECK( app.m_dmCompletion.snapshot().inputCounter == firstCounter + index + 1 );
        }

        app.m_dmTiming.update();
        const auto &summary = app.m_dmTiming.summary();
        CHECK( summary.total.events == 3 );
        CHECK( summary.total.receipts == 3 );
        CHECK( summary.total.submitted == 3 );
        CHECK( summary.total.failed == 0 );
        CHECK( summary.total.invalid == 0 );
        CHECK( summary.total.dropped == 0 );
        CHECK( summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdk )].count == 3 );
        CHECK( summary.fgValid );
    }
}

} // namespace dmPipelineIntegrationTest
} // namespace libXWCTest
