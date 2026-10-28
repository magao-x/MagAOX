/** \file dmCompletion_integration_test.cpp
 * \brief Software-only verification of the DM controller reader-completion endpoint.
 */
#include <filesystem>
#include <functional>

#include "../../../../tests/testXWC.hpp"
#include "../../MagAOXApp.hpp"
#include "../shmimMonitor.hpp"

/// \cond DM_COMPLETION_INTEGRATION_HELPERS
namespace dmCompletionIntegration
{
bool m_failPost{ false }; ///< Inject a failed saturation-worker notification.

std::function<void()> m_atPost; ///< Inspect the actual final callback operation without a worker thread.

/// Inspect or reject a local saturation-semaphore post.
int semPost( sem_t *semaphore /**< [in] Workstation-local saturation semaphore. */ );

int semPost( sem_t *semaphore )
{
    if( m_atPost )
    {
        m_atPost();
    }

    if( m_failPost )
    {
        errno = EIO;
        return -1;
    }

    return ::sem_post( semaphore );
}
} // namespace dmCompletionIntegration
/// \endcond

#define sem_post dmCompletionIntegration::semPost
#include "../dm.hpp"
#undef sem_post

namespace libXWCTest
{
namespace dmCompletionIntegrationTest
{
using MagAOX::dmCompletion::qualified;
using MagAOX::dmCompletion::Record;
using MagAOX::dmCompletion::State;
using MagAOX::dmTiming::Outcome;

/// \cond DM_COMPLETION_INTEGRATION_HELPERS
/// Restrict every test image and lock file to its own workstation temporary directory.
class Directory
{
    std::string m_path; ///< Private local image directory removed after all mappings close.

    std::string m_previous; ///< Original shared-memory directory, restored after the test.

    bool m_hadPrevious{ false }; ///< Distinguish an unset original environment variable.

  public:
    /// Select a newly created temporary image directory.
    Directory();

    /// Restore the environment and remove files belonging to this fixture.
    ~Directory();
};

Directory::Directory()
{
    char        pattern[] = "/tmp/dm-completion-integration-XXXXXX";
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

/// Retain the first ImageStreamIO directory until every callback fixture has been destroyed.
Directory &directory();

Directory &directory()
{
    static Directory value;
    return value;
}

/// Capture property changes without opening an INDI FIFO or socket.
struct Driver
{
    std::vector<pcf::IndiProperty> m_messages; ///< Ordinary-thread property snapshots.

    /// Capture a software-only property update.
    void sendSetProperty( const pcf::IndiProperty &property /**< [in] Published property snapshot. */ );

    /// Capture a software-only property definition.
    void sendDefProperty( const pcf::IndiProperty &property /**< [in] Defined property snapshot. */ );

    /// Capture a software-only property removal.
    void sendDelProperty( const pcf::IndiProperty &property /**< [in] Removed property snapshot. */ );
};

void Driver::sendSetProperty( const pcf::IndiProperty &property )
{
    m_messages.push_back( property );
}

void Driver::sendDefProperty( const pcf::IndiProperty &property )
{
    m_messages.push_back( property );
}

void Driver::sendDelProperty( const pcf::IndiProperty &property )
{
    m_messages.push_back( property );
}

/// Exercise the real base callback with local metadata and a software SDK endpoint.
struct Controller : MagAOX::app::MagAOXApp<false>,
                    MagAOX::app::dev::dm<Controller, float>,
                    MagAOX::app::dev::shmimMonitor<Controller>
{
    friend class MagAOX::app::dev::dm<Controller, float>;
    friend class MagAOX::app::dev::shmimMonitor<Controller>;
    using dmT = MagAOX::app::dev::dm<Controller, float>;
    using dmT::m_completionStreamName;
    using dmT::m_dmCompletion;
    using dmT::m_dmTiming;
    using dmT::m_indiP_dmCompletion;
    using dmT::m_indiP_dmCompletionStream;
    using dmT::m_satSemaphore;
    using dmT::setupDMCompletionINDI;
    using dmT::updateDMCompletionINDI;

    Driver *m_indiDriver{ nullptr }; ///< Local property capture sink; no server is contacted.

    IMAGE_METADATA m_metadata{}; ///< Combined-input metadata owned by the fixture, with no producer thread.

    Outcome m_result{ Outcome::success }; ///< Explicit software backend submission result.

    bool m_reportResult{ true }; ///< Allow testing an endpoint that fails to qualify its SDK result.

    int m_return{ 0 }; ///< Callback return independent of its submission qualification.

    std::function<void()> m_inCommand; ///< Observe busy state or change fixture data at the SDK endpoint.

    std::vector<std::string> m_names; ///< Image names to destroy after all local attachments close.

    /// Initialize local semaphore and stable combined-input identity without starting app threads.
    Controller();

    /// Close and destroy only images created by this fixture.
    ~Controller() noexcept;

    /// Satisfy the app interface without initializing hardware.
    int appStartup() override;

    /// Satisfy the app interface without starting threads.
    int appLogic() override;

    /// Satisfy the app interface without touching instrument state.
    int appShutdown() override;

    /// Satisfy the DM interface without hardware initialization.
    int initDM();

    /// Satisfy the DM interface without sending a zero command.
    int zeroDM();

    /// Satisfy the DM interface without releasing hardware.
    int releaseDM();

    /// Observe the real callback's SDK endpoint and return a controlled result.
    int commandDM( void *pixels /**< [in] Unused local command pointer. */ );

    /// Enable completion publication on a fixture-owned image.
    void enable();

    /// Issue a genuine monitor receipt followed by the real base callback.
    int receive();

    /// Connect one-pixel channel and output images used by the actual makeDelta implementation.
    void prepareDelta();

    /// Set the non-delta and delta channel pixels at the software SDK endpoint.
    void setChannels( float flat /**< [in] New non-delta command. */, float delta /**< [in] New delta command. */ );

    /// Read the actual delta and difference products after the last input-channel reads.
    std::pair<float, float> products();
};

Controller::Controller() : MagAOXApp( "test", false )
{
    static_cast<void>( directory() );
    m_configName         = "dm-completion-integration";
    m_calibDir           = "/tmp/dm-completion-integration";
    m_flatPath           = "/tmp";
    m_testPath           = "/tmp";
    m_metadata.inode     = 1234;
    m_metadata.cnt0      = 50;
    m_metadata.writetime = { 1700000000, 500000 };
    m_imageStream        = {};
    m_imageStream.md     = &m_metadata;
    m_dmTiming.configure( 0, 20 );
    REQUIRE( sem_init( &m_satSemaphore, 0, 0 ) == 0 );
}

Controller::~Controller() noexcept
{
    dmCompletionIntegration::m_atPost   = {};
    dmCompletionIntegration::m_failPost = false;
    m_dmCompletion.close();
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

int Controller::initDM()
{
    return 0;
}

int Controller::zeroDM()
{
    return 0;
}

int Controller::releaseDM()
{
    return 0;
}

int Controller::commandDM( void *pixels )
{
    static_cast<void>( pixels );

    if( m_inCommand )
    {
        m_inCommand();
    }

    if( m_reportResult )
    {
        dmTimingOutcome( m_result );
    }

    return m_return;
}

void Controller::enable()
{
    m_completionStreamName = "completion";
    REQUIRE( m_dmCompletion.open( m_completionStreamName ) == 0 );
    m_names.push_back( m_completionStreamName );
}

int Controller::receive()
{
    shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    return processImage( nullptr, {} );
}

void Controller::prepareDelta()
{
    m_names.insert( m_names.end(), { "flat", "delta", "shape", "out_delta", "out_diff" } );
    m_channels.push_back( new mx::improc::milkImage<float>( "flat", 1, 1 ) );
    m_channels.push_back( new mx::improc::milkImage<float>( "delta", 1, 1 ) );
    m_notDeltas     = { 0 };
    m_deltas        = { 1 };
    m_deltaChannels = { "delta" };
    m_outputShape.create( "shape", 1, 1 );
    m_outputDelta.create( "out_delta", 1, 1 );
    m_outputDiff.create( "out_diff", 1, 1 );
    m_outputShape()( 0, 0 ) = 20;
    m_outputDelta()( 0, 0 ) = 0;
    m_outputDiff()( 0, 0 )  = 0;
    setChannels( 4, 9 );
}

void Controller::setChannels( float flat, float delta )
{
    ( *m_channels[0] )()( 0, 0 ) = flat;
    ( *m_channels[1] )()( 0, 0 ) = delta;
}

std::pair<float, float> Controller::products()
{
    return { m_outputDelta()( 0, 0 ), m_outputDiff()( 0, 0 ) };
}
/// \endcond

/** \defgroup dmCompletionIntegration_tests DM Completion Integration Tests
 * \ingroup app_dev_unit_tests
 */

/// Publish qualified completion only after the actual last channel reads and saturation notification.
/** \ingroup dmCompletionIntegration_tests */
TEST_CASE( "DM completion follows all callback channel reads", "[dev::dmCompletionIntegration]" )
{
    // clang-format off
#ifdef DM_COMPLETION_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
    MagAOX::app::dev::dm<Controller, float>::makeDelta();
    MagAOX::app::dev::dm<Controller, float>::dmTimingOutcome( Outcome::success );
#endif
    // clang-format on
    Controller app;
    app.enable();
    app.prepareDelta();

    SECTION( "Completion with timing telemetry disabled" )
    {
        CHECK_FALSE( app.m_dmTiming.active() );
    }

    SECTION( "Completion with timing telemetry enabled" )
    {
        app.m_dmTiming.configure( 2, 20 );
    }

    bool inspectedCommand = false;
    bool inspectedPost    = false;
    app.m_inCommand       = [&]()
    {
        const auto record = app.m_dmCompletion.snapshot();
        CHECK( record.state == State::busy );
        CHECK_FALSE( qualified( record ) );
        CHECK( record.finishedNs == 0 );
        CHECK( app.products() == std::make_pair( 0.0F, 0.0F ) );
        app.setChannels( 6, 11 );
        inspectedCommand = true;
    };
    dmCompletionIntegration::m_atPost = [&]()
    {
        CHECK( app.products() == std::make_pair( 14.0F, -3.0F ) );
        CHECK( app.m_dmCompletion.snapshot().state == State::busy );
        inspectedPost = true;
    };

    REQUIRE( app.receive() == 0 );
    REQUIRE( inspectedCommand );
    REQUIRE( inspectedPost );
    const auto record = app.m_dmCompletion.snapshot();
    CHECK( qualified( record ) );
    CHECK( record.inputInode == 1234 );
    CHECK( record.inputCounter == 50 );
    CHECK( record.publicationSec == 1700000000 );
    CHECK( record.publicationNsec == 500000 );
    CHECK( record.finishedNs >= record.startedNs );
    CHECK_FALSE( app.m_dmTiming.active() );
    int posts = 0;
    REQUIRE( sem_getvalue( &app.m_satSemaphore, &posts ) == 0 );
    CHECK( posts == 1 );
}

/// Require a fresh monitor receipt even when a synthetic callback has a valid SDK result.
/** \ingroup dmCompletionIntegration_tests */
TEST_CASE( "DM completion requires genuine monitor provenance", "[dev::dmCompletionIntegration]" )
{
    // clang-format off
#ifdef DM_COMPLETION_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
#endif
    // clang-format on
    Controller app;
    app.enable();
    REQUIRE( app.processImage( nullptr, {} ) == 0 );
    CHECK( app.m_dmCompletion.snapshot().state == State::unqualified );
    REQUIRE( app.receive() == 0 );
    CHECK( qualified( app.m_dmCompletion.snapshot() ) );
    ++app.m_metadata.cnt0;
    REQUIRE( app.processImage( nullptr, {} ) == 0 );
    CHECK( app.m_dmCompletion.snapshot().state == State::unqualified );

    for( const auto event : { MagAOX::app::dev::shmimMonitorEvent::discarded,
                              MagAOX::app::dev::shmimMonitorEvent::connected,
                              MagAOX::app::dev::shmimMonitorEvent::disconnected } )
    {
        ++app.m_metadata.cnt0;
        app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
        const auto epoch = app.m_dmCompletion.snapshot().epoch;
        app.shmimMonitorTiming( {}, event );
        CHECK( app.m_dmCompletion.snapshot().state == State::unavailable );
        CHECK( app.m_dmCompletion.snapshot().epoch > epoch );
        REQUIRE( app.processImage( nullptr, {} ) == 0 );
        CHECK_FALSE( qualified( app.m_dmCompletion.snapshot() ) );
    }
}

/// Keep failed, partial, no-send and unqualified backend attempts from releasing readers.
/** \ingroup dmCompletionIntegration_tests */
TEST_CASE( "DM completion rejects unsuccessful callback endpoints", "[dev::dmCompletionIntegration]" )
{
    // clang-format off
#ifdef DM_COMPLETION_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::dmTimingOutcome( Outcome::partial );
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
#endif
    // clang-format on
    Controller app;
    app.enable();
    REQUIRE( app.receive() == 0 );
    REQUIRE( qualified( app.m_dmCompletion.snapshot() ) );
    const auto firstSequence = app.m_dmCompletion.snapshot().sequence;

    SECTION( "SDK failure" )
    {
        app.m_result = Outcome::failed;
        app.m_return = -4;
    }

    SECTION( "Partial SDK submission" )
    {
        app.m_result = Outcome::partial;
        app.m_return = -4;
    }

    SECTION( "No SDK submission" )
    {
        app.m_result = Outcome::noSend;
    }

    SECTION( "Missing explicit backend qualification" )
    {
        app.m_reportResult = false;
    }

    SECTION( "Saturation post failure after a successful SDK result" )
    {
        dmCompletionIntegration::m_failPost = true;
    }

    ++app.m_metadata.cnt0;
    const auto result = app.receive();
    CHECK( result == ( dmCompletionIntegration::m_failPost ? -1 : app.m_return ) );
    const auto record = app.m_dmCompletion.snapshot();
    CHECK( record.sequence == firstSequence + 1 );
    CHECK( record.inputCounter == 51 );
    CHECK_FALSE( qualified( record ) );
    CHECK( record.state == ( result < 0 ? State::error : State::unqualified ) );
    CHECK( record.finishedNs >= record.startedNs );

    if( app.m_result == Outcome::partial )
    {
        CHECK( record.outcome == Outcome::partial );
    }
}

/// Reject an input overwritten while the controller was reading its channels.
/** \ingroup dmCompletionIntegration_tests */
TEST_CASE( "DM completion validates identity again at the callback endpoint", "[dev::dmCompletionIntegration]" )
{
    // clang-format off
#ifdef DM_COMPLETION_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
#endif
    // clang-format on
    Controller app;
    app.enable();

    SECTION( "Combined counter changed" )
    {
        app.m_inCommand = [&]()
        {
            ++app.m_metadata.cnt0;
        };
    }

    SECTION( "Combined publication timestamp changed" )
    {
        app.m_inCommand = [&]()
        {
            ++app.m_metadata.writetime.tv_nsec;
        };
    }

    SECTION( "Combined stream identity changed" )
    {
        app.m_inCommand = [&]()
        {
            ++app.m_metadata.inode;
        };
    }

    SECTION( "Combined stream still being written" )
    {
        app.m_inCommand = [&]()
        {
            app.m_metadata.write = 1;
        };
    }

    REQUIRE( app.receive() == 0 );
    const auto record = app.m_dmCompletion.snapshot();
    CHECK( record.state == State::unqualified );
    CHECK_FALSE( qualified( record ) );
    CHECK( record.inputCounter == 50 );
}

/// Expose the optional endpoint and its software-only meaning through a local property sink.
/** \ingroup dmCompletionIntegration_tests */
TEST_CASE( "DM completion configuration and INDI are optional", "[dev::dmCompletionIntegration]" )
{
    // clang-format off
#ifdef DM_COMPLETION_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::setupConfig( config );
    MagAOX::app::dev::dm<Controller, float>::loadConfig( config );
    MagAOX::app::dev::dm<Controller, float>::setupDMCompletionINDI();
    MagAOX::app::dev::dm<Controller, float>::updateDMCompletionINDI();
#endif
    // clang-format on
    Driver                   driver;
    Controller               app;
    mx::app::appConfigurator config;
    REQUIRE( app.dmT::setupConfig( config ) == 0 );
    CHECK( app.m_completionStreamName.empty() );
    CHECK_FALSE( app.m_dmCompletion.enabled() );
    REQUIRE( app.setupDMCompletionINDI() == 0 );
    CHECK( app.m_indiP_dmCompletion.getNumElements() == 0 );
    const auto path = std::string( getenv( "MILK_SHM_DIR" ) ) + "/controller.conf";
    mx::app::writeConfigFile( path, { "dm" }, { "completionStream" }, { "completion" } );
    REQUIRE( config.readConfig( path ) == 0 );
    REQUIRE( app.dmT::loadConfig( config ) == 0 );
    CHECK( app.m_completionStreamName == "completion" );
    REQUIRE( app.setupDMCompletionINDI() == 0 );
    CHECK( app.m_indiP_dmCompletion.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_dmCompletion.getNumElements() == 10 );
    CHECK( app.m_indiP_dmCompletionStream.getNumElements() == 2 );
    CHECK( app.m_indiP_dmCompletionStream["name"].get<std::string>() == "completion" );
    CHECK( app.m_indiP_dmCompletionStream["meaning"].get().find( "settling is not measured" ) != std::string::npos );
    app.enable();
    app.m_indiDriver = &driver;
    REQUIRE( app.receive() == 0 );
    app.updateDMCompletionINDI();
    CHECK( app.m_indiP_dmCompletion["qualified"].get<double>() == 1 );
    CHECK( app.m_indiP_dmCompletion["inputCounter"].get<double>() == 50 );
    CHECK( app.m_indiP_dmCompletion["processingUs"].get<double>() >= 0 );
    CHECK_FALSE( driver.m_messages.empty() );
    ++app.m_metadata.cnt0;
    app.m_result = Outcome::noSend;
    REQUIRE( app.receive() == 0 );
    app.updateDMCompletionINDI();
    CHECK( app.m_indiP_dmCompletion["qualified"].get<double>() == 0 );
    CHECK( app.m_indiP_dmCompletion["state"].get<double>() == static_cast<double>( State::unqualified ) );
}

} // namespace dmCompletionIntegrationTest
} // namespace libXWCTest
