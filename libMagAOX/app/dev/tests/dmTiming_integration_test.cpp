/** \file dmTiming_integration_test.cpp
 * \brief Software-only tests of DM callback capture and background recording.
 */
#include "../../../../tests/testXWC.hpp"
#include "../../MagAOXApp.hpp"
#include "../shmimMonitor.hpp"

/// \cond DM_TIMING_INTEGRATION_HELPERS
namespace dmTimingIntegration
{
bool failPost = false; ///< Test-controlled failure of the saturation-worker notification.

/// Post a local semaphore or inject a notification failure.
int semPost( sem_t *sem /**< [in] Workstation-local semaphore. */ );

int semPost( sem_t *sem )
{
    if( failPost )
    {
        errno = EIO;
        return -1;
    }

    return ::sem_post( sem );
}
} // namespace dmTimingIntegration
/// \endcond

#define sem_post dmTimingIntegration::semPost
#include "../dm.hpp"
#undef sem_post

namespace libXWCTest
{
namespace dmTimingIntegrationTest
{
using namespace MagAOX::dmTiming;

constexpr auto compatibleCode =
    MagAOX::logger::telem_fgtimings::eventCode; ///< Legacy record code without an ODR reference.

/// \cond DM_TIMING_INTEGRATION_HELPERS
/// Capture INDI updates without opening a FIFO or network connection.
struct Driver
{
    std::vector<pcf::IndiProperty> m_messages; ///< Ordinary-thread property snapshots.

    /// Record a software-only update.
    void sendSetProperty( const pcf::IndiProperty &property /**< [in] Published snapshot. */ );

    /// Record a software-only definition.
    void sendDefProperty( const pcf::IndiProperty &property /**< [in] Defined snapshot. */ );

    /// Record a software-only removal.
    void sendDelProperty( const pcf::IndiProperty &property /**< [in] Removed snapshot. */ );
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

/// Own a DM callback without monitor, saturation or hardware threads.
struct Controller : MagAOX::app::MagAOXApp<false>,
                    MagAOX::app::dev::dm<Controller, float>,
                    MagAOX::app::dev::shmimMonitor<Controller>
{
    friend class MagAOX::app::dev::dm<Controller, float>;
    friend class MagAOX::app::dev::shmimMonitor<Controller>;
    using dmT = MagAOX::app::dev::dm<Controller, float>;
    using dmT::m_deltaChannels;
    using dmT::m_dmTiming;
    using dmT::m_indiP_dmStages;
    using dmT::m_indiP_dmTiming;
    using dmT::m_indiP_dmTimingQuality;
    using dmT::m_indiP_dmTimingState;
    using dmT::m_timingCapacity;
    using dmT::m_timingWindow;
    using dmT::setupDMTimingINDI;
    using MagAOX::app::dev::shmimMonitor<Controller>::m_imageStream;

    Driver *m_indiDriver{ nullptr }; ///< Shadow the app driver with a local capture sink.

    Outcome                           m_result{ Outcome::success }; ///< Qualified result returned by the software SDK.
    int                               m_return{ 0 };                ///< Callback result independent of SDK status.
    int64_t                           m_sdkReturn{ -1 }; ///< Exact stub SDK return marker for latency comparison.
    std::vector<flatlogs::eventCodeT> m_records;         ///< Event types emitted by the consumer.
    std::vector<uint8_t>              m_detailed;        ///< Most recent detailed FlatBuffer payload.
    std::vector<uint8_t>              m_compatible;      ///< Most recent compatible FlatBuffer payload.

    /// Initialize isolated software timing and semaphore state.
    Controller();

    /// Destroy the local semaphore after all callbacks have completed.
    ~Controller() noexcept;

    /// Satisfy the app interface without starting a hardware or logger thread.
    int appStartup() override;

    /// Satisfy the app interface without starting a monitor thread.
    int appLogic() override;

    /// Satisfy the app interface without altering the local fixture lifetime.
    int appShutdown() override;

    /// Satisfy the DM interface without initializing hardware.
    int initDM();

    /// Satisfy the DM interface without commanding hardware.
    int zeroDM();

    /// Satisfy the DM interface without releasing hardware.
    int releaseDM();

    /// Simulate actuator conversion and one SDK return, without accessing hardware.
    int commandDM( void *pixels /**< [in] Unused software command pixels. */ );

    /// Capture telemetry payloads without starting a logger thread.
    template <class logT>
    int telem( const typename logT::messageT &message /**< [in] Serialized consumer summary. */ );
};

Controller::Controller() : MagAOXApp( "test", false )
{
    m_configName  = "dm-timing-test";
    m_calibDir    = "/tmp/dm-timing-test";
    m_flatPath    = "/tmp";
    m_testPath    = "/tmp";
    m_imageStream = {};
    m_dmTiming.configure( 2, 20 );

    if( sem_init( &m_satSemaphore, 0, 0 ) != 0 )
    {
        throw std::runtime_error( "local semaphore initialization" );
    }
}

Controller::~Controller() noexcept
{
    sem_destroy( &m_satSemaphore );
    dmTimingIntegration::failPost = false;
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

    if( m_result == Outcome::noSend )
    {
        dmTimingOutcome( m_result );
        return m_return;
    }

    dmTimingMark( Marker::actuatorStart );
    dmTimingMark( Marker::actuatorEnd );
    dmTimingMark( Marker::sdkStart );
    m_sdkReturn = MagAOX::app::dev::dmTiming::monotonicNowNs();
    m_dmTiming.markAt( Marker::sdkEnd, m_sdkReturn );
    dmTimingOutcome( m_result );
    dmTimingMark( Marker::satStart );
    dmTimingMark( Marker::satEnd );
    return m_return;
}

template <class logT>
int Controller::telem( const typename logT::messageT &message )
{
    m_records.push_back( static_cast<flatlogs::eventCodeT>( logT::eventCode ) );
    auto &payload = logT::eventCode == MagAOX::logger::telem_dmtimings::eventCode ? m_detailed : m_compatible;
    payload.assign( message.builder.GetBufferPointer(),
                    message.builder.GetBufferPointer() + message.builder.GetSize() );
    return 0;
}

/// \endcond

/** \defgroup dmTimingIntegration_tests DM Timing Integration Tests
 * \ingroup app_dev_unit_tests
 */

/// Capture exact SDK markers and preserve successful sends after later callback failure.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM callback timing retains the actual SDK endpoint", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
    MagAOX::app::dev::dm<Controller, float>::dmTimingMark( Marker::sdkEnd );
    MagAOX::app::dev::dm<Controller, float>::dmTimingOutcome( Outcome::partial );
#endif
    // clang-format on

    Controller app;
    const auto receipt = MagAOX::app::dev::dmTiming::monotonicNowNs();
    app.m_dmTiming.beginEventAt( true, receipt );
    REQUIRE( app.processImage( nullptr, MagAOX::app::dev::shmimT() ) == 0 );
    app.updateDMTimings();
    const auto &summary = app.dmTimingSummary();
    CHECK( summary.total.receipts == 1 );
    CHECK( summary.total.submitted == 1 );
    CHECK( summary.metrics[metricIndex( Metric::receiptSend )].count == 1 );
    CHECK( summary.metrics[metricIndex( Metric::receiptSend )].mean ==
           Approx( ( app.m_sdkReturn - receipt ) * 1e-9 ).margin( 1e-15 ) );
    CHECK( summary.metrics[metricIndex( Metric::delta )].count == 0 );
    CHECK( summary.metrics[metricIndex( Metric::throughDelta )].count == 1 );
    CHECK( summary.metrics[metricIndex( Metric::receiptCallback )].mean >=
           summary.metrics[metricIndex( Metric::receiptSend )].mean );
    CHECK_FALSE( app.m_dmTiming.active() );

    dmTimingIntegration::failPost = true;
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, {} ) == -1 );
    app.updateDMTimings();
    CHECK( app.dmTimingSummary().total.submitted == 2 );
    CHECK( app.dmTimingSummary().total.failed == 1 );
    CHECK( app.dmTimingSummary().metrics[metricIndex( Metric::receiptSend )].count == 2 );
    CHECK_FALSE( app.m_dmTiming.active() );

    app.m_result = Outcome::partial;
    app.m_return = -4;
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, {} ) == -4 );
    app.updateDMTimings();
    CHECK( app.dmTimingSummary().total.partial == 1 );
    CHECK( app.dmTimingSummary().total.failed == 1 );
}

/// Compare observed stream publication cadence with independent local receipt cadence.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing captures the input stream publication timestamp", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
    MagAOX::app::dev::dm<Controller, float>::updateINDI();
#endif
    // clang-format on

    Driver         driver;
    Controller     app;
    IMAGE_METADATA metadata{};
    app.m_imageStream.md = &metadata;
    REQUIRE( app.setupDMTimingINDI() == 0 );
    app.m_indiDriver = &driver;
    const auto base  = MagAOX::app::dev::dmTiming::monotonicNowNs() - 5000000;

    for( unsigned i = 0; i < 3; ++i )
    {
        metadata.cnt0      = 10 + i;
        metadata.writetime = { 1700000000, i * 500000L };
        app.m_dmTiming.beginEventAt( true, base + i * i * 500000 );
        REQUIRE( app.processImage( nullptr, {} ) == 0 );
    }

    app.updateDMTimings();
    const auto &publication = app.dmTimingSummary().metrics[metricIndex( Metric::publicationInterval )];
    CHECK( publication.count == 2 );
    CHECK( publication.mean == Approx( 0.0005 ) );
    CHECK( publication.stddev == 0 );
    CHECK( publication.span == Approx( 0.002 ) );
    CHECK( app.dmTimingSummary().metrics[metricIndex( Metric::receiptInterval )].mean == Approx( 0.001 ) );
    REQUIRE( app.dmT::updateINDI() == 0 );
    CHECK( app.m_indiP_dmStages["publicationInterval_mean"].get<double>() == Approx( 0.0005 ) );
    CHECK( app.m_indiP_dmStages["publicationInterval_count"].get<double>() == 2 );
    REQUIRE( app.recordDMTimings() == 0 );
    const auto *record = MagAOX::logger::GetTelem_dmtimings_fb( app.m_detailed.data() );
    CHECK( record->metrics()->Get( metricIndex( Metric::publicationInterval ) )->mean() == Approx( 0.0005 ) );
    const auto *compatible = MagAOX::logger::GetTelem_fgtimings_fb( app.m_compatible.data() );
    CHECK( compatible->atime() == Approx( 0.001 ) );
    app.m_imageStream.md = nullptr;
}

/// Reject unset, malformed, overflowing and actively written publication metadata.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing excludes invalid publication metadata", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
#endif
    // clang-format on

    for( const auto stamp : { timespec{ 0, 0 },
                              timespec{ -1, 0 },
                              timespec{ 1700000000, -1 },
                              timespec{ 1700000000, 1000000000 },
                              timespec{ std::numeric_limits<int64_t>::max() / 1000000000 + 1, 0 },
                              timespec{ 1700000000, 500000 } } )
    {
        Controller     app;
        IMAGE_METADATA metadata{};
        app.m_imageStream.md = &metadata;
        const auto base      = MagAOX::app::dev::dmTiming::monotonicNowNs() - 5000000;

        for( unsigned i = 0; i < 4; ++i )
        {
            metadata.cnt0      = 10 + i;
            metadata.writetime = i == 1 ? stamp : timespec{ 1700000000, i * 500000L };
            metadata.write     = i == 1 && stamp.tv_sec == 1700000000 && stamp.tv_nsec == 500000;
            app.m_dmTiming.beginEventAt( true, base + i * 500000 );
            REQUIRE( app.processImage( nullptr, {} ) == 0 );
        }

        app.updateDMTimings();
        const auto &publication = app.dmTimingSummary().metrics[metricIndex( Metric::publicationInterval )];
        CHECK( publication.count == 1 );
        CHECK( publication.mean == Approx( 0.0005 ) );
        CHECK( app.dmTimingSummary().total.invalid == 0 );
        app.m_imageStream.md = nullptr;
    }
}

/// Keep receipt-only events and initial/no-send/error callbacks from borrowing stale endpoints.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing handles synthetic callbacks and rejected receipts", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    MagAOX::app::dev::dm<Controller, float>::processImage( nullptr, {} );
#endif
    // clang-format on

    Controller app;
    REQUIRE( app.processImage( nullptr, {} ) == 0 );
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::discarded );
    app.m_result = Outcome::noSend;
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, {} ) == 0 );
    app.m_result = Outcome::failed;
    app.m_return = -7;
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, {} ) == -7 );
    app.updateDMTimings();
    const auto &summary = app.dmTimingSummary();
    CHECK( summary.total.events == 4 );
    CHECK( summary.total.synthetic == 1 );
    CHECK( summary.total.receipts == 3 );
    CHECK( summary.total.rejected == 1 );
    CHECK( summary.total.noSend == 1 );
    CHECK( summary.total.failed == 1 );
    CHECK( summary.total.submitted == 1 );
    CHECK( summary.metrics[metricIndex( Metric::receiptInterval )].count == 2 );
    CHECK( summary.metrics[metricIndex( Metric::receiptSend )].count == 0 );
    CHECK_FALSE( summary.fgValid );
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::connected );
    app.updateDMTimings();
    CHECK( app.dmTimingSummary().metrics[metricIndex( Metric::receiptInterval )].count == 0 );
}

/// Emit coherent background records, including unchanged forced records, without hot-path logging.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing records only outside the command path", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::recordDMTimings( true );
    MagAOX::app::dev::dm<Controller, float>::updateDMTimings();
    MagAOX::app::dev::dm<Controller, float>::dmTimingSummary();
#endif
    // clang-format on

    Controller app;

    for( unsigned i = 0; i < 3; ++i )
    {
        app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
        REQUIRE( app.processImage( nullptr, {} ) == 0 );
    }

    REQUIRE( app.m_records.empty() );
    app.updateDMTimings();
    REQUIRE( app.dmTimingSummary().fgValid );
    REQUIRE( app.recordDMTimings() == 0 );
    REQUIRE( app.m_records.size() == 2 );
    CHECK( app.m_records[0] == MagAOX::logger::telem_dmtimings::eventCode );
    CHECK( app.m_records[1] == compatibleCode );
    const auto *detailed   = MagAOX::logger::GetTelem_dmtimings_fb( app.m_detailed.data() );
    const auto *compatible = MagAOX::logger::GetTelem_fgtimings_fb( app.m_compatible.data() );
    CHECK( detailed->sequence() == app.dmTimingSummary().sequence );
    CHECK( compatible->atime() == app.dmTimingSummary().metrics[metricIndex( Metric::receiptInterval )].mean );
    REQUIRE( app.recordDMTimings() == 0 );
    CHECK( app.m_records.size() == 2 );
    REQUIRE( app.recordDMTimings( true ) == 0 );
    CHECK( app.m_records.size() == 4 );

    app.m_dmTiming.update( MagAOX::app::dev::dmTiming::monotonicNowNs() + 3000000000LL );
    REQUIRE( app.recordDMTimings( true ) == 0 );
    CHECK( app.m_records.size() == 5 );
    CHECK( app.m_records.back() == MagAOX::logger::telem_dmtimings::eventCode );
    CHECK_FALSE( MagAOX::logger::GetTelem_dmtimings_fb( app.m_detailed.data() )->fg_valid() );
}

/// Validate runtime configuration without enabling instrument access.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing configuration is runtime controlled", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::setupConfig( config );
    MagAOX::app::dev::dm<Controller, float>::loadConfig( config );
#endif
    // clang-format on

    Controller               app;
    mx::app::appConfigurator config;
    REQUIRE( app.dmT::setupConfig( config ) == 0 );
    REQUIRE( app.m_timingWindow == 2 );
    REQUIRE( app.m_timingCapacity == 20000 );
    const std::string path = "/tmp/dm-timing-config-" + std::to_string( getpid() ) + ".conf";
    mx::app::writeConfigFile( path, { "dm", "dm" }, { "latencyTime", "latencySize" }, { "0", "12" } );
    config.readConfig( path );
    REQUIRE( app.dmT::loadConfig( config ) == 0 );
    CHECK( app.m_timingWindow == 0 );
    CHECK( app.m_timingCapacity == 12 );
    unlink( path.c_str() );
}

/// Publish actual statistics and unavailable sentinels through a local INDI sink.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing INDI follows valid, stale and disabled summaries", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::updateINDI();
    MagAOX::app::dev::dm<Controller, float>::setupDMTimingINDI();
#endif
    // clang-format on
    Driver     driver;
    Controller app;
    REQUIRE( app.setupDMTimingINDI() == 0 );
    app.m_indiDriver = &driver;
    REQUIRE( app.m_indiP_dmTiming.getNumElements() == 15 );
    REQUIRE( app.m_indiP_dmStages.getNumElements() ==
             6 * ( metricIndex( Metric::count ) - metricIndex( Metric::dispatch ) ) );
    REQUIRE( app.m_indiP_dmTimingQuality.getNumElements() == 21 );
    app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, {} ) == 0 );
    app.updateDMTimings();
    REQUIRE( app.dmT::updateINDI() == 0 );
    CHECK( app.m_indiP_dmTiming["receipt_fps"].get<double>() == -1 );
    CHECK( app.m_indiP_dmTiming["receipt_send_count"].get<double>() == 1 );
    CHECK( app.m_indiP_dmTiming["receipt_send"].get<double>() >= 0 );
    CHECK( app.m_indiP_dmTiming["receipt_send_jitter"].get<double>() == -1 );
    CHECK( app.m_indiP_dmStages["delta_count"].get<double>() == 0 );
    CHECK( app.m_indiP_dmStages["delta_mean"].get<double>() == -1 );
    CHECK( app.m_indiP_dmStages["sdkTail_count"].get<double>() == 0 );
    CHECK( app.m_indiP_dmStages["sdkTail_mean"].get<double>() == -1 );
    CHECK( app.m_indiP_dmStages["sdkService_count"].get<double>() == 0 );
    CHECK( app.m_indiP_dmStages["sdkService_mean"].get<double>() == -1 );

    for( const auto *name : { "conversion", "sourceCheck", "shapePublication" } )
    {
        CHECK( app.m_indiP_dmStages[std::string( name ) + "_count"].get<double>() == 0 );
        CHECK( app.m_indiP_dmStages[std::string( name ) + "_mean"].get<double>() == -1 );
    }

    const auto &prefix = app.dmTimingSummary().metrics[metricIndex( Metric::receiptSdk )];
    REQUIRE( prefix.count == 1 );
    CHECK( app.m_indiP_dmStages["receiptSdk_count"].get<double>() == 1 );
    CHECK( app.m_indiP_dmStages["receiptSdk_mean"].get<double>() == prefix.mean );
    CHECK( app.m_indiP_dmStages["receiptSdk_max"].get<double>() == prefix.max );
    CHECK( app.m_indiP_dmStages["receiptSdk_jitter"].get<double>() == -1 );

    for( unsigned i = 0; i < 2; ++i )
    {
        app.shmimMonitorTiming( {}, MagAOX::app::dev::shmimMonitorEvent::receipt );
        REQUIRE( app.processImage( nullptr, {} ) == 0 );
    }

    app.updateDMTimings();
    REQUIRE( app.dmT::updateINDI() == 0 );
    CHECK( app.m_indiP_dmTiming["receipt_fps"].get<double>() ==
           Approx( 1 / app.dmTimingSummary().metrics[metricIndex( Metric::receiptInterval )].mean ) );
    CHECK( app.m_indiP_dmTiming["receipt_count"].get<double>() == 2 );
    CHECK( app.m_indiP_dmTimingQuality["submitted"].get<double>() == 3 );
    CHECK( app.m_indiP_dmTimingQuality["valid"].get<double>() == 1 );
    CHECK( app.m_indiP_dmTimingState["state"].get<std::string>() == "valid" );
    REQUIRE_FALSE( driver.m_messages.empty() );
    app.m_dmTiming.update( MagAOX::app::dev::dmTiming::monotonicNowNs() + 3000000000LL );
    REQUIRE( app.dmT::updateINDI() == 0 );
    CHECK( app.m_indiP_dmTiming["receipt_fps"].get<double>() == -1 );
    CHECK( app.m_indiP_dmTiming["receipt_send"].get<double>() == -1 );
    CHECK( app.m_indiP_dmTimingQuality["valid"].get<double>() == 0 );
    CHECK( app.m_indiP_dmTimingState["state"].get<std::string>() == "stale" );
    app.m_dmTiming.configure( 0, 20 );
    app.updateDMTimings();
    REQUIRE( app.dmT::updateINDI() == 0 );
    CHECK( app.m_indiP_dmTimingState["state"].get<std::string>() == "disabled" );
    CHECK( app.m_indiP_dmTimingQuality["enabled"].get<double>() == 0 );
}

/// Age and publish historical timing when power management bypasses appLogic.
/** \ingroup dmTimingIntegration_tests */
TEST_CASE( "DM timing continues aging in the power-off loop", "[dev::dmTimingIntegration]" )
{
    // clang-format off
#ifdef DM_TIMING_INTEGRATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<Controller, float>::whilePowerOff();
#endif
    // clang-format on
    Driver     driver;
    Controller app;
    REQUIRE( app.setupDMTimingINDI() == 0 );
    app.m_indiDriver = &driver;
    const auto base  = MagAOX::app::dev::dmTiming::monotonicNowNs() - 3000000000LL;

    for( unsigned i = 0; i < 3; ++i )
    {
        auto receipt = base + i * 500000;
        app.m_dmTiming.beginEventAt( true, receipt );
        app.m_dmTiming.markAt( Marker::entry, receipt + 10 );
        app.m_dmTiming.markAt( Marker::sdkStart, receipt + 20 );
        app.m_dmTiming.markAt( Marker::sdkEnd, receipt + 30 );
        app.dmTimingOutcome( Outcome::success );
        app.m_dmTiming.markAt( Marker::finish, receipt + 40 );
        app.m_dmTiming.finishEvent();
    }

    app.m_dmTiming.update( base + 1000100 );
    REQUIRE( app.dmTimingSummary().fgValid );
    REQUIRE( app.recordDMTimings() == 0 );
    REQUIRE( app.dmT::updateINDI() == 0 );
    CHECK( app.m_indiP_dmTimingState["state"].get<std::string>() == "valid" );
    REQUIRE( app.dmT::whilePowerOff() == 0 );
    CHECK_FALSE( app.dmTimingSummary().fgValid );
    CHECK( app.dmTimingSummary().state == Availability::stale );
    CHECK( app.m_indiP_dmTimingState["state"].get<std::string>() == "stale" );
    CHECK( app.m_records.size() == 3 );
    CHECK_FALSE( MagAOX::logger::GetTelem_dmtimings_fb( app.m_detailed.data() )->fg_valid() );
}

} // namespace dmTimingIntegrationTest
} // namespace libXWCTest
