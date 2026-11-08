/** \file po4ao_run_test.cpp
 * \brief Compile and exercise the production routing worker with offline SDK/INDI boundaries.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../nnReconstructor/tests/nnTestSupport.hpp"
#include "../po4aoRouting.hpp"
#include "../po4ao_utils.hpp"
#include <condition_variable>
#include <future>

/// \cond TEST_HARNESS
namespace po4aoRunHarness
{
using namespace MagAOX::app;
/// Record routing boundary calls without starting a child or contacting a device.
struct RouteIO
{
    std::vector<std::string> events; ///< Ordered boundary calls.

    std::string current{ "aol1_modevalWFS" }; ///< Simulated downstream route.
    bool        failCalibration{ false };     ///< Withhold acknowledgement of queued writer restart.
    bool        failVerify{ false };          ///< Inject a consumption failure.
    /// Capture the current route and a test calibration identity.
    Po4aoRouteSnapshot capture();
    /// Return the observed route.
    std::string route() const
    {
        return current;
    }
    /// Change the test route.
    void switchInput( const std::string &target /**< [in] Requested input. */,
                      Po4aoRouteBackend::Deadline /**< [in] Deadline. */ );
    /// Record applying the default calibration.
    void applyDefault( Po4aoRouteBackend::Deadline /**< [in] Deadline. */ )
    {
        events.push_back( "default" );
        if( failCalibration )
        {
            throw std::runtime_error( "CACAO calibration restart not acknowledged" );
        }
    }
    /// Provide or withhold actual-consumption acknowledgement.
    void verifyConsumption( const std::string & /**< [in] Input. */,
                            Po4aoRouteBackend::Deadline /**< [in] Deadline. */ );
};
/// State used by the included production worker; transport and monitor scheduling are doubles.
struct po4ao : MagAOXApp<true>, dev::shmimMonitor<po4ao>
{
    using shmimMonitorT = dev::shmimMonitor<po4ao>;
    RouteIO               m_routeIO;
    std::string           m_outputChannel{ "po4ao_output" }; ///< Fixed instrument route under test.
    Po4aoRouteSnapshot    m_runSnapshot;
    std::thread           m_runThread;
    std::mutex            m_callbackMutex; ///< App callback and allocation ownership boundary.
    std::atomic<bool>     m_runCancel{ false }, m_runAdmission{ true };
    std::atomic<uint64_t> m_outputSequence{ 0 };
    bool m_runPending{ false }, m_runBusy{ false }, m_runTarget{ true }, m_runRunning{ true }, m_runVerified{ false },
        m_runCheckOnly{ false };
    uint64_t    m_runSequence{ 0 }, m_activeCollectionGeneration{ 1 };
    std::string m_runPhase{ "unverified" }, m_runRoute{ "unknown" }, m_runError, m_runRequestError,
        m_runOwner{ "test-1" };
    std::string m_nnDevice{ "nn" }, m_nnRoutingOwner, m_nnReleasedOwner, m_nnRoutingSource, m_nnRoutingPhase,
        m_nnInstance{ "nn-1" }, m_nnSequence{ "1" }, m_reservedInstance;
    std::string                           m_runtimeInstance{ "test" };
    std::chrono::steady_clock::time_point m_nnObserved{ std::chrono::steady_clock::now() };
    pcf::IndiProperty                     m_indiP_run, m_indiP_runStatus, m_indiP_nnRouting;
    std::atomic<int>                      m_reloadStatus{ 0 };
    WarmupCollection                      m_collection;
    std::condition_variable               m_replayCondition;
    /// Initialize a running fixture with no instrument connection.
    po4ao();
    /// Join any production worker launched by a request.
    ~po4ao();
    /// Acknowledge the reservation through its transport boundary.
    int                   sendNewProperty( const pcf::IndiProperty &p /**< [in] Native reservation request. */ );
    std::atomic<uint64_t> m_modelResetRequest{ 0 }; ///< Pending reset admission boundary.
    /// Production method declarations.
    void runPhase( const std::string &phase /**< [in] Phase. */ );
    /// Await a boundary condition.
    void waitRun( const std::function<bool()> &ready /**< [in] Predicate. */,
                  Po4aoRouteBackend::Deadline  deadline /**< [in] Deadline. */ );
    /// Send reservation intent.
    void requestWriterHold( const std::string &action /**< [in] Action. */ );
    /// Read reservation completion.
    bool writerHeld( bool released /**< [in] Restore acknowledgement. */ );
    /// Check completed restoration under the INDI lock.
    bool reservationReleased() const;
    /// Publish a request rejection.
    int rejectRun( const std::string &reason /**< [in] Rejection reason. */ );
    /// Execute the production worker.
    void runWorker( bool start /**< [in] Target. */ );
    /// Launch and report queued work.
    void updateRunControl();
    /// Native command boundary.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_run );
    /// Native observation boundary.
    INDI_SETCALLBACK_DECL( po4ao, m_indiP_nnRouting );
};
Po4aoRouteSnapshot RouteIO::capture()
{
    events.push_back( "capture" );
    return { current, "default", "" };
}
void RouteIO::switchInput( const std::string &target, Po4aoRouteBackend::Deadline )
{
    events.push_back( "switch" );
    current = target;
}
void RouteIO::verifyConsumption( const std::string &, Po4aoRouteBackend::Deadline )
{
    events.push_back( "verify" );
    if( failVerify )
    {
        throw std::runtime_error( "stalled consumer" );
    }
}
po4ao::po4ao() : MagAOXApp<true>( 0, 0 )
{
    state( stateCodes::OPERATING );
    m_indiP_run.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    for( const auto *element : { "phase", "route", "error", "owner", "verified" } )
    {
        m_indiP_runStatus.add( pcf::IndiElement( element ) );
    }
}
po4ao::~po4ao()
{
    m_runCancel = true;
    if( m_runThread.joinable() )
    {
        m_runThread.join();
    }
}
int po4ao::sendNewProperty( const pcf::IndiProperty &p )
{
    m_routeIO.events.push_back( p["action"].get<std::string>() );
    m_nnRoutingOwner = p["action"].get<std::string>() == "hold" ? m_runOwner : "";
    m_nnRoutingPhase = m_nnRoutingOwner.empty() ? "released" : "held";
    if( m_nnRoutingOwner.empty() )
    {
        m_nnReleasedOwner = m_runOwner;
    }
    m_nnRoutingSource = "nn";
    m_nnObserved      = std::chrono::steady_clock::now();
    return 0;
}
#include "../po4ao_run.hpp"
} // namespace po4aoRunHarness
/// \endcond

namespace libXWCTest::po4aoTest
{
/** \brief Failed consumption never parks the running monitor or acknowledges Stop. */
TEST_CASE( "Production PO4AO worker parks only after verified consumption", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao app;
    app.m_routeIO.failVerify = true;
    app.runWorker( false );
    REQUIRE( app.state() == MagAOX::app::stateCodes::OPERATING );
    REQUIRE( app.m_runAdmission.load() );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE_FALSE( app.m_runVerified );
    REQUIRE( app.m_runError.find( "stalled consumer" ) != std::string::npos );
    app.m_routeIO.failVerify = false;
    app.runWorker( false );
    app.updateRunControl();
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
    REQUIRE_FALSE( app.m_runAdmission.load() );
    REQUIRE( app.m_runPhase == "stopped" );
    REQUIRE( app.m_runVerified );
    REQUIRE( app.m_runSnapshot.writer == "nn" );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
#endif
    // clang-format on
}

/** \brief Stop drains app work without requiring a monitor mapping-closed acknowledgement. */
TEST_CASE( "PO4AO Stop waits for the active callback", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao       app;
    std::unique_lock<std::mutex> callbackLock( app.m_callbackMutex );
    auto                         worker   = std::async( std::launch::async,
                                                        [&]
                                                        {
                                  app.runWorker( false );
                                                        } );
    const auto                   deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( app.m_runAdmission.load() && std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::yield();
    }
    const bool admissionClosed = !app.m_runAdmission.load();
    const bool waiting         = worker.wait_for( std::chrono::milliseconds( 20 ) ) == std::future_status::timeout;
    callbackLock.unlock();
    worker.get();
    REQUIRE( admissionClosed );
    REQUIRE( waiting );
    REQUIRE( app.m_runVerified );
    app.updateRunControl();
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
    MagAOX::app::po4ao::updateRunControl;
#endif
    // clang-format on
}

/** \brief Duplicate verified targets recheck consumption without repeating either utility. */
TEST_CASE( "Production PO4AO duplicate Stop does not recalibrate", "[po4ao][runWorker]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
#endif
    // clang-format on
    po4aoRunHarness::po4ao app;
    app.runWorker( false );
    app.m_routeIO.events.clear();
    app.m_runCheckOnly = true;
    app.runWorker( false );
    REQUIRE( app.m_routeIO.events == std::vector<std::string>{ "verify" } );
    REQUIRE( app.m_runPhase == "stopped" );
    app.m_routeIO.current = "po4ao_output";
    app.runWorker( false );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE_FALSE( app.m_runVerified );
}
/** \brief A rejected native request must terminate the GUI acknowledgement wait. */
TEST_CASE( "Run conflict rejection publishes Alert", "[po4ao][reviewRegression]" )
{
    po4aoRunHarness::po4ao               app;
    MagAOX::app::MagAOXApp<true>::Driver driver;
    app.m_indiDriver = &driver;
    SECTION( "reset" )
    {
        app.m_modelResetRequest = 2;
    }
    SECTION( "loading" )
    {
        app.m_reloadStatus = 1;
    }
    SECTION( "pending activation" )
    {
        app.m_reloadStatus = 2;
    }
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( app.m_indiP_run.getState() == INDI_ALERT );
    REQUIRE_FALSE( app.m_runPending );
    REQUIRE( app.m_runPhase == "unverified" );
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( driver.m_published.size() == 2 );
    REQUIRE( driver.m_published[0].getState() == INDI_ALERT );
    REQUIRE( driver.m_published[1].getState() == INDI_ALERT );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}

/** \brief Retrying a partial operation must preserve the token, instance and recovery snapshot. */
TEST_CASE( "Run retry retains unresolved reservation", "[po4ao][reviewRegression]" )
{
    po4aoRunHarness::po4ao app;
    app.m_runPhase = "failed";
    app.m_runError = "calibration incomplete";
    app.m_runOwner = app.m_nnRoutingOwner = "unresolved-owner";
    app.m_reservedInstance                = app.m_nnInstance;
    app.m_nnRoutingPhase                  = "held";
    app.m_runSnapshot                     = { "po4ao_output", "previous", "nn" };
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( app.m_runOwner == "unresolved-owner" );
    REQUIRE( app.m_reservedInstance == app.m_nnInstance );
    REQUIRE( app.m_runSnapshot.route == "po4ao_output" );
    REQUIRE( app.m_runSnapshot.calibration == "previous" );
    REQUIRE( app.m_runSnapshot.writer == "nn" );
    REQUIRE( app.m_runError == "calibration incomplete" );
    REQUIRE( app.m_indiP_runStatus["error"].get<std::string>().find( "calibration incomplete" ) != std::string::npos );
    REQUIRE_FALSE( app.m_runPending );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}
/** \brief Only a fresh exact-token release from the same NN instance permits a new routing operation. */
TEST_CASE( "Run retry requires matching completed recovery", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao app;
    app.m_runPhase         = "failed";
    app.m_runOwner         = "retained-token";
    app.m_reservedInstance = app.m_nnInstance;
    app.m_nnRoutingPhase   = "released";
    app.m_nnReleasedOwner  = app.m_runOwner;
    bool recovered         = false;
    SECTION( "matching recovery" )
    {
        recovered = true;
    }
    SECTION( "wrong token" )
    {
        app.m_nnReleasedOwner = "earlier-token";
    }
    SECTION( "stale recovery" )
    {
        app.m_nnObserved -= std::chrono::seconds( 6 );
    }
    SECTION( "NN restarted" )
    {
        app.m_nnInstance = "new-instance";
    }
    SECTION( "restoration still running" )
    {
        app.m_nnRoutingPhase = "restoring";
    }
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    MagAOX::app::MagAOXApp<true>::Driver driver;
    app.m_indiDriver = &driver;
    if( app.m_nnInstance != app.m_reservedInstance )
    {
        REQUIRE_THROWS( app.writerHeld( true ) );
    }
    else
    {
        REQUIRE( app.writerHeld( true ) == recovered );
    }
    REQUIRE( ( app.newCallBack_m_indiP_run( request ) == 0 ) == recovered );
    REQUIRE( app.m_runPending == recovered );
    if( recovered )
    {
        REQUIRE( app.m_runOwner != "retained-token" );
        REQUIRE( driver.m_published.back().getState() == INDI_BUSY );
    }
    else
    {
        REQUIRE( app.m_runOwner == "retained-token" );
    }
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::writerHeld;
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}

/** \brief Missing calibration restart acknowledgement retains the reservation and available PO4AO output. */
TEST_CASE( "Run worker does not release or park after incomplete calibration", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao app;
    app.m_routeIO.failCalibration = true;
    app.runWorker( false );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE( app.m_nnRoutingPhase == "held" );
    REQUIRE( app.m_nnRoutingOwner == app.m_runOwner );
    REQUIRE( app.state() == MagAOX::app::stateCodes::OPERATING );
    REQUIRE_FALSE( app.m_runVerified );
    REQUIRE( app.m_routeIO.events == std::vector<std::string>{ "capture", "hold", "switch", "default" } );
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    const auto owner = app.m_runOwner;
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( app.m_runOwner == owner );
    REQUIRE( app.m_runSnapshot.writer == "nn" );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}
} // namespace libXWCTest::po4aoTest
