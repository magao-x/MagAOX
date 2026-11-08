/** \file po4ao_run_test.cpp
 * \brief Compile and exercise the production routing worker with offline SDK/INDI boundaries.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../nnReconstructor/tests/nnTestSupport.hpp"
#include "../po4aoRouting.hpp"
#include "../po4ao_utils.hpp"
#include <condition_variable>

/// \cond TEST_HARNESS
namespace po4aoRunHarness
{
using namespace MagAOX::app;
/// Record routing boundary calls without starting a child or contacting a device.
struct RouteIO
{
    std::vector<std::string> events; ///< Ordered boundary calls.

    std::string current{ "aol1_modevalWFS" }; ///< Simulated downstream route.
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
    std::atomic<bool>     m_runCancel{ false }, m_runAdmission{ true };
    std::atomic<uint64_t> m_outputSequence{ 0 };
    bool m_runPending{ false }, m_runBusy{ false }, m_runTarget{ true }, m_runRunning{ true }, m_runVerified{ false },
        m_runCheckOnly{ false };
    uint64_t    m_runSequence{ 0 }, m_activeCollectionGeneration{ 1 };
    std::string m_runPhase{ "unverified" }, m_runRoute{ "unknown" }, m_runError, m_runOwner{ "test-1" };
    std::string m_nnDevice{ "nn" }, m_nnRoutingOwner, m_nnRoutingSource, m_nnRoutingPhase, m_nnInstance{ "nn-1" },
        m_nnSequence{ "1" }, m_reservedInstance;
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
        throw std::runtime_error( "stalled consumer" );
}
po4ao::po4ao() : MagAOXApp<true>( 0, 0 )
{
    monitorState( stateCodes::OPERATING );
}
po4ao::~po4ao()
{
    m_runCancel = true;
    if( m_runThread.joinable() )
        m_runThread.join();
}
int po4ao::sendNewProperty( const pcf::IndiProperty &p )
{
    m_routeIO.events.push_back( p["action"].get<std::string>() );
    m_nnRoutingOwner  = p["action"].get<std::string>() == "hold" ? m_runOwner : "";
    m_nnRoutingPhase  = m_nnRoutingOwner.empty() ? "released" : "held";
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
    REQUIRE_FALSE( app.monitorIdle() );
    REQUIRE( app.m_runAdmission.load() );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE_FALSE( app.m_runVerified );
    REQUIRE( app.m_runError.find( "stalled consumer" ) != std::string::npos );
    app.m_routeIO.failVerify = false;
    app.runWorker( false );
    REQUIRE( app.monitorIdle() );
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
} // namespace libXWCTest::po4aoTest
