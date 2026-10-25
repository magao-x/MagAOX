/** \file ocam2KCtrl_housekeeping_test.cpp
 * \brief Offline tests of the bounded OCAM serial-housekeeping pause.
 * \ingroup ocam2KCtrl_housekeeping_unit_test
 */

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
#define OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "ocam2KCtrl_test.cpp"
#undef OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "../../../tests/outletAppTest.hpp"
/// \endcond

#include "../../../tests/testXWC.hpp"

namespace libXWCTest
{
/** \addtogroup ocam2KCtrl_housekeeping_unit_test */
namespace ocam2KCtrlTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace
{

/// Set a stable operating state in which a housekeeping pause is permitted.
void prepareHousekeeping( ocam2KCtrl_test &app /**< [in,out] Offline camera controller. */ )
{
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_reconfig    = false;
    app.m_shutdown    = false;
    app.m_indiP_temps = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_temps.setDevice( app.m_configName );
    app.m_indiP_temps.setName( "temps" );

    for( const auto &name : { "cpu", "power", "bias", "water", "left", "right", "cooling" } )
    {
        app.m_indiP_temps.add( pcf::IndiElement( name, 0 ) );
    }
}

/// Queue one successful sequence of the camera's existing automatic serial queries.
void queueHousekeeping( bool synchro /**< [in] Whether external synchronization bypasses the FPS query. */ )
{
    queueSerialResponse( "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                         "SET[-450]\nCooling Power [102]mW.\n\n" );

    if( !synchro )
    {
        queueSerialResponse( "fps [2000] Hz\n" );
    }

    queueSerialResponse( "Gain set to 42 \n\n" );
}

/// Construct an operator request using the registered housekeeping switch identity.
pcf::IndiProperty pauseRequest( const ocam2KCtrl_test &app /**< [in] Controller whose property is addressed. */,
                                bool                   pause /**< [in] Requested pause state. */ )
{
    pcf::IndiProperty request(
        pcf::IndiProperty::Switch, app.m_indiP_housekeepingPause.getDevice(), app.m_indiP_housekeepingPause.getName() );
    request.add( pcf::IndiElement( "toggle" ) );
    request["toggle"].setSwitchState( pause ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return request;
}

} // namespace
/// \endcond

/// A quiet interval suppresses all automatic serial queries and resumes the original sequence afterward.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl pauses automatic serial housekeeping", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::pollHousekeeping(); ocam2KCtrl::getFPS();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    app.m_synchro     = GENERATE( false, true );
    app.m_syncFreq    = 2000;
    app.m_ccdTemp     = -45;
    app.m_emGain      = 77;
    const int64_t now = app.housekeepingNow();

    REQUIRE( app.setHousekeepingPause( true, now ) == 0 );
    REQUIRE( app.m_housekeepingPauseUntil == now + 120000000000LL );

    for( int pass = 0; pass < 3; ++pass )
    {
        REQUIRE( app.pollHousekeeping() == 0 );
    }

    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.m_housekeepingSerialTransactions == 0 );
    REQUIRE( app.m_housekeepingSkippedTemp == 3 );
    REQUIRE( app.m_housekeepingSkippedGain == 3 );
    REQUIRE( app.m_housekeepingSkippedFps == ( app.m_synchro ? 0 : 3 ) );
    REQUIRE( app.m_ccdTemp == -45 );
    REQUIRE( app.m_emGain == 77 );
    REQUIRE( app.state() == stateCodes::OPERATING );

    REQUIRE( app.setHousekeepingPause( false, app.housekeepingNow() ) == 0 );
    queueHousekeeping( app.m_synchro );
    REQUIRE( app.pollHousekeeping() == 0 );
    const std::vector<std::string> expected =
        app.m_synchro ? std::vector<std::string>{ "temp", "gain" } : std::vector<std::string>{ "temp", "fps", "gain" };
    REQUIRE( g_edtStubState.serialCommands == expected );
    REQUIRE( app.m_housekeepingSerialTransactions == expected.size() );
    REQUIRE( app.m_housekeepingTempTime > 0 );
    REQUIRE( app.m_housekeepingGainTime > 0 );
    REQUIRE( ( app.m_housekeepingFpsTime > 0 ) == !app.m_synchro );
    REQUIRE( app.m_emGain == 42 );
}

/// Repeated On requests retain the original hard deadline and polling resumes after automatic expiry.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl housekeeping pause has a nonextendable deadline", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::housekeepingPaused(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    const int64_t now = app.housekeepingNow();
    REQUIRE( app.setHousekeepingPause( true, now ) == 0 );
    const int64_t deadline = app.m_housekeepingPauseUntil;
    REQUIRE( app.setHousekeepingPause( true, now + 60000000000LL ) == 0 );
    REQUIRE( app.m_housekeepingPauseUntil == deadline );
    REQUIRE( app.housekeepingPaused( deadline - 1 ) );
    REQUIRE_FALSE( app.housekeepingPaused( deadline ) );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );

    REQUIRE( app.setHousekeepingPause( true, now ) == 0 );
    app.m_housekeepingPauseUntil = app.housekeepingNow() - 1;
    app.m_synchro                = false;
    queueHousekeeping( false );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "fps", "gain" } );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );
}

/// Invalid lifecycle states cannot start a pause and invalidate an existing one.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl bounds housekeeping pause to stable camera states", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::housekeepingPaused();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    const int64_t now = app.housekeepingNow();
    REQUIRE( app.setHousekeepingPause( true, now ) == 0 );

    SECTION( "reconfiguration pending" )
    {
        app.m_reconfig = true;
    }

    SECTION( "power off" )
    {
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
    }

    SECTION( "power-off requested" )
    {
        app.m_powerTargetState = 0;
    }

    SECTION( "error state" )
    {
        app.state( stateCodes::ERROR );
    }

    SECTION( "connecting state" )
    {
        app.state( stateCodes::CONNECTED );
    }

    SECTION( "shutdown requested" )
    {
        app.m_shutdown = true;
    }

    REQUIRE_FALSE( app.housekeepingPaused( now + 1 ) );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );
    REQUIRE( app.setHousekeepingPause( true, now + 2 ) < 0 );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );
    REQUIRE( app.setHousekeepingPause( false, now + 3 ) == 0 );
    REQUIRE( g_edtStubState.serialCommands.empty() );
}

/// Both READY and OPERATING accept the testing interval without changing the camera FSM.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl accepts a housekeeping pause while ready", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    app.state( stateCodes::READY );
    REQUIRE( app.setHousekeepingPause( true, app.housekeepingNow() ) == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( g_edtStubState.serialCommands.empty() );
}

/// A pause cannot become active while a serial-transaction mutex is owned by another thread.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl waits for serial ownership before pausing housekeeping", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::newCallBack_m_indiP_housekeepingPause();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    prepareHousekeeping( app );
    REQUIRE( app.initHousekeeping() == 0 );
    const auto                             request = pauseRequest( app, true );
    std::promise<void>                     attempted;
    std::future<void>                      entered = attempted.get_future();
    std::unique_lock<std::recursive_mutex> transaction( app.m_cameraMutex );
    auto                                   pause = std::async( std::launch::async,
                             [&]()
                             {
                                 attempted.set_value();
                                 return app.newCallBack_m_indiP_housekeepingPause( request );
                             } );
    entered.wait();
    const auto    waiting  = pause.wait_for( std::chrono::milliseconds( 20 ) );
    const int64_t released = app.housekeepingNow();
    transaction.unlock();
    const int result = pause.get();
    REQUIRE( waiting == std::future_status::timeout );
    REQUIRE( result == 0 );
    REQUIRE( app.m_housekeepingPauseStarted >= released );
    REQUIRE( app.m_housekeepingPauseUntil == app.m_housekeepingPauseStarted + 120000000000LL );
    REQUIRE( app.housekeepingPaused( app.housekeepingNow() ) );
}

/// An operator serial request ends the quiet interval and still performs the requested I/O.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl explicit serial commands cancel housekeeping pause", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::pdvSerialWriteRead(); ocam2KCtrl::setTempSetPt();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    REQUIRE( app.setHousekeepingPause( true, app.housekeepingNow() ) == 0 );

    SECTION( "successful setting command" )
    {
        app.m_ccdTempSetpt = -40;
        queueSerialResponse( "setpoint updated\n" );
        REQUIRE( app.setTempSetPt() == 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp -40.000000" } );
    }

    SECTION( "failed explicit serial command also ends interval" )
    {
        std::string response;
        queueSerialResponse( "", -1 );
        REQUIRE( app.pdvSerialWriteRead( response, "gain 1" ) < 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain 1" } );
    }

    REQUIRE( app.m_housekeepingPauseUntil == 0 );
    REQUIRE( app.m_housekeepingSerialTransactions == 1 );
    REQUIRE_FALSE( app.housekeepingPaused( app.housekeepingNow() ) );
}

/// Turning cooling off after a pause must validate a fresh temperature instead of accepting a cached warm value.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl refreshes paused temperature before disabling cooling", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempControl(); ocam2KCtrl::getTemps();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    app.m_ccdTemp              = 22;
    app.m_tempControlStatusSet = false;
    app.m_housekeepingTempTime = app.housekeepingNow() - 1;
    REQUIRE( app.setHousekeepingPause( true, app.housekeepingNow() ) == 0 );

    SECTION( "while the pause is active" )
    {
        REQUIRE( app.m_housekeepingPauseUntil > 0 );
    }

    SECTION( "after manually resuming without a fresh poll" )
    {
        REQUIRE( app.setHousekeepingPause( false, app.housekeepingNow() ) == 0 );
    }

    SECTION( "after automatic expiry without a fresh poll" )
    {
        REQUIRE_FALSE( app.housekeepingPaused( app.m_housekeepingPauseUntil ) );
    }

    queueSerialResponse( "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                         "SET[-450]\nCooling Power [102]mW.\n\n" );
    REQUIRE( app.setTempControl() < 0 );
    REQUIRE( app.m_ccdTemp == -45 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );
}

/// Status publishes a fixed timeout, readback ages, suppressed protection polling, and external synchronization.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl reports housekeeping freshness and pause state", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::initHousekeeping(); ocam2KCtrl::updateHousekeeping();
    ocam2KCtrl::newCallBack_m_indiP_housekeepingPause();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    prepareHousekeeping( app );
    REQUIRE( app.initHousekeeping() == 0 );
    REQUIRE( app.m_indiP_housekeeping.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_housekeepingStatus.getPerm() == pcf::IndiProperty::ReadOnly );
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeeping["timeoutSec"].get<double>() == 120 );
    REQUIRE( app.m_indiP_housekeeping["remainingSec"].get<double>() == 0 );
    REQUIRE( app.m_indiP_housekeeping["tempAgeSec"].get<double>() == -1 );
    REQUIRE( app.m_indiP_housekeepingStatus["temperature"].get<std::string>() == "unavailable" );

    app.m_synchro = false;
    queueHousekeeping( false );
    REQUIRE( app.pollHousekeeping() == 0 );
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeepingStatus["temperature"].get<std::string>() == "current" );
    REQUIRE( app.m_indiP_housekeepingStatus["gain"].get<std::string>() == "current" );
    REQUIRE( app.m_indiP_housekeepingStatus["fps"].get<std::string>() == "current" );
    REQUIRE( app.m_indiP_housekeeping["tempAgeSec"].get<double>() >= 0 );

    REQUIRE( app.newCallBack_m_indiP_housekeepingPause( pauseRequest( app, true ) ) == 0 );
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeepingPause["toggle"].getSwitchState() == pcf::IndiElement::On );
    REQUIRE( app.m_indiP_housekeepingStatus["state"].get<std::string>() == "paused" );
    REQUIRE( app.m_indiP_housekeepingStatus["temperature"].get<std::string>() == "stale" );
    REQUIRE( app.m_indiP_housekeepingStatus["gain"].get<std::string>() == "stale" );
    REQUIRE( app.m_indiP_housekeepingStatus["fps"].get<std::string>() == "stale" );
    REQUIRE( app.m_indiP_housekeepingStatus["protectionPolling"].get<std::string>() == "paused" );
    REQUIRE( app.m_indiP_housekeeping["remainingSec"].get<double>() > 119 );
    REQUIRE( app.m_indiP_housekeeping["remainingSec"].get<double>() <= 120 );

    app.m_synchro = true;
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeepingStatus["fps"].get() == "external sync" );
    REQUIRE( app.newCallBack_m_indiP_housekeepingPause( pauseRequest( app, false ) ) == 0 );
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeepingPause["toggle"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.m_indiP_housekeepingStatus["state"].get<std::string>() == "polling" );
    REQUIRE( app.m_indiP_housekeepingStatus["protectionPolling"].get<std::string>() == "polling" );
    REQUIRE( app.m_indiP_housekeepingStatus["temperature"].get<std::string>() == "stale" );
    REQUIRE( app.m_indiP_housekeepingStatus["gain"].get<std::string>() == "stale" );
    queueHousekeeping( true );
    REQUIRE( app.pollHousekeeping() == 0 );
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeepingStatus["temperature"].get<std::string>() == "current" );
    REQUIRE( app.m_indiP_housekeepingStatus["gain"].get<std::string>() == "current" );
}

/// Runtime requests reject incorrect property identity and transitions that cannot provide a quiet interval.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl validates housekeeping pause requests", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::newCallBack_m_indiP_housekeepingPause();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    prepareHousekeeping( app );
    REQUIRE( app.initHousekeeping() == 0 );
    auto request = pauseRequest( app, true );

    SECTION( "incorrect property identity" )
    {
        request.setName( "another_property" );
    }

    SECTION( "camera reconfiguration is pending" )
    {
        app.m_reconfig = true;
    }

    REQUIRE( app.newCallBack_m_indiP_housekeepingPause( request ) < 0 );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );
    REQUIRE( app.m_indiP_housekeepingPause["toggle"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( g_edtStubState.serialCommands.empty() );
}

/// Malformed readbacks do not become fresh and the original gain-trip and serial-error paths remain active.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl preserves housekeeping failure and protection handling", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getTemps(); ocam2KCtrl::getFPS(); ocam2KCtrl::getEMGain(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    app.m_synchro = false;

    SECTION( "malformed temperature response remains unavailable" )
    {
        queueSerialResponse( "Temperatures : CCD[20.4]\n" );
        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_housekeepingTempTime == 0 );
    }

    SECTION( "malformed FPS response remains unavailable" )
    {
        queueSerialResponse( "fps 2000\n" );
        REQUIRE( app.getFPS() == 0 );
        REQUIRE( app.m_housekeepingFpsTime == 0 );
    }

    SECTION( "temperature serial error still stops the polling pass" )
    {
        queueSerialResponse( "", -1 );
        REQUIRE( app.pollHousekeeping() < 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "HV trip after expiry resets gain and enters the existing error state" )
    {
        REQUIRE( app.setHousekeepingPause( true, app.housekeepingNow() ) == 0 );
        app.m_housekeepingPauseUntil = app.housekeepingNow() - 1;
        app.m_emGain                 = 77;
        queueSerialResponse( "Temperatures : CCD[-45] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] "
                             "SET[-450]\nCooling Power [102]mW.\n\n" );
        queueSerialResponse( "fps [2000] Hz\n" );
        queueSerialResponse( "HV trip\n" );
        REQUIRE( app.pollHousekeeping() < 0 );
        REQUIRE( app.m_emGain == 1 );
        REQUIRE( app.m_housekeepingGainTime == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "fps", "gain" } );
    }
}

/// The pause suppresses serial queries while keeping the protection-reset confirmation timeout running.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl housekeeping pause preserves confirmation timeout", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    prepareHousekeeping( app );
    app.m_protectionResetConfirmed = 1;
    app.m_protectionResetReqTime   = mx::sys::get_curr_time() - 11;
    REQUIRE( app.setHousekeepingPause( true, app.housekeepingNow() ) == 0 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_protectionResetConfirmed == 0 );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.housekeepingPaused( app.housekeepingNow() ) );
}

/// The real app loop continues telemetry and external-FPS bookkeeping during a serial-housekeeping pause.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl app logic keeps running during housekeeping pause", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::appLogic(); ocam2KCtrl::pollHousekeeping(); ocam2KCtrl::recordCamera();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    startupScope    startup( app );
    configureStartupMode( app );
    REQUIRE( app.dev::telemeter<ocam2KCtrl>::appStartup() == 0 );
    startup.markStarted( true );
    prepareHousekeeping( app );
    app.m_synchro     = true;
    app.m_syncFreq    = 2000;
    app.m_fps         = 1900;
    app.m_maxInterval = 10;
    REQUIRE( app.setHousekeepingPause( true, app.housekeepingNow() ) == 0 );
    MagAOX::logger::telem_stdcam::lastRecord = { 0, 0 };
    fgThreadScope fgThread( app );

    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::OPERATING );
    REQUIRE_FALSE( app.m_shutdown );
    REQUIRE( app.m_fps == 2000 );
    REQUIRE( app.m_tel.logThreadRunning() );
    REQUIRE( hasRecordedTime( MagAOX::logger::telem_stdcam::lastRecord ) );
    REQUIRE( app.housekeepingPaused( app.housekeepingNow() ) );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.m_housekeepingSkippedTemp == 1 );
    REQUIRE( app.m_housekeepingSkippedGain == 1 );
    REQUIRE( app.m_housekeepingSkippedFps == 0 );
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
