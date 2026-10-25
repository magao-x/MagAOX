/** \file ocam2KCtrl_housekeeping_test.cpp
 * \brief Offline tests of confirmation-driven OCAM housekeeping and its bounded diagnostic pause.
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

/// Controller using a fixed monotonic clock so policy tests do not sleep for their deadlines.
class housekeepingClock : public ocam2KCtrl_test
{
  public:
    int64_t m_now{ 100000000000 }; ///< Test-controlled monotonic time, in nanoseconds.

    /// Return the current test-controlled monotonic time.
    int64_t housekeepingNow() const override;
};

inline int64_t housekeepingClock::housekeepingNow() const
{
    return m_now;
}

using Phase = ocam2KCtrl::ConfirmationPhase;

/// Set a stable camera state without creating any serial readback or setting request.
void prepareHousekeeping( housekeepingClock &app /**< [in,out] Offline camera controller. */ )
{
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_reconfig    = false;
    app.m_shutdown    = false;
    app.m_synchro     = false;
    app.m_indiP_temps = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_temps.setDevice( app.m_configName );
    app.m_indiP_temps.setName( "temps" );

    for( const auto &name : { "cpu", "power", "bias", "water", "left", "right", "cooling" } )
    {
        app.m_indiP_temps.add( pcf::IndiElement( name, 0 ) );
    }
}

/// Seed cached, confirmed settings and settled temperature for pause and quiet-state tests.
void prepareSettledHousekeeping( housekeepingClock &app /**< [in,out] Offline camera controller. */ )
{
    prepareHousekeeping( app );
    app.m_gainConfirmation.phase = Phase::confirmed;
    app.m_fpsConfirmation.phase  = Phase::confirmed;
    app.m_tempConfirmation.phase = Phase::confirmed;
    app.m_housekeepingTempTime   = app.m_now;
    app.m_housekeepingGainTime   = app.m_now;
    app.m_housekeepingFpsTime    = app.m_now;
    app.m_tempChanging           = false;
    app.m_tempSettledSamples     = app.m_tempSettleSamples;
    app.m_tempDue                = app.m_now + 10000000000;
    app.m_tempExpectedControl    = -1;
    app.m_tempExpectedSet        = false;
    app.m_coolingKnown           = true;
    app.m_tempControlStatus      = true;
    app.m_tempControlStatusSet   = true;
    app.m_tempControlOnTarget    = true;
    app.m_temps.CCD              = -45;
    app.m_temps.SET              = -45;
    app.m_temps.COOLING_POWER    = 102;
    app.m_ccdTemp                = -45;
    app.m_ccdTempSetpt           = -45;
    app.m_emGain                 = 42;
    app.m_fps                    = 2000;
}

/// Queue a valid OCAM temperature reply with controllable thermal state.
void queueTemperature( float ccd   = -45 /**< [in] Measured CCD temperature, in C. */,
                       float set   = -45 /**< [in] Reported setpoint, in C. */,
                       float power = 102 /**< [in] Reported cooling power, in mW. */ )
{
    queueSerialResponse( "Temperatures : CCD[" + std::to_string( ccd ) +
                         "] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] SET[" +
                         std::to_string( set * 10 ) + "]\nCooling Power [" + std::to_string( power ) + " ]mW.\n\n" );
}

/// Construct an operator request using the registered housekeeping switch identity.
pcf::IndiProperty pauseRequest( const housekeepingClock &app /**< [in] Controller whose property is addressed. */,
                                bool                     pause /**< [in] Requested pause state. */ )
{
    pcf::IndiProperty request(
        pcf::IndiProperty::Switch, app.m_indiP_housekeepingPause.getDevice(), app.m_indiP_housekeepingPause.getName() );
    request.add( pcf::IndiElement( "toggle" ) );
    request["toggle"].setSwitchState( pause ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return request;
}

} // namespace
/// \endcond

/// A settled simulated minute reads temperature every ten seconds and never polls cached gain or FPS.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl settled housekeeping is temperature only", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::pollHousekeeping(); ocam2KCtrl::getTemps(); ocam2KCtrl::getFPS(); ocam2KCtrl::getEMGain();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    const int64_t start = app.m_now;

    for( int second = 1; second <= 60; ++second )
    {
        app.m_now = start + int64_t( second ) * 1000000000;

        if( second % 10 == 0 )
        {
            queueTemperature();
        }

        const size_t previous = g_edtStubState.serialCommands.size();
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( g_edtStubState.serialCommands.size() - previous <= 1 );
        REQUIRE( app.m_gainConfirmation.phase == Phase::confirmed );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
    }

    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>( 6, "temp" ) );
    REQUIRE( app.m_housekeepingTempQueries == 6 );
    REQUIRE( app.m_housekeepingGainQueries == 0 );
    REQUIRE( app.m_housekeepingFpsQueries == 0 );
    REQUIRE( app.m_housekeepingGainTime == start );
    REQUIRE( app.m_housekeepingFpsTime == start );
    REQUIRE( app.m_housekeepingTempTime == app.m_now );
}

/// Pending confirmations win initial ties, then older due temperature work is serviced without starvation.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl services one oldest due housekeeping query", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::beginConfirmation(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.beginConfirmation( app.m_gainConfirmation, 42, true );
    app.beginConfirmation( app.m_fpsConfirmation, 2000, true );
    app.m_tempDue = app.m_now;
    queueSerialResponse( "Gain set to 41 \n\n" );
    queueSerialResponse( "fps [1999] Hz\n" );
    queueTemperature();

    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain" } );
    ++app.m_now;
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain", "fps" } );
    ++app.m_now;
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain", "fps", "temp" } );
    REQUIRE( app.m_gainConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_fpsConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_housekeepingTempQueries == 1 );
    REQUIRE( app.m_housekeepingGainQueries == 1 );
    REQUIRE( app.m_housekeepingFpsQueries == 1 );
}

/// A startup observation establishes actual settings once without dispatching default setting writes.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl establishes unavailable housekeeping values once", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::resetHousekeeping(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareHousekeeping( app );
    app.m_emGainSet = 1;
    app.m_fpsSet    = 0;
    app.m_tempDue   = app.m_now;
    queueSerialResponse( "Gain set to 42 \n\n" );
    queueSerialResponse( "fps [2000] Hz\n" );
    queueTemperature();

    for( int pass = 0; pass < 3; ++pass )
    {
        REQUIRE( app.pollHousekeeping() == 0 );
        ++app.m_now;
    }

    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain", "fps", "temp" } );
    REQUIRE( app.m_emGain == 42 );
    REQUIRE( app.m_fps == 2000 );
    REQUIRE( app.m_gainConfirmation.phase == Phase::confirmed );
    REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
    REQUIRE( app.m_emGainSet == 1 );
    REQUIRE( app.m_fpsSet == 0 );
}

/// Mismatches and malformed replies never extend a request deadline or cause automatic setting retries.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl bounds gain confirmation retries", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::beginConfirmation(); ocam2KCtrl::pollHousekeeping(); ocam2KCtrl::getEMGain();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.beginConfirmation( app.m_gainConfirmation, 77, true );
    const int64_t deadline = app.m_gainConfirmation.deadline;
    app.m_tempDue          = deadline + 100000000000;
    queueSerialResponse( "Gain set to 42 \n\n" );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_gainConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_gainConfirmation.deadline == deadline );
    app.m_now += 1000000000;
    queueSerialResponse( "Gain set to 42 \n\n" );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_gainConfirmation.deadline == deadline );
    app.m_now = deadline;
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_gainConfirmation.phase == Phase::failed );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain", "gain" } );

    app.m_now += 1000000000;
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_gainConfirmation.phase == Phase::failed );
    REQUIRE( g_edtStubState.serialCommands.size() == 2 );
}

/// A newer requested value supersedes an earlier acknowledgement and keeps its own fixed deadline.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl supersedes pending gain confirmation", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setEMGain(); ocam2KCtrl::beginConfirmation(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_protectionReset = true;
    app.m_emGainSet       = 77;
    queueSerialResponse( "setting accepted\n" );
    REQUIRE( app.setEMGain() == 0 );
    const uint64_t firstGeneration = app.m_gainConfirmation.generation;
    app.m_emGainSet                = 88;
    queueSerialResponse( "setting accepted\n" );
    REQUIRE( app.setEMGain() == 0 );
    REQUIRE( app.m_gainConfirmation.generation > firstGeneration );
    queueSerialResponse( "Gain set to 77 \n\n" );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_gainConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_gainConfirmation.expected == 88 );
    app.m_now += 1000000000;
    queueSerialResponse( "Gain set to 88 \n\n" );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_gainConfirmation.phase == Phase::confirmed );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain 77", "gain 88", "gain", "gain" } );
}

/// A supported numeric gain acknowledgement confirms the integer actually sent without another read.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl accepts verified setter acknowledgements", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setEMGain(); ocam2KCtrl::setFPS(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );

    SECTION( "gain uses the dispatched integer for confirmation" )
    {
        app.m_protectionReset = true;
        app.m_emGainSet       = 42.9;
        queueSerialResponse( "Gain set to 42 \n\n" );
        REQUIRE( app.setEMGain() == 0 );
        REQUIRE( app.m_gainConfirmation.phase == Phase::confirmed );
        REQUIRE( app.m_gainConfirmation.expected == 42 );
        REQUIRE( app.m_emGain == 42 );
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain 42" } );
    }

    SECTION( "FPS zero confirms a positive automatic rate" )
    {
        app.m_fpsSet = 0;
        queueSerialResponse( "fps [2000] Hz\n" );
        REQUIRE( app.setFPS( false ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
        REQUIRE( app.m_fps == 2000 );
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "fps 0.000000" } );
    }
}

/// External same-value requests reuse valid confirmations or obtain one fresh report without serial polling.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl confirms same-value external FPS requests", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setFPS(); ocam2KCtrl::setCallBack_m_indiP_syncFreq(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<housekeepingClock> app;
    app.driver();
    prepareSettledHousekeeping( app );
    app.m_synchro                   = true;
    app.m_syncFreq                  = 2000;
    app.m_fpsSet                    = 2000;
    app.m_syncFreqTime              = app.m_now;
    app.m_fpsConfirmation.requested = true;
    app.m_fpsConfirmation.expected  = 2000;
    app.m_indiP_syncFreq = pcf::IndiProperty( pcf::IndiProperty::Number, app.m_syncDevice, app.m_syncFreqProp );
    auto *driver         = static_cast<outletHarness::Driver *>( app.m_indiDriver );
    static_cast<void>( driver->takePublished() );

    SECTION( "valid unchanged source confirmation is reusable" )
    {
        REQUIRE( app.setFPS( false ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
        REQUIRE( driver->takePublished().find( "<getProperties" ) == std::string::npos );
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE_FALSE( app.m_reconfig );
    }

    SECTION( "invalidated source observation requires one fresh equal report" )
    {
        app.m_syncFreqTime = 0;
        REQUIRE( app.setFPS( false ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::confirming );
        const std::string request = driver->takePublished();
        const size_t      first   = request.find( "<getProperties" );
        REQUIRE( first != std::string::npos );
        REQUIRE( request.find( "<getProperties", first + 1 ) == std::string::npos );
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( driver->takePublished().find( "<getProperties" ) == std::string::npos );
        pcf::IndiProperty report( pcf::IndiProperty::Number, app.m_syncDevice, app.m_syncFreqProp );
        report.add( pcf::IndiElement( "current", 2000 ) );
        ++app.m_now;
        REQUIRE( app.setCallBack_m_indiP_syncFreq( report ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
        REQUIRE_FALSE( app.m_reconfig );
        REQUIRE( app.m_workerFPS.load() == 2000 );
    }

    SECTION( "an unexpected source change invalidates the previous target confirmation" )
    {
        pcf::IndiProperty report( pcf::IndiProperty::Number, app.m_syncDevice, app.m_syncFreqProp );
        report.add( pcf::IndiElement( "current", 1800 ) );
        ++app.m_now;
        REQUIRE( app.setCallBack_m_indiP_syncFreq( report ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::unavailable );
        REQUIRE( app.m_fps == 1800 );
        REQUIRE( app.m_workerFPS.load() == 1800 );
        REQUIRE( app.m_reconfig );
    }

    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.m_housekeepingFpsQueries == 0 );
}

/// External updates preserve observed FPS and reconfiguration while only the latest matching request can confirm.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl bounds external FPS confirmation", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setFPS(); ocam2KCtrl::setCallBack_m_indiP_syncFreq(); ocam2KCtrl::beginConfirmation();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<housekeepingClock> app;
    app.driver();
    prepareSettledHousekeeping( app );
    app.m_synchro        = true;
    app.m_syncFreq       = 2000;
    app.m_fpsSet         = 1800;
    app.m_modeName       = "science";
    app.m_indiP_syncFreq = pcf::IndiProperty( pcf::IndiProperty::Number, app.m_syncDevice, app.m_syncFreqProp );
    REQUIRE( app.setFPS( false ) == 0 );
    const uint64_t oldGeneration = app.m_fpsConfirmation.generation;
    app.m_fpsSet                 = 1900;
    REQUIRE( app.setFPS( false ) == 0 );
    REQUIRE( app.m_fpsConfirmation.generation > oldGeneration );
    pcf::IndiProperty report( pcf::IndiProperty::Number, app.m_syncDevice, app.m_syncFreqProp );
    report.add( pcf::IndiElement( "current", 1800 ) );
    REQUIRE( app.setCallBack_m_indiP_syncFreq( report ) == 0 );
    REQUIRE( app.m_fpsConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_fpsConfirmation.expected == 1900 );
    REQUIRE( app.m_fps == 1800 );
    REQUIRE( app.m_workerFPS.load() == 1800 );
    REQUIRE( app.m_reconfig );
    REQUIRE( app.m_nextMode == "science" );

    SECTION( "latest matching report confirms" )
    {
        report["current"].set( 1900 );
        ++app.m_now;
        REQUIRE( app.setCallBack_m_indiP_syncFreq( report ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
    }

    SECTION( "expired request cannot be confirmed by a later report" )
    {
        app.m_now = app.m_fpsConfirmation.deadline;
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::failed );
        report["current"].set( 1900 );
        ++app.m_now;
        REQUIRE( app.setCallBack_m_indiP_syncFreq( report ) == 0 );
        REQUIRE( app.m_fpsConfirmation.phase == Phase::failed );
        REQUIRE( app.m_fps == 1900 );
    }

    REQUIRE( g_edtStubState.serialCommands.empty() );
}

/// Temperature polling accelerates for convergence and requires consecutive fresh settled samples.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl changes temperature cadence on thermal convergence", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getTemps(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_tempDue = app.m_now;
    queueTemperature( -40 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempChanging );
    REQUIRE( app.m_tempSettledSamples == 0 );
    REQUIRE( app.m_tempDue == app.m_now + 1000000000 );
    app.m_now = app.m_tempDue;
    queueTemperature();
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempChanging );
    REQUIRE( app.m_tempSettledSamples == 1 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempSettledSamples == 1 );
    app.m_now = app.m_tempDue;
    queueTemperature();
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE_FALSE( app.m_tempChanging );
    REQUIRE( app.m_tempDue == app.m_now + 10000000000 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "temp", "temp" } );
}

/// A temperature command keeps its requested target across mismatching or malformed readbacks.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl preserves pending temperature targets", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempSetPt(); ocam2KCtrl::getTemps(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_ccdTempSetpt = -40;
    queueSerialResponse( "setting accepted\n" );
    REQUIRE( app.setTempSetPt() == 0 );
    const int64_t deadline = app.m_tempConfirmation.deadline;

    SECTION( "mismatching returned setpoint" )
    {
        queueTemperature( -45, -45 );
    }

    SECTION( "malformed response" )
    {
        queueSerialResponse( "Temperatures : CCD[-45]\n" );
    }

    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_ccdTempSetpt == -40 );
    REQUIRE( app.m_tempConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_tempConfirmation.deadline == deadline );
    REQUIRE( app.m_tempSettledSamples == 0 );
}

/// A confirmed setpoint may cool for longer than its acknowledgement timeout without becoming unconfirmed.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl separates temperature acknowledgement from settling", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempSetPt(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_ccdTempSetpt = -40;
    queueSerialResponse( "setting accepted\n" );
    REQUIRE( app.setTempSetPt() == 0 );
    const int64_t deadline = app.m_tempConfirmation.deadline;
    queueTemperature( 0, -40 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempConfirmation.phase == Phase::confirmed );
    REQUIRE( app.m_tempChanging );
    app.m_now = deadline + 1000000000;
    queueTemperature( -5, -40 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempConfirmation.phase == Phase::confirmed );
    REQUIRE( app.m_tempChanging );
    REQUIRE( app.m_tempDue == app.m_now + 1000000000 );
}

/// Disabled cooling settles immediately despite an inactive setpoint, while ambiguous power is not proof of state.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl distinguishes disabled and unknown cooling", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getTemps(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_tempDue = app.m_now;

    SECTION( "low power well above setpoint establishes disabled cooling" )
    {
        queueTemperature( 20, -45, 2 );
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( app.m_coolingKnown );
        REQUIRE_FALSE( app.m_tempControlStatus );
        REQUIRE_FALSE( app.m_tempChanging );
        REQUIRE( app.m_tempDue == app.m_now + 10000000000 );
    }

    SECTION( "low power near setpoint is ambiguous" )
    {
        queueTemperature( -45, -45, 2 );
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE_FALSE( app.m_coolingKnown );
        REQUIRE( app.m_tempChanging );
    }
}

/// Unknown or malformed observations restart a bounded fast state check instead of polling fast forever.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl bounds unknown temperature state checks", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getTemps(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_tempDue = app.m_now;

    SECTION( "ambiguous cooling power" )
    {
        queueTemperature( -45, -45, 2 );
    }

    SECTION( "malformed temperature" )
    {
        queueSerialResponse( "Temperatures : CCD[-45]\n" );
    }

    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempChanging );
    REQUIRE( app.m_tempSettledSamples == 0 );
    REQUIRE( app.m_tempConfirmation.phase == Phase::confirming );
    REQUIRE( app.m_tempConfirmation.deadline == app.m_now + 10000000000 );
    REQUIRE( app.m_tempDue == app.m_now + 1000000000 );
    app.m_now = app.m_tempConfirmation.deadline;
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempConfirmation.phase == Phase::failed );
    REQUIRE_FALSE( app.m_tempChanging );
    REQUIRE( app.m_tempDue == app.m_now + 10000000000 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
}

/// Expired temperature requests stop fast retries, retain slow monitoring, and cannot later acknowledge themselves.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl retains slow temperature monitoring after confirmation expires", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempSetPt(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_ccdTempSetpt = -40;
    queueSerialResponse( "setting accepted\n" );
    REQUIRE( app.setTempSetPt() == 0 );
    app.m_now = app.m_tempConfirmation.deadline;
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_tempConfirmation.phase == Phase::failed );
    REQUIRE_FALSE( app.m_tempChanging );
    REQUIRE( app.m_tempDue == app.m_now + 10000000000 );
    REQUIRE( g_edtStubState.serialCommands.size() == 1 );
    app.m_now = app.m_tempDue;
    queueTemperature( -40, -40 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_housekeepingTempTime == app.m_now );
    REQUIRE( app.m_tempConfirmation.phase == Phase::failed );
    REQUIRE_FALSE( app.m_tempChanging );
}

/// Cooling-Off always obtains a valid fresh temperature and applies the existing strict greater-than-19 C guard.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl refreshes temperature before disabling cooling", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempControl(); ocam2KCtrl::getTemps();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_ccdTemp              = 22;
    app.m_tempControlStatusSet = false;

    SECTION( "cached warm temperature cannot permit disabling a now cold camera" )
    {
        queueTemperature( -45 );
        REQUIRE( app.setTempControl() < 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "exactly 19 C remains too cold" )
    {
        queueTemperature( 19 );
        REQUIRE( app.setTempControl() < 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "fresh warm temperature permits the requested Off despite getter state changes" )
    {
        queueTemperature( 20 );
        queueSerialResponse( "setting accepted\n" );
        REQUIRE( app.setTempControl() == 0 );
        REQUIRE_FALSE( app.m_tempControlStatusSet );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp", "temp off" } );
        REQUIRE( app.m_tempConfirmation.phase == Phase::confirming );
    }

    SECTION( "malformed temperature cannot bypass the guard" )
    {
        queueSerialResponse( "Temperatures : CCD[22]\n" );
        REQUIRE( app.setTempControl() < 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "failed temperature transport cannot bypass the guard" )
    {
        queueSerialResponse( "", -1 );
        REQUIRE( app.setTempControl() < 0 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }
}

/// Pausing suppresses all serial traffic; resuming makes only one temperature refresh due.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl pauses only settled housekeeping", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    REQUIRE( app.setHousekeepingPause( true, app.m_now ) == 0 );
    const int64_t deadline = app.m_housekeepingPauseUntil;
    app.m_now += 60000000000;
    REQUIRE( app.setHousekeepingPause( true, app.m_now ) == 0 );
    REQUIRE( app.m_housekeepingPauseUntil == deadline );

    for( int pass = 0; pass < 3; ++pass )
    {
        REQUIRE( app.pollHousekeeping() == 0 );
    }

    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.m_housekeepingSerialTransactions == 0 );
    REQUIRE( app.setHousekeepingPause( false, app.m_now ) == 0 );
    queueTemperature();
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    REQUIRE( app.m_gainConfirmation.phase == Phase::confirmed );
    REQUIRE( app.m_fpsConfirmation.phase == Phase::confirmed );
}

/// Settings, thermal changes, and invalid lifecycle states cannot enter the diagnostic quiet interval.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl rejects unsafe housekeeping pauses", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::beginConfirmation();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );

    SECTION( "gain acknowledgement pending" )
    {
        app.beginConfirmation( app.m_gainConfirmation, 77, true );
    }

    SECTION( "initial observation has not started" )
    {
        app.m_gainConfirmation.phase = Phase::unavailable;
    }

    SECTION( "FPS acknowledgement pending" )
    {
        app.beginConfirmation( app.m_fpsConfirmation, 1000, true );
    }

    SECTION( "temperature acknowledgement pending" )
    {
        app.beginConfirmation( app.m_tempConfirmation, -40, true );
    }

    SECTION( "thermal change pending" )
    {
        app.m_tempChanging = true;
    }

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

    SECTION( "shutdown requested" )
    {
        app.m_shutdown = true;
    }

    REQUIRE( app.setHousekeepingPause( true, app.m_now ) < 0 );
    REQUIRE( app.m_housekeepingPauseUntil == 0 );
    REQUIRE( g_edtStubState.serialCommands.empty() );
}

/// Terminal failures have no pending fast work and do not prevent an otherwise idle diagnostic pause.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl permits a housekeeping pause after terminal confirmation failure", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setHousekeepingPause(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_gainConfirmation.phase = Phase::failed;
    app.m_fpsConfirmation.phase  = Phase::failed;
    app.m_tempConfirmation.phase = Phase::failed;
    REQUIRE( app.setHousekeepingPause( true, app.m_now ) == 0 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.housekeepingPaused( app.m_now ) );
    REQUIRE( app.m_gainConfirmation.phase == Phase::failed );
    REQUIRE( app.m_fpsConfirmation.phase == Phase::failed );
    REQUIRE( app.m_tempConfirmation.phase == Phase::failed );
    REQUIRE( g_edtStubState.serialCommands.empty() );
}

/// Automatic expiry resumes a single temperature read, and explicit settings cancel an active pause.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl ends housekeeping pause at its actual boundary", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::housekeepingPaused(); ocam2KCtrl::endHousekeepingPause(); ocam2KCtrl::setTempSetPt();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    REQUIRE( app.setHousekeepingPause( true, app.m_now ) == 0 );

    SECTION( "hard deadline" )
    {
        app.m_now = app.m_housekeepingPauseUntil;
        queueTemperature();
        REQUIRE( app.pollHousekeeping() == 0 );
        REQUIRE( app.m_housekeepingPauseUntil == 0 );
        REQUIRE( app.m_housekeepingPauseEnded == app.m_now );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "explicit setting" )
    {
        app.m_now += 1000000000;
        app.m_ccdTempSetpt = -40;
        queueSerialResponse( "setting accepted\n" );
        REQUIRE( app.setTempSetPt() == 0 );
        REQUIRE( app.m_housekeepingPauseUntil == 0 );
        REQUIRE( app.m_housekeepingPauseEnded == app.m_now );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp -40.000000" } );
    }

    SECTION( "lifecycle change" )
    {
        app.m_now += 1000000000;
        app.m_reconfig = true;
        REQUIRE_FALSE( app.housekeepingPaused( app.m_now ) );
        REQUIRE( app.m_housekeepingPauseEnded == app.m_now );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }
}

/// Pause activation waits for serial ownership before establishing its monotonic boundaries.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl serializes housekeeping pause activation", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::newCallBack_m_indiP_housekeepingPause(); ocam2KCtrl::setHousekeepingPause();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<housekeepingClock> app;
    app.driver();
    prepareSettledHousekeeping( app );
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
    const auto waiting = pause.wait_for( std::chrono::milliseconds( 20 ) );
    transaction.unlock();
    const int result = pause.get();
    REQUIRE( waiting == std::future_status::timeout );
    REQUIRE( result == 0 );
    REQUIRE( app.m_housekeepingPauseStarted == app.m_now );
    REQUIRE( app.m_housekeepingPauseUntil == app.m_now + 120000000000 );
}

/// Diagnostic pause still services the independent ten-second protection-reset confirmation expiry.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl housekeeping pause preserves protection request expiry", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_protectionResetConfirmed = 1;
    app.m_protectionResetReqTime   = mx::sys::get_curr_time() - 11;
    REQUIRE( app.setHousekeepingPause( true, app.m_now ) == 0 );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( app.m_protectionResetConfirmed == 0 );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.housekeepingPaused( app.m_now ) );
}

/// Cached confirmations stay confirmed as they age, and the operator sees confirmation-only protection polling.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl reports cached housekeeping confirmation separately from age", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::initHousekeeping(); ocam2KCtrl::updateHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<housekeepingClock> app;
    app.driver();
    prepareSettledHousekeeping( app );
    REQUIRE( app.initHousekeeping() == 0 );
    app.m_now += 60000000000;
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeeping.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_housekeepingStatus.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_housekeeping["gainAgeSec"].get<double>() == 60 );
    REQUIRE( app.m_indiP_housekeeping["fpsAgeSec"].get<double>() == 60 );
    REQUIRE( app.m_indiP_housekeepingStatus["gain"].get<std::string>() == "confirmed" );
    REQUIRE( app.m_indiP_housekeepingStatus["fps"].get<std::string>() == "confirmed" );
    REQUIRE( app.m_indiP_housekeepingStatus["protectionPolling"].get() == "confirmation only" );
    REQUIRE( app.m_indiP_housekeepingStatus["temperature"].get<std::string>() == "stale" );
    REQUIRE( app.newCallBack_m_indiP_housekeepingPause( pauseRequest( app, true ) ) == 0 );
    app.updateHousekeeping();
    REQUIRE( app.m_indiP_housekeepingStatus["state"].get<std::string>() == "paused" );
    REQUIRE( app.m_indiP_housekeepingStatus["protectionPolling"].get<std::string>() == "paused" );
    REQUIRE( app.m_indiP_housekeepingStatus["gain"].get<std::string>() == "confirmed" );
}

/// Invalid configuration is rejected before it can produce busy loops or an unbounded acknowledgement window.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl validates housekeeping configuration", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setupConfig(); ocam2KCtrl::loadConfigImpl();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    app.setupConfig();
    const std::string path   = uniqueConfigPath( "housekeeping" );
    const auto [name, value] = GENERATE( std::pair<std::string, std::string>{ "tempIntervalSec", "0" },
                                         std::pair<std::string, std::string>{ "tempIntervalSec", "nan" },
                                         std::pair<std::string, std::string>{ "tempChangingIntervalSec", "11" },
                                         std::pair<std::string, std::string>{ "confirmationIntervalSec", "-1" },
                                         std::pair<std::string, std::string>{ "confirmationTimeoutSec", "0.5" },
                                         std::pair<std::string, std::string>{ "tempSettleSamples", "0" },
                                         std::pair<std::string, std::string>{ "tempSettleSamples", "-1" },
                                         std::pair<std::string, std::string>{ "tempSettleSamples", "2.5" } );
    mx::app::writeConfigFile( path, { "housekeeping" }, { name }, { value } );
    REQUIRE( app.config.readConfig( path ) == 0 );
    REQUIRE( app.loadConfigImpl() < 0 );
    std::remove( path.c_str() );
}

/// Recovery invalidates cached confirmations and source generations without automatically reapplying gain.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl resets housekeeping observations for lifecycle recovery", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::resetHousekeeping(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );
    app.m_syncFreqTime            = app.m_now;
    const uint64_t gainGeneration = app.m_gainConfirmation.generation;
    const uint64_t fpsGeneration  = app.m_fpsConfirmation.generation;
    const uint64_t tempGeneration = app.m_tempConfirmation.generation;
    app.resetHousekeeping();
    REQUIRE( app.m_gainConfirmation.phase == Phase::unavailable );
    REQUIRE( app.m_fpsConfirmation.phase == Phase::unavailable );
    REQUIRE( app.m_tempConfirmation.phase == Phase::unavailable );
    REQUIRE( app.m_gainConfirmation.generation > gainGeneration );
    REQUIRE( app.m_fpsConfirmation.generation > fpsGeneration );
    REQUIRE( app.m_tempConfirmation.generation > tempGeneration );
    REQUIRE( app.m_housekeepingGainTime == 0 );
    REQUIRE( app.m_housekeepingFpsTime == 0 );
    REQUIRE( app.m_housekeepingTempTime == 0 );
    REQUIRE( app.m_syncFreqTime == 0 );
    REQUIRE( app.m_tempChanging );
    REQUIRE_FALSE( app.m_coolingKnown );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    queueSerialResponse( "Gain set to 1 \n\n" );
    REQUIRE( app.pollHousekeeping() == 0 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain" } );
    REQUIRE( app.m_emGain == 1 );
}

/// Actual HV trip replies and transport failures keep their existing error-recovery behavior.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl preserves scheduled housekeeping failures", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getEMGain(); ocam2KCtrl::pollHousekeeping();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    prepareSettledHousekeeping( app );

    SECTION( "temperature transport error" )
    {
        app.m_tempDue = app.m_now;
        queueSerialResponse( "", -1 );
        REQUIRE( app.pollHousekeeping() < 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp" } );
    }

    SECTION( "gain HV response" )
    {
        app.beginConfirmation( app.m_gainConfirmation, 77, true );
        queueSerialResponse( "HV trip\n" );
        REQUIRE( app.pollHousekeeping() < 0 );
        REQUIRE( app.m_emGain == 1 );
        REQUIRE( app.m_housekeepingGainTime == 0 );
        REQUIRE( app.state() == stateCodes::ERROR );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain" } );
    }
}

/// No-due-work and paused control passes continue external FPS bookkeeping and telemetry progress.
/** \ingroup ocam2KCtrl_housekeeping_unit_test */
TEST_CASE( "ocam2KCtrl app logic progresses without serial housekeeping", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::appLogic(); ocam2KCtrl::pollHousekeeping(); ocam2KCtrl::recordCamera();
    #endif
    // clang-format on
    resetStubState();
    housekeepingClock app;
    startupScope      startup( app );
    configureStartupMode( app );
    REQUIRE( app.dev::telemeter<ocam2KCtrl>::appStartup() == 0 );
    startup.markStarted( true );
    prepareSettledHousekeeping( app );
    app.m_synchro     = true;
    app.m_syncFreq    = 2000;
    app.m_fps         = 1900;
    app.m_maxInterval = 10;

    SECTION( "no serial work is due" )
    {
    }

    SECTION( "diagnostic pause is active" )
    {
        REQUIRE( app.setHousekeepingPause( true, app.m_now ) == 0 );
    }

    MagAOX::logger::telem_stdcam::lastRecord = { 0, 0 };
    fgThreadScope fgThread( app );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::OPERATING );
    REQUIRE_FALSE( app.m_shutdown );
    REQUIRE( app.m_fps == 2000 );
    REQUIRE( app.m_tel.logThreadRunning() );
    REQUIRE( hasRecordedTime( MagAOX::logger::telem_stdcam::lastRecord ) );
    REQUIRE( g_edtStubState.serialCommands.empty() );
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
