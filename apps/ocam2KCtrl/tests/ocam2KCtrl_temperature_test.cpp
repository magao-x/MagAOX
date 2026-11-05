/** \file ocam2KCtrl_temperature_test.cpp
 * \brief Offline tests of OCAM temperature settings and fresh thermal evidence for EM-gain requests.
 * \ingroup ocam2KCtrl_temperature_unit_test
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
/** \addtogroup ocam2KCtrl_temperature_unit_test */
namespace ocam2KCtrlTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace
{

/// Initialize the shared temperature interlock and its offline INDI properties.
void prepareGainGuard( ocam2KCtrl_test   &app /**< [in,out] Controller whose gain guard is initialized. */,
                       const std::string &mode /**< [in] Temperature criterion to enforce. */ )
{
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_emGainRequiresTemp = true;
    app.m_emGainTempMode     = mode;
    app.m_emGainTemp         = -50;
    app.m_emGainMaxTemp      = -45;
    app.m_ccdTempSetpt       = -50;
    app.m_protectionReset    = true;
    app.m_maxEMGain          = 600;
    app.createStandardIndiNumber<float>( app.m_indiP_emGain, "emgain", 0, 600, 1, "%0.0f" );
    app.createStandardIndiToggleSw( app.m_indiP_emGainTempReady, "emgain_temp_ready" );
    app.createStandardIndiToggleSw( app.m_indiP_synchro, "synchro" );
    app.m_indiP_temps = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_temps.setDevice( app.m_configName );
    app.m_indiP_temps.setName( "temps" );

    for( const std::string name : { "cpu", "power", "bias", "water", "left", "right", "cooling" } )
    {
        app.m_indiP_temps.add( pcf::IndiElement( name ) );
    }
}

/// Submit amplified gain through the normal shared callback.
int requestAmplifiedGain( ocam2KCtrl_test &app /**< [in,out] Controller receiving the gain request. */ )
{
    pcf::IndiProperty request( pcf::IndiProperty::Number, app.m_indiP_emGain.getDevice(), "emgain" );
    request.add( pcf::IndiElement( "target" ) );
    request["target"].set( 100 );
    return app.newCallBack_stdCamera( request );
}

/// Queue a complete serial temperature response with independent measured and applied thermal state.
void queueGainGuardTemperature( float temperature /**< [in] Measured detector temperature, in C. */,
                                float setpoint     = -50 /**< [in] Applied detector setpoint, in C. */,
                                float coolingPower = 102 /**< [in] Measured cooling power, in mW. */ )
{
    queueSerialResponse( "Temperatures : CCD[" + std::to_string( temperature ) +
                         "] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] SET[" +
                         std::to_string( setpoint * 10 ) + "]\nCooling Power [" + std::to_string( coolingPower ) +
                         "]mW.\n\n" );
}

} // namespace
/// \endcond

/// Shared temperature settings are bounded by OCAM capabilities and initialize each power-on target.
/** \ingroup ocam2KCtrl_temperature_unit_test */
TEST_CASE( "ocam2KCtrl uses shared temperature configuration", "[ocam2KCtrl]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::loadConfigImpl(); dev::stdCamera<ocam2KCtrl>::setTempLimits();
    dev::stdCamera<ocam2KCtrl>::powerOnTemp();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    app.setupConfig();
    REQUIRE( app.m_minTemp == -50 );
    REQUIRE( app.m_maxTemp == 20 );
    REQUIRE( app.m_startupTemp == -45 );
    const std::string path = uniqueConfigPath( "temperatures" );

    SECTION( "default startup target" )
    {
        REQUIRE( app.loadConfigImpl() == 0 );

        for( int cycle = 0; cycle < 2; ++cycle )
        {
            app.m_ccdTempSetpt = 10;
            REQUIRE( app.powerOnDefaults() == 0 );
            REQUIRE( app.powerOnTemp() == 0 );
            REQUIRE( app.m_ccdTempSetpt == -45 );
        }
    }

    SECTION( "configured startup and user bounds" )
    {
        mx::app::writeConfigFile(
            path, { "camera", "camera", "camera" }, { "minTemp", "maxTemp", "startupTemp" }, { "-40", "15", "-30" } );
        REQUIRE( app.config.readConfig( path ) == 0 );
        REQUIRE( app.loadConfigImpl() == 0 );
        REQUIRE( app.m_minTemp == -40 );
        REQUIRE( app.m_maxTemp == 15 );

        for( int cycle = 0; cycle < 2; ++cycle )
        {
            app.m_ccdTempSetpt = 10;
            REQUIRE( app.powerOnDefaults() == 0 );
            REQUIRE( app.powerOnTemp() == 0 );
            REQUIRE( app.m_ccdTempSetpt == -30 );
        }
    }

    SECTION( "hardware bounds narrow wider user settings" )
    {
        mx::app::writeConfigFile( path, { "camera", "camera" }, { "minTemp", "maxTemp" }, { "-60", "35" } );
        REQUIRE( app.config.readConfig( path ) == 0 );
        REQUIRE( app.loadConfigImpl() == 0 );
        REQUIRE( app.m_minTemp == -50 );
        REQUIRE( app.m_maxTemp < 30 );
        setPoweredOn( app );
        app.m_ccdTempSetpt = 30;
        REQUIRE( app.setTempSetPt() == -1 );
        REQUIRE( g_edtStubState.serialCommands.empty() );
    }

    SECTION( "invalid power-on target" )
    {
        mx::app::writeConfigFile( path, { "camera" }, { "startupTemp" }, { "21" } );
        REQUIRE( app.config.readConfig( path ) == 0 );
        app.loadConfig();
        REQUIRE( app.m_shutdown );
    }

    SECTION( "disabled startup override remains supported" )
    {
        mx::app::writeConfigFile( path, { "camera" }, { "startupTemp" }, { "-999" } );
        REQUIRE( app.config.readConfig( path ) == 0 );
        REQUIRE( app.loadConfigImpl() == 0 );
        app.m_ccdTempSetpt = 5;
        REQUIRE( app.powerOnTemp() == 0 );
        REQUIRE( app.m_ccdTempSetpt == 5 );
    }

    std::remove( path.c_str() );
}

/// Effective endpoints are inclusive and nonfinite targets do not cause serial traffic.
/** \ingroup ocam2KCtrl_temperature_unit_test */
TEST_CASE( "ocam2KCtrl validates shared temperature endpoints", "[ocam2KCtrl]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempSetPt(); dev::stdCamera<ocam2KCtrl>::validateTempSetPt();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    setPoweredOn( app );
    SECTION( "inclusive effective endpoints are accepted" )
    {
        for( float target : { -50.0f, 20.0f } )
        {
            app.m_ccdTempSetpt = target;
            queueSerialResponse( "setpoint updated\n" );
            REQUIRE( app.setTempSetPt() == 0 );
        }

        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp -50.000000", "temp 20.000000" } );
    }

    SECTION( "nonfinite targets do not cause serial traffic" )
    {
        for( float target : { std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity() } )
        {
            app.m_ccdTempSetpt = target;
            REQUIRE( app.setTempSetPt() == -1 );
        }

        REQUIRE( g_edtStubState.serialCommands.empty() );
    }
}

/// Shared callbacks preserve the accepted OCAM target on range rejection or serial failure.
/** \ingroup ocam2KCtrl_temperature_unit_test */
TEST_CASE( "ocam2KCtrl restores rejected temperature requests", "[ocam2KCtrl]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setTempSetPt(); dev::stdCamera<ocam2KCtrl>::newCallBack_temp();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.createStandardIndiNumber<float>( app.m_indiP_temp, "temp_ccd", app.m_minTemp, app.m_maxTemp, 0, "%0.1f" );
    app.m_ccdTempSetpt = 5;
    app.m_indiP_temp["target"].set( 5 );
    auto request = [&]( float target /**< [in] Requested temperature, in C. */ )
    {
        pcf::IndiProperty received(
            pcf::IndiProperty::Number, app.m_indiP_temp.getDevice(), app.m_indiP_temp.getName() );
        received.add( pcf::IndiElement( "target" ) );
        received["target"].set( target );
        return app.newCallBack_temp( received );
    };

    REQUIRE( request( 21 ) == -1 );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( app.m_ccdTempSetpt == 5 );
    REQUIRE( app.m_indiP_temp["target"].get<float>() == 5 );
    queueSerialResponse( "", -1 );
    REQUIRE( request( -30 ) == -1 );
    REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "temp -30.000000" } );
    REQUIRE( app.m_ccdTempSetpt == 5 );
    REQUIRE( app.m_indiP_temp["target"].get<float>() == 5 );
    REQUIRE( app.m_indiP_temp.getState() == INDI_ALERT );
    REQUIRE( app.state() == stateCodes::OPERATING );
}

/// Maximum-temperature protection uses a complete fresh serial reading without changing OCAM lock status.
/** \ingroup ocam2KCtrl_temperature_unit_test */
TEST_CASE( "ocam2KCtrl maximum-temperature gain guard requires fresh serial readings", "[ocam2KCtrl]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getTemps(); ocam2KCtrl::configureAcquisition(); ocam2KCtrl::powerOnDefaults();
    ocam2KCtrl::setTempSetPt(); dev::stdCamera<ocam2KCtrl>::newCallBack_emgain();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    prepareGainGuard( app, "maxTemp" );
    auto requestGain = [&]()
    {
        return requestAmplifiedGain( app );
    };
    auto poll = [&]( float temperature /**< [in] Scripted measured temperature, in C. */ )
    {
        queueGainGuardTemperature( temperature );
        return app.getTemps();
    };

    REQUIRE( requestGain() == -1 );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE( poll( -46 ) == 0 );
    REQUIRE( app.m_ccdTempValid );
    REQUIRE_FALSE( app.m_tempControlOnTarget );
    REQUIRE( app.m_tempControlStatusStr == "OFF TARGET" );
    queueSerialResponse( "gain set\n" );
    REQUIRE( requestGain() == 0 );
    REQUIRE( g_edtStubState.serialCommands.back() == "gain 100" );

    SECTION( "the ceiling is inclusive and warmer readings reject gain" )
    {
        REQUIRE( poll( -45 ) == 0 );
        queueSerialResponse( "gain set\n" );
        REQUIRE( requestGain() == 0 );
        REQUIRE( poll( -44.9 ) == 0 );
        REQUIRE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( g_edtStubState.serialCommands.back() == "temp" );
    }

    SECTION( "malformed or failed polls and sentinel readings invalidate old cold measurements" )
    {
        queueSerialResponse( "Temperatures : CCD[-46]\n" );
        REQUIRE( app.getTemps() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        REQUIRE( app.m_ccdTempValid );
        queueSerialResponse( "", -1 );
        REQUIRE( app.getTemps() == -1 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -999 ) == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
    }

    SECTION( "power defaults and temperature commands invalidate previous readings" )
    {
        REQUIRE( app.powerOnDefaults() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        REQUIRE( app.m_ccdTempValid );
        queueSerialResponse( "target set\n" );
        REQUIRE( app.setTempSetPt() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        queueSerialResponse( "gain set\n" );
        REQUIRE( requestGain() == 0 );
    }

    SECTION( "a complete setpoint acknowledgement still requires a subsequent temperature poll" )
    {
        queueGainGuardTemperature( -46 );
        REQUIRE( app.setTempSetPt() == 0 );
        REQUIRE( app.m_tempConfirmation.phase == ocam2KCtrl::ConfirmationPhase::confirmed );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        queueSerialResponse( "Gain set to 100\n" );
        REQUIRE( requestGain() == 0 );
    }

    SECTION( "complete cooling acknowledgements still require a subsequent temperature poll" )
    {
        app.m_tempControlStatusSet = true;
        queueGainGuardTemperature( -46 );
        queueGainGuardTemperature( -46 );
        REQUIRE( app.setTempControl() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        queueSerialResponse( "Gain set to 100\n" );
        REQUIRE( requestGain() == 0 );
    }

    SECTION( "acquisition configuration requires a subsequent temperature poll" )
    {
        configureStartupMode( app );
        app.m_raw_height         = 121;
        app.m_ocamDescrambleFile = "ocam_descramble_stub.txt";
        app.m_fpsSet             = 0;
        app.m_synchro            = false;
        app.m_synchroSet         = false;
        queueSerialResponse( "mode selected\n" );
        queueSerialResponse( "fps [2000] Hz\n" );
        queueSerialResponse( "synchro off\n" );
        queueSerialResponse( "fps [2000] Hz\n" );
        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        queueSerialResponse( "gain set\n" );
        REQUIRE( requestGain() == 0 );
    }

    SECTION( "acquisition recovery invalidates cached cold temperature" )
    {
        app.requestAcquisitionRecovery();
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( poll( -46 ) == 0 );
        queueSerialResponse( "Gain set to 100\n" );
        REQUIRE( requestGain() == 0 );
    }
}

/// Preserved requested targets and cached cooling state cannot substitute for fresh applied thermal evidence.
/** \ingroup ocam2KCtrl_temperature_unit_test */
TEST_CASE( "ocam2KCtrl gain guard rejects unconfirmed thermal observations", "[ocam2KCtrl][housekeeping]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::getTemps(); ocam2KCtrl::acceptTemps(); ocam2KCtrl::setTempSetPt();
    ocam2KCtrl::setTempControl(); ocam2KCtrl::requestAcquisitionRecovery(); ocam2KCtrl::resetHousekeeping();
    dev::stdCamera<ocam2KCtrl>::newCallBack_emgain(); dev::stdCamera<ocam2KCtrl>::emGainTempReady();
    #endif
    // clang-format on
    const std::string mode = GENERATE( "locked", "maxTemp" );
    CAPTURE( mode );
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    prepareGainGuard( app, mode );
    queueGainGuardTemperature( -50 );
    REQUIRE( app.getTemps() == 0 );
    REQUIRE( app.emGainTempReady() );
    REQUIRE( app.m_ccdTempValid );

    SECTION( "an outstanding target cannot reuse an old on-target readback" )
    {
        queueSerialResponse( "setting accepted\n" );
        REQUIRE( app.setTempSetPt() == 0 );
        queueGainGuardTemperature( -46, -46 );
        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_ccdTempSetpt == -50 );
        REQUIRE( app.m_temps.SET == -46 );
        REQUIRE( app.m_tempConfirmation.phase == ocam2KCtrl::ConfirmationPhase::confirming );
    }

    SECTION( "a failed target cannot reuse an old on-target readback" )
    {
        queueSerialResponse( "setting accepted\n" );
        REQUIRE( app.setTempSetPt() == 0 );
        app.m_tempConfirmation.deadline = app.housekeepingNow() - 1;
        queueGainGuardTemperature( -46, -46 );
        REQUIRE( app.getTemps() == 0 );
        REQUIRE( app.m_ccdTempSetpt == -50 );
        REQUIRE( app.m_tempConfirmation.phase == ocam2KCtrl::ConfirmationPhase::failed );
    }

    SECTION( "ambiguous cooling cannot reuse an earlier active status" )
    {
        queueGainGuardTemperature( -50, -50, 2 );
        REQUIRE( app.getTemps() == 0 );
        REQUIRE_FALSE( app.m_coolingKnown );
        REQUIRE( app.m_tempControlStatus );
    }

    SECTION( "a refused cooling-Off preflight cannot apply fresh evidence to the restored target" )
    {
        app.m_tempExpectedSet      = false;
        app.m_tempControlStatusSet = false;
        queueGainGuardTemperature( -46, -46 );
        REQUIRE( app.setTempControl() < 0 );
        REQUIRE( app.m_ccdTempSetpt == -50 );
        REQUIRE( app.m_temps.SET == -46 );
        REQUIRE_FALSE( app.m_tempControlStatusSet );
    }

    SECTION( "acquisition recovery cannot retain a previous temperature lock" )
    {
        app.requestAcquisitionRecovery();
        REQUIRE_FALSE( app.m_coolingKnown );
    }

    REQUIRE_FALSE( app.m_ccdTempValid );
    REQUIRE_FALSE( app.m_tempControlOnTarget );
    REQUIRE_FALSE( app.emGainTempReady() );
    REQUIRE( app.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( requestAmplifiedGain( app ) == -1 );
    REQUIRE( g_edtStubState.serialCommands.back() == "temp" );
    const bool failed = app.m_tempConfirmation.phase == ocam2KCtrl::ConfirmationPhase::failed;

    queueGainGuardTemperature( -50 );
    REQUIRE( app.getTemps() == 0 );
    REQUIRE( app.m_ccdTempValid );
    REQUIRE( app.m_tempControlOnTarget );
    REQUIRE( app.emGainTempReady() );
    REQUIRE( app.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::On );

    if( failed )
    {
        REQUIRE( app.m_tempConfirmation.phase == ocam2KCtrl::ConfirmationPhase::failed );
    }

    queueSerialResponse( "Gain set to 100\n" );
    REQUIRE( requestAmplifiedGain( app ) == 0 );
    REQUIRE( g_edtStubState.serialCommands.back() == "gain 100" );
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
