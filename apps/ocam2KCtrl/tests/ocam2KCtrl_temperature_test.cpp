/** \file ocam2KCtrl_temperature_test.cpp
 * \brief Offline tests of shared OCAM temperature settings, power-on targets, and request rollback.
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
} // namespace ocam2KCtrlTest
} // namespace libXWCTest
