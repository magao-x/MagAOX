/** \file stdMotionStage_test.cpp
 * \brief Catch2 tests for the stdMotionStage helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup testing
 */

#include "../../../../tests/testXWC.hpp"

#include "stdMotionStage_test.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{
namespace appTest
{
namespace devTest
{

/** \defgroup stdMotionStage_tests libXWC::app::dev::stdMotionStage Unit Tests
 * \ingroup app_dev_unit_tests
 */

/// Verify stdMotionStage logs and rejects invalid preset-name selections before issuing motion requests.
/**
 * \ingroup stdMotionStage_tests
 */
TEST_CASE( "stdMotionStage rejects invalid preset-name selections", "[dev::stdMotionStage]" )
{
    // clang-format off
    #ifdef STDMOTIONSTAGE_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdMotionStage<libXWCTest::appTest::devTest::stdMotionStageHarness>::newCallBack_m_indiP_presetName( pcf::IndiProperty() );
    #endif
    // clang-format on

    SECTION( "quoted preset names are logged and rejected" )
    {
        stdMotionStageHarness app;

        app.configurePresets( { "open", "focus" }, "preset" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest( { { "\"focus\"", pcf::IndiElement::On } } ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_ERROR );
        REQUIRE( stdMotionStageHarness::lastLogMessage() == "Unknown presetName selected: \"focus\"" );
        REQUIRE( app.moveCalls() == 0 );
        REQUIRE( app.lastMoveTarget() == -1.0f );
    }

    SECTION( "invalid names are rejected even when a valid preset is also selected" )
    {
        stdMotionStageHarness app;

        app.configurePresets( { "open", "focus" }, "filter" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest(
                     { { "open", pcf::IndiElement::On }, { "bogus", pcf::IndiElement::On } } ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_ERROR );
        REQUIRE( stdMotionStageHarness::lastLogMessage() == "Unknown filterName selected: bogus" );
        REQUIRE( app.moveCalls() == 0 );
        REQUIRE( app.lastMoveTarget() == -1.0f );
    }

    SECTION( "multiple invalid names are joined in the log message" )
    {
        stdMotionStageHarness app;

        app.configurePresets( { "open", "focus" }, "preset" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest(
                     { { "bogus1", pcf::IndiElement::On }, { "bogus2", pcf::IndiElement::On } } ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_ERROR );
        REQUIRE( stdMotionStageHarness::lastLogMessage() == "Unknown presetName selected: bogus1, bogus2" );
        REQUIRE( app.moveCalls() == 0 );
        REQUIRE( app.lastMoveTarget() == -1.0f );
    }
}

/// Verify that setupConfig() and loadConfig() handle omitted positions, configured values, and the defaultPositions flag.
/**
 * \ingroup stdMotionStage_tests
 *
 * The config files are written under /tmp and read back through mx::app::appConfigurator.
 */
TEST_CASE( "stdMotionStage setupConfig and loadConfig", "[dev::stdMotionStage]" )
{
    SECTION( "setupConfig registers the expected options" )
    {
        stdMotionStageHarness app;
        mx::app::appConfigurator config;

        REQUIRE( app.setupConfig( config ) == 0 );
    }

    SECTION( "loadConfig fills default positions from preset order when positions are omitted" )
    {
        mx::app::writeConfigFile( "/tmp/stdMotionStage_test.conf",
                                  { "presets" },
                                  { "names" },
                                  { "open,focus,block" } );

        mx::app::appConfigurator config;

        stdMotionStageHarness app;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdMotionStage_test.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );

        REQUIRE( app.presetNamesValue().size() == 3 );
        REQUIRE( app.presetPositionsValue().size() == 3 );
        REQUIRE( app.presetPositionsValue()[0] == 1 );
        REQUIRE( app.presetPositionsValue()[1] == 2 );
        REQUIRE( app.presetPositionsValue()[2] == 3 );
        REQUIRE( app.powerOnHomeValue() == false );
        REQUIRE( app.homePresetValue() == -1 );
    }

    SECTION( "loadConfig repairs zero-valued positions and keeps configured non-zero positions" )
    {
        mx::app::writeConfigFile( "/tmp/stdMotionStage_test.conf",
                                  { "stage", "stage", "presets", "presets" },
                                  { "powerOnHome", "homePreset", "names", "positions" },
                                  { "true", "2", "open,focus,block", "0,5,0" } );

        mx::app::appConfigurator config;

        stdMotionStageHarness app;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdMotionStage_test.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );

        REQUIRE( app.powerOnHomeValue() == true );
        REQUIRE( app.homePresetValue() == 2 );
        REQUIRE( app.presetPositionsValue().size() == 3 );
        REQUIRE( app.presetPositionsValue()[0] == 1 );
        REQUIRE( app.presetPositionsValue()[1] == 5 );
        REQUIRE( app.presetPositionsValue()[2] == 3 );
    }

    SECTION( "loadConfig leaves positions alone when defaultPositions is disabled" )
    {
        mx::app::writeConfigFile( "/tmp/stdMotionStage_test.conf",
                                  { "presets", "presets" },
                                  { "names", "positions" },
                                  { "open,focus,block", "0,5,0" } );

        mx::app::appConfigurator config;

        stdMotionStageHarness app;
        app.setDefaultPositions( false );

        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdMotionStage_test.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );

        // Without the default fill and repair logic the raw configured values pass through, including the zeros.
        REQUIRE( app.presetPositionsValue().size() == 3 );
        REQUIRE( app.presetPositionsValue()[0] == 0 );
        REQUIRE( app.presetPositionsValue()[1] == 5 );
        REQUIRE( app.presetPositionsValue()[2] == 0 );
    }
}

/// Verify that stdMotionStage::appStartup() builds and registers the INDI properties and reports each failure.
/**
 * \ingroup stdMotionStage_tests
 *
 * Registration failures are injected one property at a time through the harness
 * override of registerIndiPropertyNew().
 */
TEST_CASE( "stdMotionStage appStartup", "[dev::stdMotionStage]" )
{
    SECTION( "appStartup succeeds with fractional presets and builds property elements" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus", "block" }, "preset" );

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );
        REQUIRE( app.presetPropertyName() == "preset" );
        REQUIRE( app.presetNamePropertyName() == "presetName" );
        REQUIRE( app.homePropertyName() == "home" );
        REQUIRE( app.stopPropertyName() == "stop" );
    }

    SECTION( "appStartup succeeds with non-fractional (integer step) presets" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus", "block" }, "preset" );
        app.setFractionalPresets( false );

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );
    }

    SECTION( "appStartup fails when there are no configured preset names" )
    {
        stdMotionStageHarness app;
        app.configurePresets( {}, "preset" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
    }

    SECTION( "appStartup fails when the preset property fails to register" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        app.setFailRegisterName( "preset" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
    }

    SECTION( "appStartup fails when the presetName property fails to register" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        app.setFailRegisterName( "presetName" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
    }

    SECTION( "appStartup fails when the home property fails to register" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        app.setFailRegisterName( "home" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
    }

    SECTION( "appStartup fails when the stop property fails to register" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        app.setFailRegisterName( "stop" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
    }
}

/// Verify the trivial lifecycle functions of stdMotionStage.
/**
 * \ingroup stdMotionStage_tests
 *
 * appLogic(), whilePowerOff(), and appShutdown() return 0. onPowerOff() marks the
 * stage as not moving by setting m_moving to -2.
 */
TEST_CASE( "stdMotionStage appLogic, onPowerOff, whilePowerOff, and appShutdown", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;

    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appLogic() == 0 );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::whilePowerOff() == 0 );

    app.setMoving( 1 );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::onPowerOff() == 0 );
    REQUIRE( app.movingValue() == -2 );

    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appShutdown() == 0 );
}

/// Verify the static st_newCallBack_stdMotionStage dispatcher routes to the correct handler by property name.
/**
 * \ingroup stdMotionStage_tests
 */
TEST_CASE( "stdMotionStage static callback dispatcher routes by property name", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;
    app.configurePresets( { "open", "focus" }, "preset" );
    app.setPresetPositions( { 1, 2 } );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

    SECTION( "stop requests are routed to newCallBack_m_indiP_stop" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "stest" );
        ip.setName( "stop" );
        ip.add( pcf::IndiElement( "request" ) );
        ip["request"].setSwitchState( pcf::IndiElement::On );

        REQUIRE( dev::stdMotionStage<stdMotionStageHarness>::st_newCallBack_stdMotionStage( &app, ip ) == 0 );
        REQUIRE( app.stopCalls() == 1 );
    }

    SECTION( "home requests are routed to newCallBack_m_indiP_home" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "stest" );
        ip.setName( "home" );
        ip.add( pcf::IndiElement( "request" ) );
        ip["request"].setSwitchState( pcf::IndiElement::On );

        REQUIRE( dev::stdMotionStage<stdMotionStageHarness>::st_newCallBack_stdMotionStage( &app, ip ) == 0 );
        REQUIRE( app.homingCalls() == 1 );
    }

    SECTION( "preset requests are routed to newCallBack_m_indiP_preset" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "stest" );
        ip.setName( "preset" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 2.0f );

        REQUIRE( dev::stdMotionStage<stdMotionStageHarness>::st_newCallBack_stdMotionStage( &app, ip ) == 0 );
        REQUIRE( app.moveCalls() == 1 );
    }

    SECTION( "presetName requests are routed to newCallBack_m_indiP_presetName" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "stest" );
        ip.setName( "presetName" );
        ip.add( pcf::IndiElement( "open" ) );
        ip["open"].setSwitchState( pcf::IndiElement::On );

        REQUIRE( dev::stdMotionStage<stdMotionStageHarness>::st_newCallBack_stdMotionStage( &app, ip ) == 0 );
        REQUIRE( app.moveCalls() == 1 );
    }

    SECTION( "unrecognized property names are rejected" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "stest" );
        ip.setName( "somethingElse" );

        REQUIRE( dev::stdMotionStage<stdMotionStageHarness>::st_newCallBack_stdMotionStage( &app, ip ) == -1 );
    }
}

/// Verify the numeric preset position callback of stdMotionStage.
/**
 * \ingroup stdMotionStage_tests
 *
 * A mismatched property and a missing target element are rejected. A valid target
 * triggers one moveTo() call and records the target.
 */
TEST_CASE( "stdMotionStage numerical preset callback", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;
    app.configurePresets( { "open", "focus", "block" }, "preset" );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

    SECTION( "mismatched property is rejected" )
    {
        REQUIRE( app.applyPresetRequest( 2.0f, true, false ) == -1 );
        REQUIRE( app.moveCalls() == 0 );
    }

    SECTION( "missing target/current element is rejected and logged" )
    {
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetRequest( 2.0f, false, true ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( app.moveCalls() == 0 );
    }

    SECTION( "a valid target position is accepted" )
    {
        app.setMoving( 1 );

        REQUIRE( app.applyPresetRequest( 2.0f ) == 0 );
        REQUIRE( app.moveCalls() == 1 );
        REQUIRE( app.lastMoveTarget() == 2.0f );
        REQUIRE( app.movingStateValue() == 0 );
        REQUIRE( app.presetNameIndexValue() == -1 );
        REQUIRE( app.presetTargetValue() == 2.0f );
    }
}

/// Verify the named preset callback of stdMotionStage across its accept, reject, and no-op paths.
/**
 * \ingroup stdMotionStage_tests
 */
TEST_CASE( "stdMotionStage preset-name callback", "[dev::stdMotionStage]" )
{
    SECTION( "mismatched property is rejected" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "wrongDevice" );
        ip.setName( "presetName" );

        REQUIRE( app.newCallBack_m_indiP_presetName( ip ) == -1 );
    }

    SECTION( "no selection on is a no-op" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );

        REQUIRE( app.applyPresetNameRequest( { { "open", pcf::IndiElement::Off } } ) == 0 );
        REQUIRE( app.moveCalls() == 0 );
    }

    SECTION( "more than one selection is rejected and logged" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest(
                     { { "open", pcf::IndiElement::On }, { "focus", pcf::IndiElement::On } } ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_ERROR );
        REQUIRE( stdMotionStageHarness::lastLogMessage() == "More than one preset selected" );
        REQUIRE( app.moveCalls() == 0 );
    }

    SECTION( "a valid single selection is accepted and logged" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus", "block" }, "preset" );
        app.setPresetPositions( { 1, 2, 3 } );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest( { { "focus", pcf::IndiElement::On } } ) == 0 );
        REQUIRE( app.moveCalls() == 1 );
        REQUIRE( app.lastMoveTarget() == 2.0f );
        REQUIRE( app.movingStateValue() == 1 );
        REQUIRE( app.presetNameIndexValue() == 1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_NOTICE );
    }
}

/// Verify the home request callback of stdMotionStage. Only a matching property with the request element on triggers startHoming().
/**
 * \ingroup stdMotionStage_tests
 */
TEST_CASE( "stdMotionStage home callback", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;
    app.configurePresets( { "open", "focus" }, "preset" );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

    SECTION( "mismatched property is rejected" )
    {
        REQUIRE( app.applyHomeRequest( pcf::IndiElement::On, true, false ) == -1 );
        REQUIRE( app.homingCalls() == 0 );
    }

    SECTION( "missing request element is a no-op" )
    {
        REQUIRE( app.applyHomeRequest( pcf::IndiElement::On, false ) == 0 );
        REQUIRE( app.homingCalls() == 0 );
    }

    SECTION( "request element off is a no-op" )
    {
        REQUIRE( app.applyHomeRequest( pcf::IndiElement::Off ) == 0 );
        REQUIRE( app.homingCalls() == 0 );
    }

    SECTION( "request element on triggers startHoming" )
    {
        REQUIRE( app.applyHomeRequest( pcf::IndiElement::On ) == 0 );
        REQUIRE( app.homingCalls() == 1 );
        REQUIRE( app.movingStateValue() == 0 );
    }
}

/// Verify the stop request callback of stdMotionStage, including the unique-key ambiguity guard.
/**
 * \ingroup stdMotionStage_tests
 */
TEST_CASE( "stdMotionStage stop callback", "[dev::stdMotionStage]" )
{
    SECTION( "mismatched property is rejected" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

        REQUIRE( app.applyStopRequest( pcf::IndiElement::On, true, "wrongDevice" ) == -1 );
        REQUIRE( app.stopCalls() == 0 );
    }

    SECTION( "missing request element is a no-op" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

        REQUIRE( app.applyStopRequest( pcf::IndiElement::On, false ) == 0 );
        REQUIRE( app.stopCalls() == 0 );
    }

    SECTION( "request element off is a no-op" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

        REQUIRE( app.applyStopRequest( pcf::IndiElement::Off ) == 0 );
        REQUIRE( app.stopCalls() == 0 );
    }

    SECTION( "request element on triggers stop" )
    {
        stdMotionStageHarness app;
        app.configurePresets( { "open", "focus" }, "preset" );
        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

        REQUIRE( app.applyStopRequest( pcf::IndiElement::On ) == 0 );
        REQUIRE( app.stopCalls() == 1 );
        REQUIRE( app.movingStateValue() == 0 );
    }

    SECTION( "colliding unique keys with a mismatched name are rejected and logged" )
    {
        // pcf::IndiProperty::createUniqueKey() joins the device name and the property name with a period.
        // If the device name itself contains a period, two different device and name pairs can produce
        // the same unique key. The stop callback guards against this ambiguity by also comparing names.
        stdMotionStageHarness app;
        app.setConfigName( "stest.st" );
        app.configurePresets( { "open", "focus" }, "preset" );
        REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

        stdMotionStageHarness::resetLogState();

        // The real key is "stest.st" + "." + "stop", which is "stest.st.stop".
        // The colliding key is "stest" + "." + "st.stop". That is the same string with a different name.
        REQUIRE( app.applyStopRequest( pcf::IndiElement::On, true, "stest", "st.stop" ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        // stdMotionStage logs this particular mismatch with the default priority because no level is passed.
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_DEFAULT );
        REQUIRE( app.stopCalls() == 0 );
    }
}

/// Verify the preset-name resolution and telemetry recording logic of stdMotionStage.
/**
 * \ingroup stdMotionStage_tests
 *
 * recordStage() is called after each state change. The harness telem() override counts
 * records instead of writing them, so the test checks the count before and after each call.
 * The moves are chosen so each branch of the preset name resolution is reached.
 */
TEST_CASE( "stdMotionStage recordStage and preset-name resolution", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;
    app.configurePresets( { "open", "focus", "block" }, "preset" );
    app.setPresetPositions( { 1, 2, 3 } );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

    // Before any motion m_preset defaults to 0, so telemetryPresetName() resolves to an empty string.
    int before = stdMotionStageHarness::telemCount();
    REQUIRE( app.recordStage( true ) == 0 );
    REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

    // Calling again with no state change and force=false should not record again.
    before = stdMotionStageHarness::telemCount();
    REQUIRE( app.recordStage( false ) == 0 );
    REQUIRE( stdMotionStageHarness::telemCount() == before );

    // A raw numeric move to a known preset position resolves via the fallback presetIndex branch.
    REQUIRE( app.applyPresetRequest( 3.0f ) == 0 ); // This is index 2, named block. presetNameIndex is cleared.
    REQUIRE( app.presetNameIndexValue() == -1 );

    before = stdMotionStageHarness::telemCount();
    REQUIRE( app.recordStage( false ) == 0 );
    REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

    // A named preset move while still moving resolves via the primary presetNameIndex branch.
    REQUIRE( app.applyPresetNameRequest( { { "focus", pcf::IndiElement::On } } ) == 0 ); // This is index 1.
    app.setMoving( 1 );
    REQUIRE( app.presetNameIndexValue() == 1 );

    before = stdMotionStageHarness::telemCount();
    REQUIRE( app.recordStage( false ) == 0 );
    REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

    // Simulate arrival by clearing m_moving while movingState and presetNameIndex are unchanged.
    // This resolves via the same-position secondary presetNameIndex branch.
    app.setMoving( 0 );

    before = stdMotionStageHarness::telemCount();
    REQUIRE( app.recordStage( false ) == 0 );
    REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

    // Move to a position with no matching configured preset. The harness presetNumber() then
    // reports -1. This reaches the final no-match fallback in activePresetNameIndex() and activePresetName().
    REQUIRE( app.applyPresetRequest( 42.0f ) == 0 );
    REQUIRE( app.presetNameIndexValue() == -1 );

    before = stdMotionStageHarness::telemCount();
    REQUIRE( app.recordStage( false ) == 0 );
    REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );
}

/// Verify stdMotionStage::updateINDI(), including the no-driver early return and the full INDI update path.
/**
 * \ingroup stdMotionStage_tests
 *
 * The full path uses a real but never-activated INDI driver from the harness, so no INDI
 * server is needed. Each successful update also records one telemetry entry.
 */
TEST_CASE( "stdMotionStage updateINDI", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;
    app.configurePresets( { "open", "focus", "block" }, "preset" );
    app.setPresetPositions( { 1, 2, 3 } );
    REQUIRE( app.dev::stdMotionStage<stdMotionStageHarness>::appStartup() == 0 );

    SECTION( "updateINDI is a no-op without an INDI driver" )
    {
        int before = stdMotionStageHarness::telemCount();
        REQUIRE( app.updateINDI() == 0 );
        REQUIRE( stdMotionStageHarness::telemCount() == before );
    }

    SECTION( "updateINDI drives INDI property state once a driver is present" )
    {
        app.setFakeIndiDriver( true );

        // First call. The preset index defaults to 0 because nothing has matched yet.
        // The stage is moving, so the property state is busy.
        app.setPresetNumberValue( 0 );
        app.setMoving( 1 );
        int before = stdMotionStageHarness::telemCount();
        REQUIRE( app.updateINDI() == 0 );
        REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

        // Second call. Switch the active preset and stop moving. This reaches the idle-state branch
        // and the branches that toggle the old presetName element off and the new one on.
        REQUIRE( app.applyPresetRequest( 2.0f ) == 0 ); // This is index 1.
        app.setMoving( 0 );
        before = stdMotionStageHarness::telemCount();
        REQUIRE( app.updateINDI() == 0 );
        REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

        // Third call. A numeric preset move while moving. This reaches the busy branch taken when
        // m_moving is set and m_movingState is less than 1.
        REQUIRE( app.applyPresetRequest( 3.0f ) == 0 );
        app.setMoving( 1 );
        before = stdMotionStageHarness::telemCount();
        REQUIRE( app.updateINDI() == 0 );
        REQUIRE( stdMotionStageHarness::telemCount() == before + 1 );

        app.setFakeIndiDriver( false );
    }
}

/// Verify the out-of-range guard in setPresetNameTracking().
/**
 * \ingroup stdMotionStage_tests
 *
 * The guard cannot be reached through the only normal call site, because
 * newCallBack_m_indiP_presetName() always passes an index it has already validated.
 * The harness wrapper callSetPresetNameTracking() calls the protected function directly.
 */
TEST_CASE( "stdMotionStage setPresetNameTracking rejects out-of-range indices", "[dev::stdMotionStage]" )
{
    stdMotionStageHarness app;
    app.configurePresets( { "open", "focus" }, "preset" );

    REQUIRE( app.callSetPresetNameTracking( -1 ) == -1 );
    REQUIRE( app.presetNameIndexValue() == -1 );

    REQUIRE( app.callSetPresetNameTracking( 5 ) == -1 );
    REQUIRE( app.presetNameIndexValue() == -1 );

    REQUIRE( app.callSetPresetNameTracking( 1 ) == 0 );
    REQUIRE( app.presetNameIndexValue() == 1 );
}

} // namespace devTest
} // namespace appTest
} // namespace libXWCTest
