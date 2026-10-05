/** \file zaberCtrl_test.cpp
 * \brief Catch2 tests for the zaberCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../zaberCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup zaberCtrl_unit_test zaberCtrl Unit Tests
 * \brief Unit tests for the zaberCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `zaberCtrl` unit tests.
/** \ingroup zaberCtrl_unit_test
 */
namespace zaberCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class zaberCtrl_test : public zaberCtrl
{

  public:
    /// Construct a testable controller instance.
    zaberCtrl_test( const std::string &device )
    {
        m_configName = device;
        m_stageName  = "stage";

        XWCTEST_SETUP_INDI_NEW_PROP( pos );
        XWCTEST_SETUP_INDI_NEW_PROP( rawPos );

        // stdMotionStage:
        XWCTEST_SETUP_INDI_NEW_PROP( preset );
        XWCTEST_SETUP_INDI_NEW_PROP( presetName );
        XWCTEST_SETUP_INDI_NEW_PROP( home );
        XWCTEST_SETUP_INDI_NEW_PROP( stop );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageState, stest, curr_state );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageMaxRawPos, stest, max_pos );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageRawPos, stest, curr_pos );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageTgtPos, stest, tgt_pos );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageTemp, stest, temp );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageParked, stest, parked );
    }

    /// Set the current test position state.
    void setStagePosition( double pos, double countsPerMillimeter )
    {
        m_pos                 = pos;
        m_countsPerMillimeter = countsPerMillimeter;
    }

    /// Set the configured preset positions and names for testing.
    void setPresets( const std::vector<float> &positions, const std::vector<std::string> &names )
    {
        m_presetPositions = positions;
        m_presetNames     = names;
    }

    /// Set the current parked state for testing.
    void setParked( bool parked )
    {
        m_parked = parked;
    }

    /// Set the current motion and preset telemetry values for testing.
    void setStageTelemetry( int8_t moving, float preset, float presetTarget )
    {
        m_moving        = moving;
        m_preset        = preset;
        m_preset_target = presetTarget;
    }

    /// Set the current motion-state classification for testing.
    void setMovingState( int8_t movingState )
    {
        m_movingState = movingState;
    }

    /// Set the configured home-preset index for testing.
    void setHomePresetIndex( int homePresetIndex )
    {
        m_homePreset = homePresetIndex;
    }

    /// Track a specific preset-name alias for testing.
    int setPresetAliasIndex( int presetNameIndex )
    {
        return setPresetNameTracking( presetNameIndex );
    }

    /// Clear any tracked preset-name alias for testing.
    void clearPresetAliasIndex()
    {
        clearPresetNameTracking();
    }

    /// Resolve the active preset-name index for the current position.
    int activeAliasIndex()
    {
        return activePresetNameIndex( presetNumber() );
    }

    /// Resolve the active preset name for the current position.
    std::string activeAliasName()
    {
        return activePresetName( presetNumber() );
    }

    /// Resolve the preset name that telemetry should record.
    std::string telemetryAliasName()
    {
        return telemetryPresetName();
    }

    /// Invoke the base-class power-off handling under test.
    int stageOnPowerOff()
    {
        return dev::stdMotionStage<zaberCtrl>::onPowerOff();
    }

    /// Invoke the powered-off telemetry sync under test.
    int syncPoweredOffTelemetry()
    {
        return syncPowerOffStageTelemetry();
    }

    /// Apply a stage-state INDI update for the configured test stage.
    int applyStageState( const std::string &stageState )
    {
        pcf::IndiProperty ip;
        ip.setDevice( "stest" );
        ip.setName( "curr_state" );
        ip.add( pcf::IndiElement( m_stageName ) );
        ip[m_stageName].set( stageState );

        return setCallBack_m_indiP_stageState( ip );
    }

    /// Get the current FSM state.
    stateCodes::stateCodeT fsmState()
    {
        return state();
    }

    /// Get the current homing bookkeeping state.
    int homingState() const
    {
        return m_homingState;
    }

    /// Get the current logged moving state.
    int8_t movingState() const
    {
        return m_moving;
    }

    /// Get the current logged preset value.
    float presetValue() const
    {
        return m_preset;
    }

    /// Get the current logged preset target value.
    float presetTargetValue() const
    {
        return m_preset_target;
    }
};
/// \endcond

/// Verify zaberCtrl callback validation and preset-alias helpers behave as expected.
/**
 * \ingroup zaberCtrl_unit_test
 */
SCENARIO( "INDI Callbacks", "[zaberCtrl]" )
{
    // clang-format off
    #ifdef ZABERCTRL_TEST_DOXYGEN_REF
    zaberCtrl::newCallBack_m_indiP_pos( pcf::IndiProperty() );
    zaberCtrl::newCallBack_m_indiP_rawPos( pcf::IndiProperty() );
    zaberCtrl::setCallBack_m_indiP_stageState( pcf::IndiProperty() );
    zaberCtrl::activePresetName( 0 );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, pos );
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, rawPos );
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, preset );
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, presetName );
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, home );
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, stop );

    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageState, stest, curr_state );
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageMaxRawPos, stest, max_pos );
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageRawPos, stest, curr_pos );
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageTgtPos, stest, tgt_pos );
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageTemp, stest, temp );
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageParked, stest, parked );
}

SCENARIO( "Power-off stage telemetry", "[zaberCtrl]" )
{
    zaberCtrl_test zct( "stest" );

    zct.setPresets( { -1, 1, 2, 3 }, { "none", "one", "two", "three" } );
    zct.setStagePosition( 2.0, 1000.0 );

    WHEN( "the stage powers off while parked" )
    {
        zct.setParked( true );
        zct.setStageTelemetry( 0, 2, 2 );

        REQUIRE( zct.syncPoweredOffTelemetry() == 0 );
        REQUIRE( zct.presetValue() == 2 );
        REQUIRE( zct.presetTargetValue() == 2 );
    }

    WHEN( "the stage powers off while not parked" )
    {
        zct.setParked( false );
        zct.setStageTelemetry( 0, 2, 2 );

        REQUIRE( zct.syncPoweredOffTelemetry() == 0 );
        REQUIRE( zct.presetValue() == 0 );
        REQUIRE( zct.presetTargetValue() == 0 );
    }
}

SCENARIO( "Homing READY transitions update the controller FSM promptly", "[zaberCtrl]" )
{
    zaberCtrl_test zct( "stest" );

    WHEN( "homing completes without a configured post-home preset move" )
    {
        zct.setHomePresetIndex( -1 );

        REQUIRE( zct.applyStageState( "HOMING" ) == 0 );
        REQUIRE( zct.fsmState() == stateCodes::HOMING );
        REQUIRE( zct.homingState() == 1 );

        REQUIRE( zct.applyStageState( "READY" ) == 0 );
        REQUIRE( zct.fsmState() == stateCodes::READY );
        REQUIRE( zct.homingState() == 0 );
    }

    WHEN( "homing completes and a post-home preset move is still pending" )
    {
        zct.setHomePresetIndex( 1 );

        REQUIRE( zct.applyStageState( "HOMING" ) == 0 );
        REQUIRE( zct.fsmState() == stateCodes::HOMING );
        REQUIRE( zct.homingState() == 1 );

        REQUIRE( zct.applyStageState( "READY" ) == 0 );
        REQUIRE( zct.fsmState() == stateCodes::HOMING );
        REQUIRE( zct.homingState() == 2 );
    }
}

SCENARIO( "Preset-name aliases follow the selected shared-position preset", "[zaberCtrl]" )
{
    zaberCtrl_test zct( "stest" );

    zct.setPresets( { -1, 10, 20, 20 }, { "none", "open", "science", "focus" } );
    zct.setStagePosition( 20.0, 1000.0 );
    zct.setStageTelemetry( 0, 3, 3 );

    WHEN( "a specific alias was selected for a shared preset position" )
    {
        REQUIRE( zct.setPresetAliasIndex( 3 ) == 0 );

        REQUIRE( zct.activeAliasIndex() == 3 );
        REQUIRE( zct.activeAliasName() == "focus" );
    }

    WHEN( "the stage is moving toward a selected alias" )
    {
        REQUIRE( zct.setPresetAliasIndex( 3 ) == 0 );
        zct.setMovingState( 1 );
        zct.setStageTelemetry( 1, 2, 3 );

        REQUIRE( zct.activeAliasIndex() == 3 );
        REQUIRE( zct.activeAliasName() == "focus" );
    }

    WHEN( "no alias is being tracked" )
    {
        zct.clearPresetAliasIndex();

        REQUIRE( zct.activeAliasIndex() == 2 );
        REQUIRE( zct.activeAliasName() == "science" );
    }

    WHEN( "the alias tracking is cleared on power off" )
    {
        REQUIRE( zct.setPresetAliasIndex( 3 ) == 0 );

        REQUIRE( zct.stageOnPowerOff() == 0 );
        REQUIRE( zct.movingState() == -2 );
        REQUIRE( zct.presetValue() == 3 );
        REQUIRE( zct.presetTargetValue() == 3 );
        REQUIRE( zct.activeAliasIndex() == 3 );
        REQUIRE( zct.activeAliasName() == "focus" );
        REQUIRE( zct.telemetryAliasName() == "focus" );
    }
}

} // namespace zaberCtrlTest

} // namespace libXWCTest
