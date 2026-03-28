/** \file zaberCtrl_test.cpp
 * \brief Catch2 tests for the zaberCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../zaberCtrl.hpp"

using namespace MagAOX::app;

namespace ZCTRLTEST
{

class zaberCtrl_test : public zaberCtrl
{

  public:
    /// Construct a testable controller instance.
    zaberCtrl_test( const std::string &device )
    {
        m_configName = device;

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

SCENARIO( "INDI Callbacks", "[zaberCtrl]" )
{
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
        REQUIRE( zct.presetValue() == 0 );
        REQUIRE( zct.presetTargetValue() == 0 );
        REQUIRE( zct.activeAliasIndex() == 2 );
        REQUIRE( zct.activeAliasName() == "science" );
    }
}

} // namespace ZCTRLTEST
