/** \file zaberCtrl_test.cpp
 * \brief Catch2 tests for the zaberCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../../../libMagAOX/libMagAOX.hpp"

// Validation tests send empty properties; data-bearing fixtures must execute the real callback.
#undef INDI_VALIDATE_CALLBACK_PROPS
#define INDI_VALIDATE_CALLBACK_PROPS( prop1, prop2 ) \
    INDI_VALIDATE_CALLBACK_PROPS_IMPL( prop1, prop2 ) \
    if( ( prop2 ).getElements().empty() )             \
    {                                                \
        return 0;                                    \
    }

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
    zaberCtrl_test( const std::string &device /**< [in] test INDI device name */ );

    /// Set the current test position state.
    void setStagePosition( double pos, /**< [in] measured position in millimeters */
                           double countsPerMillimeter /**< [in] device microsteps per millimeter */ );

    /// Set the configured preset positions and names for testing.
    void setPresets( const std::vector<float> &positions, /**< [in] configured physical positions */
                     const std::vector<std::string> &names /**< [in] corresponding preset names */ );

    /// Set the current parked state for testing.
    void setParked( bool parked /**< [in] reported parking state */ );

    /// Set the current motion and preset telemetry values for testing.
    void setStageTelemetry( int8_t moving, /**< [in] motion or power state */
                            float preset, /**< [in] current numerical preset */
                            float presetTarget /**< [in] commanded preset target */ );

    /// Set the current motion-state classification for testing.
    void setMovingState( int8_t movingState /**< [in] named or arbitrary move classification */ );

    /// Set the configured home-preset index for testing.
    void setHomePresetIndex( int homePresetIndex /**< [in] post-home preset, negative to disable */ );

    /// Track a specific preset-name alias for testing.
    int setPresetAliasIndex( int presetNameIndex /**< [in] requested preset alias index */ );

    /// Clear any tracked preset-name alias for testing.
    void clearPresetAliasIndex();

    /// Resolve the active preset-name index for the current position.
    int activeAliasIndex();

    /// Resolve the active preset name for the current position.
    std::string activeAliasName();

    /// Resolve the preset name that telemetry should record.
    std::string telemetryAliasName();

    /// Invoke the base-class power-off handling under test.
    int stageOnPowerOff();

    /// Invoke the powered-off telemetry sync under test.
    int syncPoweredOffTelemetry();

    /// Apply a stage-state INDI update for the configured test stage.
    int applyStageState( const std::string &stageState /**< [in] reported low-level FSM state */ );

    /// Get the current FSM state.
    stateCodes::stateCodeT fsmState();

    /// Get the current homing bookkeeping state.
    int homingState() const;

    /// Get the current logged moving state.
    int8_t movingState() const;

    /// Get the current logged preset value.
    float presetValue() const;

    /// Get the current logged preset target value.
    float presetTargetValue() const;
};

zaberCtrl_test::zaberCtrl_test( const std::string &device )
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

void zaberCtrl_test::setStagePosition( double pos, double countsPerMillimeter )
{
    m_pos                 = pos;
    m_countsPerMillimeter = countsPerMillimeter;
}

void zaberCtrl_test::setPresets( const std::vector<float> &positions, const std::vector<std::string> &names )
{
    m_presetPositions = positions;
    m_presetNames     = names;
}

void zaberCtrl_test::setParked( bool parked )
{
    m_parked = parked;
}

void zaberCtrl_test::setStageTelemetry( int8_t moving, float preset, float presetTarget )
{
    m_moving        = moving;
    m_preset        = preset;
    m_preset_target = presetTarget;
}

void zaberCtrl_test::setMovingState( int8_t movingState )
{
    m_movingState = movingState;
}

void zaberCtrl_test::setHomePresetIndex( int homePresetIndex )
{
    m_homePreset = homePresetIndex;
}

int zaberCtrl_test::setPresetAliasIndex( int presetNameIndex )
{
    return setPresetNameTracking( presetNameIndex );
}

void zaberCtrl_test::clearPresetAliasIndex()
{
    clearPresetNameTracking();
}

int zaberCtrl_test::activeAliasIndex()
{
    return activePresetNameIndex( presetNumber() );
}

std::string zaberCtrl_test::activeAliasName()
{
    return activePresetName( presetNumber() );
}

std::string zaberCtrl_test::telemetryAliasName()
{
    return telemetryPresetName();
}

int zaberCtrl_test::stageOnPowerOff()
{
    return dev::stdMotionStage<zaberCtrl>::onPowerOff();
}

int zaberCtrl_test::syncPoweredOffTelemetry()
{
    return syncPowerOffStageTelemetry();
}

int zaberCtrl_test::applyStageState( const std::string &stageState )
{
    pcf::IndiProperty ip;
    ip.setDevice( "stest" );
    ip.setName( "curr_state" );
    ip.add( pcf::IndiElement( m_stageName ) );
    ip[m_stageName].set( stageState );

    return setCallBack_m_indiP_stageState( ip );
}

stateCodes::stateCodeT zaberCtrl_test::fsmState()
{
    return state();
}

int zaberCtrl_test::homingState() const
{
    return m_homingState;
}

int8_t zaberCtrl_test::movingState() const
{
    return m_moving;
}

float zaberCtrl_test::presetValue() const
{
    return m_preset;
}

float zaberCtrl_test::presetTargetValue() const
{
    return m_preset_target;
}
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

/// Parked stages retain their measured preset telemetry while unparked stages clear it.
/** \ingroup zaberCtrl_unit_test
 */
SCENARIO( "Power-off stage telemetry", "[zaberCtrl]" )
{
    // clang-format off
    #ifdef ZABERCTRL_TEST_DOXYGEN_REF
    zaberCtrl::syncPowerOffStageTelemetry();
    #endif
    // clang-format on

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

/// Low-level homing transitions update the controller FSM and post-home bookkeeping.
/** \ingroup zaberCtrl_unit_test
 */
SCENARIO( "Homing READY transitions update the controller FSM promptly", "[zaberCtrl]" )
{
    // clang-format off
    #ifdef ZABERCTRL_TEST_DOXYGEN_REF
    zaberCtrl::setCallBack_m_indiP_stageState( pcf::IndiProperty() );
    #endif
    // clang-format on

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

/// Requested aliases remain distinct at shared positions, including after power-off.
/** \ingroup zaberCtrl_unit_test
 */
SCENARIO( "Preset-name aliases follow the selected shared-position preset", "[zaberCtrl]" )
{
    // clang-format off
    #ifdef ZABERCTRL_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::activePresetNameIndex( 0 );
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::activePresetName( 0 );
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::telemetryPresetName();
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::onPowerOff();
    #endif
    // clang-format on

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

    WHEN( "the alias at the retained position is preserved on power off" )
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

/// Verify power-off resolves the retained position rather than an interrupted named target.
/** \ingroup zaberCtrl_unit_test
 */
TEST_CASE( "Powered-off preset names follow the retained position", "[zaberCtrl][parked]" )
{
    // clang-format off
    #ifdef ZABERCTRL_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::onPowerOff();
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::activePresetNameIndex( 0 );
    MagAOX::app::dev::stdMotionStage<zaberCtrl>::telemetryPresetName();
    zaberCtrl::syncPowerOffStageTelemetry();
    #endif
    // clang-format on

    zaberCtrl_test zct( "stest" );
    zct.setPresets( { -1, 10, 20, 20 }, { "none", "open", "science", "focus" } );
    zct.setParked( true );
    zct.setStagePosition( 10.0, 1000.0 );
    zct.setStageTelemetry( 1, 1, 3 );
    zct.setMovingState( 1 );
    REQUIRE( zct.setPresetAliasIndex( 3 ) == 0 );

    // A running named move still reports the requested alias.
    REQUIRE( zct.activeAliasName() == "focus" );

    SECTION( "Power-off interrupts motion at another preset" )
    {
        REQUIRE( zct.stageOnPowerOff() == 0 );
        REQUIRE( zct.syncPoweredOffTelemetry() == 0 );
        REQUIRE( zct.movingState() == -2 );
        REQUIRE( zct.presetValue() == 1 );
        REQUIRE( zct.presetTargetValue() == 1 );
        REQUIRE( zct.activeAliasIndex() == 1 );
        REQUIRE( zct.activeAliasName() == "open" );
        REQUIRE( zct.telemetryAliasName() == "open" );
    }

    SECTION( "Power-off interrupts motion between presets" )
    {
        zct.setStagePosition( 15.0, 1000.0 );
        REQUIRE( zct.stageOnPowerOff() == 0 );
        REQUIRE( zct.syncPoweredOffTelemetry() == 0 );
        REQUIRE( zct.presetValue() == 0 );
        REQUIRE( zct.presetTargetValue() == 0 );
        REQUIRE( zct.activeAliasIndex() == 0 );
        REQUIRE( zct.activeAliasName() == "none" );
        REQUIRE( zct.telemetryAliasName().empty() );
    }

    SECTION( "Not-homed sentinel also cannot report a different commanded position" )
    {
        zct.setStageTelemetry( -1, 1, 3 );
        REQUIRE( zct.activeAliasName() == "open" );
        REQUIRE( zct.telemetryAliasName() == "open" );
    }

    SECTION( "Power-off at the alias position preserves its name" )
    {
        zct.setStagePosition( 20.0, 1000.0 );
        REQUIRE( zct.stageOnPowerOff() == 0 );
        REQUIRE( zct.syncPoweredOffTelemetry() == 0 );
        REQUIRE( zct.presetValue() == 2 );
        REQUIRE( zct.activeAliasIndex() == 3 );
        REQUIRE( zct.activeAliasName() == "focus" );
        REQUIRE( zct.telemetryAliasName() == "focus" );
    }
}

} // namespace zaberCtrlTest

} // namespace libXWCTest
