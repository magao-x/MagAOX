/** \file elliptecCtrl_test.cpp
 * \brief Catch2 tests for the elliptecCtrl client of elliptecLowLevel.
 * \author Miles Lucas
 *
 * \ingroup elliptecCtrl_files
 */

/** \defgroup elliptecCtrl_unit_test elliptecCtrl Unit Tests
 * \brief Unit tests for the elliptecCtrl application.
 *
 * \ingroup application_unit_test
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include <filesystem>

#define ELLIPTECCTRL_TEST_NOSEND
#include "../elliptecCtrl.hpp"

namespace libXWCTest
{
namespace elliptecCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace
{

/// Test harness that feeds low-level property updates and inspects the commands sent back.
class elliptecCtrl_test : public MagAOX::app::elliptecCtrl
{
  public:
    /// Construct a client of the low-level app `ellbus` for stage `qwp1`.
    explicit elliptecCtrl_test( const std::string &name /**< [in] app name and scratch directory */ )
    {
        const std::string root = "/tmp/elliptecCtrl_test/" + name;
        std::filesystem::remove_all( root );
        std::filesystem::create_directories( root + "/telem" );

        m_configName = name;
        m_basePath   = root;
        m_sysPath    = root + "/sys";
        m_tel.logPath( root + "/telem" );
        m_tel.logExt( "bintel" );
        m_tel.logName( m_configName );
        m_tel.m_logLevel = logPrio::LOG_TELEM;
        m_maxInterval    = 3600.0;

        m_lowLevelName    = "ellbus";
        m_stageName       = "qwp1";
        m_presetPositions = { 115.0F, 300.0F };
        m_presetNames     = { "lab", "sky" };

        m_indiP_preset.setDevice( m_configName );
        m_indiP_preset.setName( "preset" );
        m_indiP_presetName.setDevice( m_configName );
        m_indiP_presetName.setName( "presetName" );
        m_indiP_home.setDevice( m_configName );
        m_indiP_home.setName( "home" );
        m_indiP_stop.setDevice( m_configName );
        m_indiP_stop.setName( "stop" );
    }

    /// Shut down telemetry.
    ~elliptecCtrl_test()
    {
        m_tel.logShutdown( true );
    }

    /// Run appStartup from INITIALIZED.
    int startup()
    {
        state( MagAOX::app::stateCodes::INITIALIZED );
        return appStartup();
    }

    /// Run one loop.
    int logic()
    {
        return appLogic();
    }

    /// A low-level property update for one stage.
    static pcf::IndiProperty llProp( const std::string      &name,  /**< [in] property name */
                                     const std::string      &stage, /**< [in] element name */
                                     pcf::IndiProperty::Type type )
    {
        pcf::IndiProperty ip( type );
        ip.setDevice( "ellbus" );
        ip.setName( name );
        ip.add( pcf::IndiElement( stage ) );
        return ip;
    }

    /// Report a low-level state.
    int llState( const std::string &st, const std::string &stage = "qwp1" )
    {
        pcf::IndiProperty ip = llProp( "curr_state", stage, pcf::IndiProperty::Text );
        ip[stage].set( st );
        return setCallBack_m_indiP_llState( ip );
    }

    /// Report a low-level position.
    int llPos( double deg, const std::string &stage = "qwp1" )
    {
        pcf::IndiProperty ip = llProp( "curr_pos", stage, pcf::IndiProperty::Number );
        ip[stage].set( deg );
        return setCallBack_m_indiP_llPos( ip );
    }

    /// Report a low-level error.
    int llError( const std::string &err )
    {
        pcf::IndiProperty ip = llProp( "last_error", "qwp1", pcf::IndiProperty::Text );
        ip["qwp1"].set( err );
        return setCallBack_m_indiP_llError( ip );
    }

    /// Request a preset by number.
    int requestPreset( float n )
    {
        pcf::IndiProperty ip = m_indiP_preset;
        ip["target"].set( n );
        return newCallBack_m_indiP_preset( ip );
    }

    /// Request a preset by name.
    int requestPresetName( const std::string &name )
    {
        pcf::IndiProperty ip = m_indiP_presetName;
        for( auto &n : m_presetNames )
        {
            ip[n].setSwitchState( n == name ? pcf::IndiElement::On : pcf::IndiElement::Off );
        }
        return newCallBack_m_indiP_presetName( ip );
    }

    /// Press `home`.
    int pressHome()
    {
        pcf::IndiProperty ip = m_indiP_home;
        ip["request"].setSwitchState( pcf::IndiElement::On );
        return newCallBack_m_indiP_home( ip );
    }

    /// Press `stop`.
    int pressStop()
    {
        pcf::IndiProperty ip = m_indiP_stop;
        ip["request"].setSwitchState( pcf::IndiElement::On );
        return newCallBack_m_indiP_stop( ip );
    }

    /// Request an absolute angle.
    int absDeg( double deg )
    {
        pcf::IndiProperty ip = m_ipAbsDeg;
        ip["target"].set( deg );
        return newCallBack_m_ipAbsDeg( ip );
    }

    /// Set the relative step.
    int relDeg( double deg )
    {
        pcf::IndiProperty ip = m_ipRelDeg;
        ip["target"].set( deg );
        return newCallBack_m_ipRelDeg( ip );
    }

    /// Press `relMove`.
    int relMove()
    {
        pcf::IndiProperty ip = m_ipRelMove;
        ip["request"].setSwitchState( pcf::IndiElement::On );
        return newCallBack_m_ipRelMove( ip );
    }

    /// Name of the last property sent to the low level, and clear it.
    std::string lastCommandName()
    {
        std::string n = m_lastCommand.getName();
        m_lastCommand = pcf::IndiProperty();
        return n;
    }

    /// Numeric value of the last command for this stage.
    double lastCommandValue() const
    {
        return m_lastCommand[m_stageName].get<double>();
    }

    /// Device the last command was sent to.
    std::string lastCommandDevice() const
    {
        return m_lastCommand.getDevice();
    }

    /// Operator status text.
    std::string status()
    {
        return m_ipStatus["current"].getValue();
    }

    /// Set the post-home offset.
    void homeOffset( double deg )
    {
        m_homeOffsetDeg = deg;
    }

    /// Set `stage.powerOnHome`.
    void powerOnHome( bool on )
    {
        m_powerOnHome = on;
    }

    /// Current stdMotionStage moving code.
    int8_t moving() const
    {
        return m_moving;
    }

    /// Current preset (1-based, 0 for none).
    float preset() const
    {
        return m_preset;
    }

    /// Read a configuration file and load it.
    int loadConfigFile( const std::string &path )
    {
        config.readConfig( path );
        return loadConfigImpl( config );
    }

    /// Configured low-level name.
    const std::string &lowLevelName() const
    {
        return m_lowLevelName;
    }

    /// Configured stage name.
    const std::string &stageName() const
    {
        return m_stageName;
    }

    /// Configured home offset.
    double homeOffsetDeg() const
    {
        return m_homeOffsetDeg;
    }
};

/// Start a client and bring its stage to READY at the given position.
void readyAt( elliptecCtrl_test &app, double deg )
{
    REQUIRE( app.startup() == 0 );
    app.llPos( deg );
    app.llState( "READY" );
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
}

} // namespace
/// \endcond

/// Low-level states map onto the client FSM and status text.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl follows the low-level state of its stage", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::setCallBack_m_indiP_llState( ip );
    MagAOX::app::elliptecCtrl::setCallBack_m_indiP_llPos( ip );
    MagAOX::app::elliptecCtrl::setCallBack_m_indiP_llError( ip );
    MagAOX::app::elliptecCtrl::applyState_( "" );
    MagAOX::app::elliptecCtrl::updateStatus_();
    // clang-format on
#endif

    using MagAOX::app::stateCodes;

    elliptecCtrl_test app( "ellctrl-states" );
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    REQUIRE( app.status() == "Not Connected" );

    REQUIRE( app.llState( "POWEROFF" ) == 0 );
    REQUIRE( app.state() == stateCodes::POWEROFF );
    REQUIRE( app.status() == "Powered Off" );

    app.llState( "POWERON" );
    REQUIRE( app.state() == stateCodes::POWERON );

    app.llState( "NODEVICE" );
    REQUIRE( app.state() == stateCodes::NODEVICE );
    REQUIRE( app.status() == "Wrong or missing device" );

    app.llState( "NOTHOMED" );
    REQUIRE( app.state() == stateCodes::NOTHOMED );
    REQUIRE( app.moving() == -1 );
    REQUIRE( app.status() == "Not Homed" );

    app.llState( "OPERATING" );
    REQUIRE( app.state() == stateCodes::OPERATING );
    REQUIRE( app.status() == "Moving" );

    app.llState( "HOMING" );
    REQUIRE( app.state() == stateCodes::HOMING );
    REQUIRE( app.status() == "Homing" );

    app.llState( "READY" );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.status() == "OK" );

    app.llError( "move failed (stopped away from target)" );
    REQUIRE( app.status() == "Error: move failed (stopped away from target)" );
    app.llError( "" );
    REQUIRE( app.status() == "OK" );

    app.llState( "SOMETHING" );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );

    // Updates for other stages on the bus are ignored.
    app.llState( "READY" );
    REQUIRE( app.llState( "POWEROFF", "qwp2" ) == 0 );
    REQUIRE( app.llPos( 42.0, "qwp2" ) == 0 );
    REQUIRE( app.state() == stateCodes::READY );
}

/// Presets, absolute and relative moves are sent as absolute targets in degrees.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl sends absolute targets to the low level", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::moveTo( 0 );
    MagAOX::app::elliptecCtrl::newCallBack_m_ipAbsDeg( ip );
    MagAOX::app::elliptecCtrl::newCallBack_m_ipRelDeg( ip );
    MagAOX::app::elliptecCtrl::newCallBack_m_ipRelMove( ip );
    MagAOX::app::elliptecCtrl::sendNumber_( "", 0 );
    // clang-format on
#endif

    elliptecCtrl_test app( "ellctrl-targets" );
    readyAt( app, 0.0 );

    REQUIRE( app.requestPreset( 1 ) == 0 );
    REQUIRE( app.lastCommandDevice() == "ellbus" );
    REQUIRE( app.lastCommandValue() == Approx( 115.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );
    REQUIRE( app.state() == MagAOX::app::stateCodes::OPERATING );

    app.llPos( 115.2 );
    app.llState( "READY" );
    REQUIRE( app.preset() == Approx( 1.0F ) );

    REQUIRE( app.requestPresetName( "sky" ) == 0 );
    REQUIRE( app.lastCommandValue() == Approx( 300.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );

    REQUIRE( app.absDeg( 400.0 ) == 0 );
    REQUIRE( app.lastCommandValue() == Approx( 40.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );

    // Relative moves become absolute targets: current position + step, wrapped.
    app.llPos( 350.0 );
    REQUIRE( app.relDeg( 900.0 ) == 0 ); // clamped to 360
    REQUIRE( app.relDeg( 20.0 ) == 0 );
    REQUIRE( app.relMove() == 0 );
    REQUIRE( app.lastCommandValue() == Approx( 10.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );

    // Presets near 0/360 match across the wrap.
    app.llPos( 299.7 );
    app.llState( "READY" );
    REQUIRE( app.preset() == Approx( 2.0F ) );
    app.llPos( 301.0 );
    REQUIRE( app.preset() == Approx( 0.0F ) );
}

/// Motion is refused unless the stage is connected and not homing.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl refuses motion when the stage cannot move", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::motionAllowed_( "" );
    // clang-format on
#endif

    elliptecCtrl_test app( "ellctrl-refuse" );
    REQUIRE( app.startup() == 0 );

    REQUIRE( app.requestPreset( 1 ) == 0 );
    REQUIRE( app.absDeg( 10.0 ) == 0 );
    REQUIRE( app.relMove() == 0 );
    REQUIRE( app.pressHome() == 0 );
    REQUIRE( app.lastCommandName().empty() );

    app.llState( "POWEROFF" );
    REQUIRE( app.requestPreset( 1 ) == 0 );
    REQUIRE( app.lastCommandName().empty() );

    // Homing blocks motion until it finishes.
    app.llState( "NOTHOMED" );
    REQUIRE( app.pressHome() == 0 );
    REQUIRE( app.lastCommandName() == "req_home" );
    REQUIRE( app.requestPreset( 1 ) == 0 );
    REQUIRE( app.lastCommandName().empty() );
}

/// Home, the post-home offset, and stop.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl home sequence and stop", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::startHoming();
    MagAOX::app::elliptecCtrl::stop();
    MagAOX::app::elliptecCtrl::sendRequest_( "" );
    // clang-format on
#endif

    using MagAOX::app::stateCodes;

    SECTION( "without an offset the home ends when the low level reports READY" )
    {
        elliptecCtrl_test app( "ellctrl-home" );
        REQUIRE( app.startup() == 0 );
        app.llState( "NOTHOMED" );

        REQUIRE( app.pressHome() == 0 );
        REQUIRE( app.lastCommandName() == "req_home" );
        REQUIRE( app.state() == stateCodes::HOMING );

        app.llState( "HOMING" );
        app.llPos( 0.0 );
        app.llState( "READY" );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.lastCommandName().empty() );
    }

    SECTION( "with an offset the home continues to the offset angle" )
    {
        elliptecCtrl_test app( "ellctrl-home-offset" );
        app.homeOffset( 45.0 );
        REQUIRE( app.startup() == 0 );
        app.llState( "NOTHOMED" );

        REQUIRE( app.pressHome() == 0 );
        app.lastCommandName();
        app.llState( "HOMING" );
        app.llPos( 0.0 );
        app.llState( "READY" );

        REQUIRE( app.lastCommandValue() == Approx( 45.0 ) );
        REQUIRE( app.lastCommandName() == "tgt_pos" );
        REQUIRE( app.state() == stateCodes::HOMING );
        REQUIRE( app.status() == "Homed. Applying offset" );

        // A READY report from before the offset move does not end the home.
        app.llState( "READY" );
        REQUIRE( app.state() == stateCodes::HOMING );

        app.llState( "OPERATING" );
        REQUIRE( app.state() == stateCodes::HOMING );

        app.llPos( 45.1 );
        app.llState( "READY" );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "the offset completes on a position update after READY" )
    {
        elliptecCtrl_test app( "ellctrl-home-offset-pos" );
        app.homeOffset( 45.0 );
        REQUIRE( app.startup() == 0 );
        app.llState( "NOTHOMED" );
        REQUIRE( app.pressHome() == 0 );
        app.llState( "HOMING" );
        app.llPos( 0.0 );
        app.llState( "READY" ); // sends the offset
        app.llState( "READY" ); // stale
        app.llPos( 45.0 );      // arrives
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "a failed home returns to NOTHOMED" )
    {
        elliptecCtrl_test app( "ellctrl-home-fail" );
        REQUIRE( app.startup() == 0 );
        app.llState( "NOTHOMED" );
        REQUIRE( app.pressHome() == 0 );
        app.llState( "HOMING" );
        app.llState( "NOTHOMED" );
        REQUIRE( app.state() == stateCodes::NOTHOMED );

        // A new home can be requested.
        app.lastCommandName();
        REQUIRE( app.pressHome() == 0 );
        REQUIRE( app.lastCommandName() == "req_home" );
    }

    SECTION( "stop sends a halt request" )
    {
        elliptecCtrl_test app( "ellctrl-stop" );
        readyAt( app, 0.0 );
        REQUIRE( app.pressStop() == 0 );
        REQUIRE( app.lastCommandName() == "req_halt" );
    }
}

/// The power-on home runs once after power-on, not after a reconnect.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl power-on home", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::appLogic();
    // clang-format on
#endif

    elliptecCtrl_test app( "ellctrl-power-home" );
    app.powerOnHome( true );
    REQUIRE( app.startup() == 0 );

    app.llState( "POWEROFF" );
    app.llState( "POWERON" );
    app.llState( "NOTCONNECTED" );
    REQUIRE( app.logic() == 0 );
    REQUIRE( app.lastCommandName().empty() );

    app.llState( "NOTHOMED" );
    REQUIRE( app.logic() == 0 );
    REQUIRE( app.lastCommandName() == "req_home" );

    app.llState( "HOMING" );
    app.llState( "READY" );
    REQUIRE( app.logic() == 0 );

    // A communication loss and reconnect does not home again.
    app.llState( "NOTCONNECTED" );
    app.llState( "READY" );
    REQUIRE( app.logic() == 0 );
    REQUIRE( app.lastCommandName().empty() );

    // A failed home leaves NOTHOMED, but does not trigger another automatic home.
    app.llState( "NOTHOMED" );
    REQUIRE( app.logic() == 0 );
    REQUIRE( app.lastCommandName().empty() );

    // The next power cycle homes again.
    app.llState( "POWEROFF" );
    app.llState( "NOTHOMED" );
    REQUIRE( app.logic() == 0 );
    REQUIRE( app.lastCommandName() == "req_home" );
}

/// Configuration: names, home offset wrapping, and preset validation.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl configuration", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::setupConfig();
    MagAOX::app::elliptecCtrl::loadConfigImpl( config );
    MagAOX::app::elliptecCtrl::validatePresets_();
    // clang-format on
#endif

    {
        elliptecCtrl_test app( "ellctrl-config" );
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test.conf",
                                  { "stage", "stage", "presets", "presets" },
                                  { "lowLevelName", "homeOffset", "names", "positions" },
                                  { "elliptecqwp", "-15", "lab, sky", "115, 300" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test.conf" ) == 0 );
        REQUIRE( app.lowLevelName() == "elliptecqwp" );
        REQUIRE( app.stageName() == "ellctrl-config" );
        REQUIRE( app.homeOffsetDeg() == Approx( 345.0 ) );
    }
    {
        elliptecCtrl_test app( "ellctrl-config-close" );
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test_close.conf",
                                  { "stage", "presets", "presets" },
                                  { "stageName", "names", "positions" },
                                  { "qwpX", "a, b", "359.8, 0.5" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test_close.conf" ) == -1 );
        REQUIRE( app.stageName() == "qwpX" );
    }
    {
        elliptecCtrl_test app( "ellctrl-config-tol" );
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test_tol.conf", { "stage" }, { "presetTolerance" }, { "0" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test_tol.conf" ) == -1 );
    }
}

} // namespace elliptecCtrlTest
} // namespace libXWCTest
