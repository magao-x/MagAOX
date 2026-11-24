/** \file elliptecCtrl_test.cpp
 * \brief Catch2 tests for the elliptecCtrl client of elliptecLowLevel.
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
                                     pcf::IndiProperty::Type type /**< [in] property type */ )
    {
        pcf::IndiProperty ip( type );
        ip.setDevice( "ellbus" );
        ip.setName( name );
        ip.add( pcf::IndiElement( stage ) );
        return ip;
    }

    /// Report a low-level state.
    int llState( const std::string &st, /**< [in] low-level state string */
                 const std::string &stage = "qwp1" /**< [in] stage element to report for */ )
    {
        pcf::IndiProperty ip = llProp( "curr_state", stage, pcf::IndiProperty::Text );
        ip[stage].set( st );
        return setCallBack_m_indiP_llState( ip );
    }

    /// Report a low-level position.
    int llPos( double             deg, /**< [in] position in degrees */
               const std::string &stage = "qwp1" /**< [in] stage element to report for */ )
    {
        pcf::IndiProperty ip = llProp( "curr_pos", stage, pcf::IndiProperty::Number );
        ip[stage].set( deg );
        return setCallBack_m_indiP_llPos( ip );
    }

    /// Report a low-level error.
    int llError( const std::string &err /**< [in] error text, empty when OK */ )
    {
        pcf::IndiProperty ip = llProp( "last_error", "qwp1", pcf::IndiProperty::Text );
        ip["qwp1"].set( err );
        return setCallBack_m_indiP_llError( ip );
    }

    /// Request a preset by number.
    int requestPreset( float n /**< [in] 1-based preset number */ )
    {
        pcf::IndiProperty ip = m_indiP_preset;
        ip["target"].set( n );
        return newCallBack_m_indiP_preset( ip );
    }

    /// Request a preset by name.
    int requestPresetName( const std::string &name /**< [in] preset name to select */ )
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

    /// Request a position in degrees.
    int position( double deg /**< [in] target in degrees */ )
    {
        pcf::IndiProperty ip = m_indiP_position;
        ip["target"].set( deg );
        return newCallBack_m_indiP_position( ip );
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

    /// Set `stage.homePreset`.
    void homePreset( int n /**< [in] preset number (1 is the first preset), 0 or negative to disable */ )
    {
        m_homePreset = n;
    }

    /// Set `stage.powerOnHome`.
    void powerOnHome( bool on /**< [in] whether to home after power-on */ )
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

    /// Target preset (1-based, 0 for none).
    float presetTarget() const
    {
        return m_preset_target;
    }

    /// Read a configuration file and load it.
    int loadConfigFile( const std::string &path /**< [in] configuration file path */ )
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

    /// Configured home preset.
    int homePresetIndex() const
    {
        return m_homePreset;
    }

    /// Preset names published in the `presetName` switch vector.
    const std::vector<std::string> &presetNames() const
    {
        return m_presetNames;
    }

    /// Whether the `presetName` switch vector has an element.
    bool hasPresetNameElement( const std::string &name /**< [in] element name */ ) const
    {
        return m_indiP_presetName.find( name );
    }

    /// Preset name that `stdMotionStage` would switch on for the current position.
    std::string activeName()
    {
        return activePresetName( static_cast<int>( presetNumber() ) );
    }
};

/// Start a client and bring its stage to READY at the given position.
void readyAt( elliptecCtrl_test &app, /**< [in,out] the client under test */
              double             deg /**< [in] position in degrees */ )
{
    REQUIRE( app.startup() == 0 );
    app.llPos( deg );
    app.llState( "READY" );
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
}

} // namespace
/// \endcond

/// Low-level states map onto the client FSM.
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
    // clang-format on
#endif

    using MagAOX::app::stateCodes;

    elliptecCtrl_test app( "ellctrl-states" );
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );

    REQUIRE( app.llState( "POWEROFF" ) == 0 );
    REQUIRE( app.state() == stateCodes::POWEROFF );

    app.llState( "POWERON" );
    REQUIRE( app.state() == stateCodes::POWERON );

    app.llState( "NODEVICE" );
    REQUIRE( app.state() == stateCodes::NODEVICE );

    app.llState( "NOTHOMED" );
    REQUIRE( app.state() == stateCodes::NOTHOMED );
    REQUIRE( app.moving() == -1 );

    app.llState( "OPERATING" );
    REQUIRE( app.state() == stateCodes::OPERATING );

    app.llState( "HOMING" );
    REQUIRE( app.state() == stateCodes::HOMING );

    app.llState( "READY" );
    REQUIRE( app.state() == stateCodes::READY );

    // Errors are logged but do not change the state.
    REQUIRE( app.llError( "move failed (stopped away from target)" ) == 0 );
    REQUIRE( app.llError( "" ) == 0 );
    REQUIRE( app.state() == stateCodes::READY );

    app.llState( "SOMETHING" );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );

    // Updates for other stages on the bus are ignored.
    app.llState( "READY" );
    REQUIRE( app.llState( "POWEROFF", "qwp2" ) == 0 );
    REQUIRE( app.llPos( 42.0, "qwp2" ) == 0 );
    REQUIRE( app.state() == stateCodes::READY );
}

/// Presets and position requests are sent as absolute targets in degrees.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl sends absolute targets to the low level", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::moveTo( 0 );
    MagAOX::app::elliptecCtrl::newCallBack_m_indiP_position( ip );
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
    REQUIRE( app.presetTarget() == Approx( 1.0F ) );

    app.llPos( 115.2 );
    app.llState( "READY" );
    REQUIRE( app.preset() == Approx( 1.0F ) );

    REQUIRE( app.requestPresetName( "sky" ) == 0 );
    REQUIRE( app.lastCommandValue() == Approx( 300.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );
    REQUIRE( app.presetTarget() == Approx( 2.0F ) );

    // Position targets outside [0, 360) wrap.
    REQUIRE( app.position( 400.0 ) == 0 );
    REQUIRE( app.lastCommandValue() == Approx( 40.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );

    REQUIRE( app.position( -10.0 ) == 0 );
    REQUIRE( app.lastCommandValue() == Approx( 350.0 ) );
    REQUIRE( app.lastCommandName() == "tgt_pos" );

    // Presets near 0/360 match across the wrap.
    app.llPos( 299.7 );
    app.llState( "READY" );
    REQUIRE( app.preset() == Approx( 2.0F ) );
    app.llPos( 301.0 );
    REQUIRE( app.preset() == Approx( 0.0F ) );
}

/// A `none` preset is published, reported away from every preset, and does not move the stage.
/** \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl publishes a none preset", "[elliptecCtrl]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::appStartup();
    MagAOX::app::elliptecCtrl::presetNumber();
    MagAOX::app::elliptecCtrl::presetIndexForPosition_( 0 );
    // clang-format on
#endif

    elliptecCtrl_test app( "ellctrl-none" );
    readyAt( app, 10.0 );

    REQUIRE( app.presetNames() == std::vector<std::string>{ "none", "lab", "sky" } );
    REQUIRE( app.hasPresetNameElement( "none" ) );

    // Away from every preset, including 359 degrees (the "none" placeholder position is -1).
    REQUIRE( app.preset() == Approx( 0.0F ) );
    REQUIRE( app.activeName() == "none" );
    app.llPos( 359.0 );
    REQUIRE( app.activeName() == "none" );

    app.llPos( 115.0 );
    REQUIRE( app.preset() == Approx( 1.0F ) );
    REQUIRE( app.activeName() == "lab" );

    // Selecting "none", by name or as preset 0, does not move the stage.
    REQUIRE( app.requestPresetName( "none" ) == 0 );
    REQUIRE( app.lastCommandName().empty() );
    REQUIRE( app.requestPreset( 0 ) == 0 );
    REQUIRE( app.lastCommandName().empty() );
    REQUIRE( app.activeName() == "lab" );
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
    REQUIRE( app.position( 10.0 ) == 0 );
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

/// Home, the post-home preset, and stop.
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

    SECTION( "without a home preset the home ends when the low level reports READY" )
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

    SECTION( "with a home preset the home is followed by a preset move" )
    {
        elliptecCtrl_test app( "ellctrl-home-preset" );
        app.homePreset( 2 );
        REQUIRE( app.startup() == 0 );
        app.llState( "NOTHOMED" );

        REQUIRE( app.pressHome() == 0 );
        app.lastCommandName();
        app.llState( "HOMING" );
        app.llPos( 0.0 );
        app.llState( "READY" );

        REQUIRE( app.lastCommandValue() == Approx( 300.0 ) );
        REQUIRE( app.lastCommandName() == "tgt_pos" );
        REQUIRE( app.state() == stateCodes::OPERATING );

        // A later READY does not move again.
        app.llPos( 300.0 );
        app.llState( "READY" );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.preset() == Approx( 2.0F ) );
        REQUIRE( app.lastCommandName().empty() );
    }

    SECTION( "a home preset of 0 (none) does not move after the home" )
    {
        elliptecCtrl_test app( "ellctrl-home-none" );
        app.homePreset( 0 );
        REQUIRE( app.startup() == 0 );
        app.llState( "NOTHOMED" );

        REQUIRE( app.pressHome() == 0 );
        app.lastCommandName();
        app.llState( "HOMING" );
        app.llPos( 0.0 );
        app.llState( "READY" );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE( app.lastCommandName().empty() );
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

/// Configuration: names, home preset, and preset validation.
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
                                  { "lowLevelName", "homePreset", "names", "positions" },
                                  { "elliptecqwp", "1", "lab, sky", "115, 300" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test.conf" ) == 0 );
        REQUIRE( app.lowLevelName() == "elliptecqwp" );
        REQUIRE( app.stageName() == "ellctrl-config" );
        REQUIRE( app.homePresetIndex() == 1 );
    }
    {
        elliptecCtrl_test app( "ellctrl-config-homepreset" );
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test_homepreset.conf",
                                  { "stage", "presets", "presets" },
                                  { "homePreset", "names", "positions" },
                                  { "3", "lab, sky", "115, 300" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test_homepreset.conf" ) == -1 );
    }
    {
        elliptecCtrl_test app( "ellctrl-config-homepreset-last" );
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test_homepreset_last.conf",
                                  { "stage", "presets", "presets" },
                                  { "homePreset", "names", "positions" },
                                  { "2", "lab, sky", "115, 300" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test_homepreset_last.conf" ) == 0 );
        REQUIRE( app.homePresetIndex() == 2 );
    }
    {
        elliptecCtrl_test app( "ellctrl-config-mismatch" );
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test_mismatch.conf",
                                  { "presets", "presets" },
                                  { "names", "positions" },
                                  { "lab, sky", "115" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecCtrl_test_mismatch.conf" ) == -1 );
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
