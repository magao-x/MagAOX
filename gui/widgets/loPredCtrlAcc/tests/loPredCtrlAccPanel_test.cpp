/** \file loPredCtrlAccPanel_test.cpp
 * \brief Exercise loPredCtrlAccGUI against a socket-free INDI publisher.
 * \ingroup loPredCtrlAccGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>

#include "../../../lib/tests/offlineIndiPublisher.hpp"
#include "../loPredCtrlAccPanel.hpp"

#include <cstring>

namespace libXWCTest
{
namespace loPredCtrlAccGUITest
{
/// \cond TEST_HARNESS
/// Bind the real Qt panel to the shared offline publisher.
class Fixture
{
  public:
    guiTest::OfflinePublisher m_publisher;          ///< Queued property delivery and captured commands.
    xqt::loPredCtrlAccPanel   m_panel;              ///< Real operator widgets.
    int                       m_confirmations{ 0 }; ///< Reset questions asked.
    bool                      m_confirm{ false };   ///< Answer to the reset question.

    /// Register panel subscriptions before any property is sent.
    Fixture();

    /// Deliver a definition or an update through the real publisher routing.
    void publish( const pcf::IndiProperty &property, /**< [in] INDI property. */
                  bool                     definition = false /**< [in] Deliver a definition. */ );

    /// Look up a named Qt control or display.
    template <class Widget>
    Widget *widget( const char *name /**< [in] Object name. */ );

    /// Text of a status row.
    std::string status( const char *key /**< [in] Row key after "status.". */ );

    /// Define every loPredCtrlAcc property in an idle, removed state, as at controller startup.
    void defineAll();

    /// Publish one switch element.
    void
    publishSwitch( const char                          *property, /**< [in] Switch property. */
                   const char                          *element,  /**< [in] Element. */
                   pcf::IndiElement::SwitchStateType    state,    /**< [in] Element state. */
                   pcf::IndiProperty::PropertyStateType propertyState = pcf::IndiProperty::Ok /**< [in] State. */ );

    /// Publish one text element.
    void publishText( const char *property, /**< [in] Text property. */
                      const char *element,  /**< [in] Element. */
                      const char *text /**< [in] Value. */ );
};

Fixture::Fixture()
{
    m_panel.setResetConfirmation(
        [this]()
        {
            ++m_confirmations;
            return m_confirm;
        } );
    m_publisher.addSubscriber( &m_panel );
    QApplication::processEvents();
}

void Fixture::publish( const pcf::IndiProperty &property, bool definition )
{
    if( definition )
    {
        m_publisher.handleDefProperty( property );
    }
    else
    {
        m_publisher.handleSetProperty( property );
    }

    QApplication::processEvents();
}

template <class Widget>
Widget *Fixture::widget( const char *name )
{
    auto *result = m_panel.findChild<Widget *>( QString::fromLatin1( name ) );
    REQUIRE( result );
    return result;
}

std::string Fixture::status( const char *key )
{
    return widget<QLabel>( ( std::string( "status." ) + key ).c_str() )->text().toStdString();
}

void Fixture::publishSwitch( const char                          *property,
                             const char                          *element,
                             pcf::IndiElement::SwitchStateType    state,
                             pcf::IndiProperty::PropertyStateType propertyState )
{
    pcf::IndiProperty update( pcf::IndiProperty::Switch, "loPredCtrlAcc", property );
    update.add( pcf::IndiElement( element, state ) );
    update.setState( propertyState );
    publish( update );
}

void Fixture::publishText( const char *property, const char *element, const char *text )
{
    pcf::IndiProperty update( pcf::IndiProperty::Text, "loPredCtrlAcc", property );
    update.add( pcf::IndiElement( element, text ) );
    publish( update );
}

void Fixture::defineAll()
{
    const auto text = [this]( const char *name, std::vector<std::pair<const char *, const char *>> elements )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Text, "loPredCtrlAcc", name );
        for( const auto &[element, value] : elements )
        {
            property.add( pcf::IndiElement( element, value ) );
        }
        publish( property, true );
    };
    const auto toggle = [this]( const char *name )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Switch, "loPredCtrlAcc", name );
        property.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
        publish( property, true );
    };

    text( "fsm", { { "state", "OPERATING" } } );
    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    mode.add( pcf::IndiElement( "standard", pcf::IndiElement::On ) );
    mode.add( pcf::IndiElement( "accel_assisted", pcf::IndiElement::Off ) );
    publish( mode, true );
    toggle( "learn" );
    toggle( "learn_std" );
    toggle( "predict" );
    for( const char *name : { "reset_model", "zero_dm", "release_tt" } )
    {
        pcf::IndiProperty request( pcf::IndiProperty::Switch, "loPredCtrlAcc", name );
        request.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );
        publish( request, true );
    }

    // As createStandardIndiNumber<float>( m_indiP_gain, "gain", -10, 10, .001, "%0.4f" ).
    pcf::IndiProperty gain( pcf::IndiProperty::Number, "loPredCtrlAcc", "gain" );
    for( const char *element : { "current", "target" } )
    {
        gain.add( pcf::IndiElement( element, 0.3 ) );
        gain[element].setMin( "-10" );
        gain[element].setMax( "10" );
        gain[element].setStep( "0.001" );
        gain[element].setFormat( "%0.4f" );
    }
    publish( gain, true );

    // As createStandardIndiNumber<float>( m_indiP_learnStdExtra, "learn_std_extra", 0, 3600, .1, "%0.1f" ).
    pcf::IndiProperty extra( pcf::IndiProperty::Number, "loPredCtrlAcc", "learn_std_extra" );
    for( const char *element : { "current", "target" } )
    {
        extra.add( pcf::IndiElement( element, 0.0 ) );
        extra[element].setMin( "0" );
        extra[element].setMax( "3600" );
        extra[element].setStep( "0.1" );
        extra[element].setFormat( "%0.1f" );
    }
    publish( extra, true );

    text( "ddspc_control",
          { { "requested_mode", "standard" },
            { "standard_ready", "false" },
            { "assisted_ready", "false" },
            { "learning", "false" },
            { "prediction", "off" },
            { "accel_sync", "false" },
            { "main_loop_tt", "passthrough" },
            { "standard_state", "awaiting_wfs" },
            { "assisted_state", "wfs_stale" } } );
    text( "safety", { { "output_state", "disabled" }, { "dm_write_enabled", "false" }, { "zero_confirmed", "true" } } );
    pcf::IndiProperty performance( pcf::IndiProperty::Number, "loPredCtrlAcc", "ddspc_performance" );
    performance.add( pcf::IndiElement( "learn_overruns", 0.0 ) );
    publish( performance, true );
    pcf::IndiProperty adc( pcf::IndiProperty::Number, "loPredCtrlAcc", "adc_values" );
    adc.add( pcf::IndiElement( "rate_hz", 8000.0 ) );
    adc.add( pcf::IndiElement( "valid", 1.0 ) );
    publish( adc, true );
}
/// \endcond

/** \brief Opening the panel sends nothing, leaves every control disabled without data, and enables only
 * controls whose readbacks have arrived from an OPERATING controller.
 */
TEST_CASE( "loPredCtrlAccGUI opens without commands and enables only from readbacks", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    for( const char *name : { "learn", "learn_std", "predict" } )
    {
        CHECK_FALSE( fixture.widget<QWidget>( name )->isEnabled() );
    }
    for( const char *name :
         { "ddspc_mode.standard", "ddspc_mode.accel_assisted", "reset_model", "zero_dm", "release_tt" } )
    {
        CHECK_FALSE( fixture.widget<QPushButton>( name )->isEnabled() );
    }
    CHECK_FALSE( fixture.widget<QWidget>( "gain.entry" )->isEnabled() );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().toStdString() == "Controller unavailable" );
    CHECK( fixture.status( "fsm.state" ) == "---" );

    fixture.defineAll();
    CHECK( fixture.m_publisher.m_commands.empty() ); // Nothing is sent on open or on receiving state.
    for( const char *name : { "learn", "learn_std", "predict" } )
    {
        CHECK( fixture.widget<QWidget>( name )->isEnabled() );
    }
    CHECK( fixture.widget<QPushButton>( "ddspc_mode.standard" )->isEnabled() );
    CHECK( fixture.widget<QPushButton>( "ddspc_mode.standard" )->isChecked() );
    CHECK_FALSE( fixture.widget<QPushButton>( "ddspc_mode.accel_assisted" )->isChecked() );
    CHECK( fixture.widget<QPushButton>( "reset_model" )->isEnabled() );
    CHECK( fixture.widget<QPushButton>( "zero_dm" )->isEnabled() );
    CHECK( fixture.widget<QPushButton>( "release_tt" )->isEnabled() );
    CHECK( fixture.widget<QWidget>( "gain.entry" )->isEnabled() );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().toStdString() ==
           "PREDICTION OFF: main loop is correcting tip/tilt" );
    CHECK( fixture.status( "mainLoopTT" ) == "passed through (main loop)" );
    CHECK( fixture.status( "fsm.state" ) == "OPERATING" );
    CHECK( fixture.status( "safety.output_state" ) == "disabled" );
    CHECK( fixture.status( "accRate" ) == "8000.0" );
    CHECK( fixture.status( "standard" ) == "not ready (awaiting_wfs)" );

    fixture.publishText( "fsm", "state", "READY" ); // Not OPERATING: nothing may be operated.
    CHECK_FALSE( fixture.widget<QWidget>( "predict" )->isEnabled() );
    CHECK_FALSE( fixture.widget<QPushButton>( "reset_model" )->isEnabled() );
    CHECK_FALSE( fixture.widget<QPushButton>( "zero_dm" )->isEnabled() );
    CHECK( fixture.widget<QLabel>( "warnings" )->text().contains( "not OPERATING" ) );

    fixture.m_panel.close();
    CHECK( fixture.m_publisher.m_commands.empty() ); // Nor on close.
}

/** \brief External changes, as from cursesINDI, move the controls and status to the reported state. */
TEST_CASE( "loPredCtrlAccGUI follows external INDI changes", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    fixture.defineAll();
    auto *learn = fixture.widget<QWidget>( "learn" )->findChild<QSlider *>();
    REQUIRE( learn );
    CHECK( learn->value() == learn->minimum() );
    fixture.publishSwitch( "learn", "toggle", pcf::IndiElement::On );
    CHECK( learn->value() == learn->maximum() );

    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    mode.add( pcf::IndiElement( "standard", pcf::IndiElement::Off ) );
    mode.add( pcf::IndiElement( "accel_assisted", pcf::IndiElement::On ) );
    fixture.publish( mode );
    fixture.publishText( "ddspc_control", "requested_mode", "accel_assisted" );
    CHECK( fixture.widget<QPushButton>( "ddspc_mode.accel_assisted" )->isChecked() );
    CHECK_FALSE( fixture.widget<QPushButton>( "ddspc_mode.standard" )->isChecked() );

    fixture.publishText( "ddspc_control", "assisted_ready", "true" );
    CHECK( fixture.status( "assisted" ) == "ready" );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "zeroed" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().startsWith( "PREDICTION ACTIVE" ) );
    CHECK( fixture.status( "mainLoopTT" ) == "zeroed (DDSPC only)" );

    // Prediction off while the DDSPC zero is unconfirmed: the main loop's tip/tilt is still withheld.
    fixture.publishText( "ddspc_control", "prediction", "off" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().toStdString() ==
           "PREDICTION OFF: tip/tilt is not corrected" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "passthrough" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().toStdString() ==
           "PREDICTION OFF: main loop is correcting tip/tilt" );

    pcf::IndiProperty performance( pcf::IndiProperty::Number, "loPredCtrlAcc", "ddspc_performance" );
    performance.add( pcf::IndiElement( "learn_overruns", 3.0 ) );
    fixture.publish( performance );
    CHECK( fixture.status( "ddspc_performance.learn_overruns" ) == "3" );
    CHECK( fixture.m_publisher.m_commands.empty() );
}

/** \brief Mode changes and resets are offered only with prediction off and the zero confirmed; reset asks first,
 * and every command has the exact property and elements the controller validates.
 */
TEST_CASE( "loPredCtrlAccGUI enforces the controller's mode and reset interlock", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    fixture.defineAll();
    auto *standard = fixture.widget<QPushButton>( "ddspc_mode.standard" );
    auto *assisted = fixture.widget<QPushButton>( "ddspc_mode.accel_assisted" );
    auto *reset    = fixture.widget<QPushButton>( "reset_model" );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    CHECK_FALSE( assisted->isEnabled() );
    CHECK_FALSE( reset->isEnabled() );
    CHECK( fixture.widget<QLabel>( "interlock" )->text().contains( "Turn prediction off" ) );
    assisted->click();
    reset->click();
    CHECK( fixture.m_publisher.m_commands.empty() );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off );
    fixture.publishText( "safety", "zero_confirmed", "false" );
    CHECK_FALSE( assisted->isEnabled() );
    CHECK( fixture.widget<QLabel>( "interlock" )->text().contains( "zero to be confirmed" ) );

    fixture.publishText( "safety", "zero_confirmed", "true" );
    REQUIRE( assisted->isEnabled() );
    assisted->click();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 1 );
    const auto &command = fixture.m_publisher.m_commands.back();
    CHECK( command.getDevice() == "loPredCtrlAcc" );
    CHECK( command.getName() == "ddspc_mode" );
    CHECK( command.getElements().size() == 2 );
    CHECK( command["accel_assisted"].getSwitchState() == pcf::IndiElement::On );
    CHECK( command["standard"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( standard->isChecked() ); // Still the readback until the controller reports the change.
    CHECK_FALSE( assisted->isChecked() );

    fixture.m_confirm = false;
    reset->click();
    CHECK( fixture.m_confirmations == 1 );
    CHECK( fixture.m_publisher.m_commands.size() == 1 ); // Declined: nothing sent.
    fixture.m_confirm = true;
    reset->click();
    REQUIRE( fixture.m_publisher.m_commands.size() == 2 );
    CHECK( fixture.m_publisher.m_commands.back().getName() == "reset_model" );
    CHECK( fixture.m_publisher.m_commands.back()["request"].getSwitchState() == pcf::IndiElement::On );
}

/** \brief Zero and tip/tilt release stay available during prediction, the case they recover from, and each sends
 * exactly its request switch.
 */
TEST_CASE( "loPredCtrlAccGUI sends DM recovery requests at any time", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    fixture.defineAll();
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "safety", "zero_confirmed", "false" );

    for( const char *name : { "zero_dm", "release_tt" } )
    {
        auto *button = fixture.widget<QPushButton>( name );
        REQUIRE( button->isEnabled() );
        button->click();
        QApplication::processEvents();
        const auto &command = fixture.m_publisher.m_commands.back();
        CHECK( command.getDevice() == "loPredCtrlAcc" );
        CHECK( command.getName() == name );
        CHECK( command.getElements().size() == 1 );
        CHECK( command["request"].getSwitchState() == pcf::IndiElement::On );
    }

    CHECK( fixture.m_publisher.m_commands.size() == 2 );
    CHECK( fixture.m_confirmations == 0 ); // Neither asks: both only stop prediction.
}

/** \brief A lost connection or a departed device clears every readback and disables every control. */
TEST_CASE( "loPredCtrlAccGUI disables everything when the controller disappears", "[loPredCtrlAccGUI]" )
{
    for( const bool wholeDevice : { false, true } )
    {
        Fixture fixture;
        fixture.defineAll();
        REQUIRE( fixture.widget<QPushButton>( "reset_model" )->isEnabled() );

        if( wholeDevice )
        {
            pcf::IndiProperty gone;
            gone.setDevice( "loPredCtrlAcc" );
            fixture.m_publisher.handleDelProperty( gone );
        }
        else
        {
            fixture.m_panel.onDisconnect();
        }
        QApplication::processEvents();

        for( const char *name : { "learn", "learn_std", "predict" } )
        {
            CHECK_FALSE( fixture.widget<QWidget>( name )->isEnabled() );
        }
        for( const char *name :
             { "ddspc_mode.standard", "ddspc_mode.accel_assisted", "reset_model", "zero_dm", "release_tt" } )
        {
            CHECK_FALSE( fixture.widget<QPushButton>( name )->isEnabled() );
        }
        CHECK_FALSE( fixture.widget<QWidget>( "gain.entry" )->isEnabled() );
        CHECK( fixture.status( "fsm.state" ) == "---" );
        CHECK( fixture.status( "safety.output_state" ) == "---" );
        CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().toStdString() == "Controller unavailable" );
        CHECK( fixture.m_panel.windowTitle().contains( wholeDevice ? "not running" : "disconnected" ) );
    }
}
/** \brief The banner, Last stop row and warnings say why prediction stopped or why a request was refused. */
TEST_CASE( "loPredCtrlAccGUI explains why prediction stopped", "[loPredCtrlAccGUI]" )
{
    // clang-format off
#ifdef LOPREDCTRLACCGUI_TEST_DOXYGEN_REF
    xqt::loPredCtrlAccPanel::stopReason;
    xqt::loPredCtrlAccPanel::refusalReason;
#endif
    // clang-format on
    Fixture fixture;
    fixture.defineAll();
    auto banner = [&]()
    {
        return fixture.widget<QLabel>( "prediction.banner" )->text().toStdString();
    };
    auto warnings = [&]()
    {
        return fixture.widget<QLabel>( "warnings" )->text();
    };
    auto safety = [&]( const char *outputState, pcf::IndiProperty::PropertyStateType state )
    {
        pcf::IndiProperty update( pcf::IndiProperty::Text, "loPredCtrlAcc", "safety" );
        update.add( pcf::IndiElement( "output_state", outputState ) );
        update.setState( state );
        fixture.publish( update );
    };
    CHECK( fixture.status( "stopReason" ) == "not started since the controller started" );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    safety( "transport_acknowledged_not_physical_application", pcf::IndiProperty::Ok );
    CHECK( fixture.status( "stopReason" ) == "running" );

    // A fault before its zero is copied: the fault code with the condition behind it.
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Alert );
    fixture.publishText( "ddspc_control", "prediction", "off" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "zeroed" );
    safety( "input_or_model_qualification_lost (input_or_model_qualification_lost: acc_window stale_sample)",
            pcf::IndiProperty::Alert );
    const std::string acc = "the ACC window was unusable: no ACC sample within adc.max_age_us "
                            "[input_or_model_qualification_lost: acc_window stale_sample]";
    CHECK( banner() == "PREDICTION STOPPED: " + acc );
    CHECK( fixture.status( "stopReason" ) == acc );
    CHECK( warnings().contains( QString::fromStdString( "Output fault: " + acc ) ) );
    CHECK( warnings().contains( "Prediction stopped by the output fault" ) );

    // After the zero is copied, and once tip/tilt is back with the main loop, the reason stays.
    fixture.publishText( "ddspc_control", "main_loop_tt", "passthrough" );
    safety( "fault_zero_acknowledged (input_or_model_qualification_lost: wfs_stale)", pcf::IndiProperty::Alert );
    CHECK( banner() ==
           "PREDICTION STOPPED: no WFS frame arrived within input_timeout [input_or_model_qualification_lost: "
           "wfs_stale]" );
    safety( "fault_zero_acknowledged (nonfinite_or_command_limit)", pcf::IndiProperty::Alert );
    CHECK( fixture.status( "stopReason" ) ==
           "a DDSPC command was non-finite or beyond its limit [nonfinite_or_command_limit]" );
    safety( "fault_zero_acknowledged", pcf::IndiProperty::Alert );
    CHECK( fixture.status( "stopReason" ).find( "does not report its reason" ) != std::string::npos );

    // An operator stop is not a fault.
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off );
    safety( "disabled_zero_acknowledged", pcf::IndiProperty::Ok );
    CHECK( fixture.status( "stopReason" ).find( "turned off by request" ) == 0 );
    CHECK( banner() == "PREDICTION OFF: main loop is correcting tip/tilt" );
    CHECK( warnings().isEmpty() );

    // A refused request: the condition the controller reports.
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Alert );
    CHECK( warnings().contains(
        "Prediction request refused: the standard model is not ready (awaiting_wfs) and learning is off" ) );
    pcf::IndiProperty integrator( pcf::IndiProperty::Number, "loPredCtrlAcc", "integrator" );
    integrator.add( pcf::IndiElement( "enabled", 1.0 ) );
    integrator.add( pcf::IndiElement( "ready", 0.0 ) );
    fixture.publish( integrator, true );
    CHECK( warnings().contains(
        "Prediction request refused: the main loop's integrator parameters have not been received" ) );
}

/** \brief The extra-learning entry is available only once the controller publishes learn_std_extra, and sends the
 * typed seconds as its target.
 */
TEST_CASE( "loPredCtrlAccGUI sets the extra learning time", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    auto   *entry = fixture.widget<QWidget>( "learn_std_extra.entry" );
    CHECK_FALSE( entry->isEnabled() );

    fixture.defineAll();
    REQUIRE( entry->isEnabled() );
    CHECK( fixture.m_publisher.m_commands.empty() );

    auto *edit = entry->findChild<QLineEdit *>( QStringLiteral( "value" ) );
    REQUIRE( edit );
    edit->setText( QStringLiteral( "2.5" ) );
    emit edit->returnPressed();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 1 );
    const auto &command = fixture.m_publisher.m_commands.back();
    CHECK( command.getDevice() == "loPredCtrlAcc" );
    CHECK( command.getName() == "learn_std_extra" );
    CHECK( command["target"].get() == "2.5" );

    // A controller without the property (an older version) leaves the entry disabled.
    pcf::IndiProperty gone( pcf::IndiProperty::Number, "loPredCtrlAcc", "learn_std_extra" );
    fixture.m_publisher.handleDelProperty( gone );
    QApplication::processEvents();
    CHECK_FALSE( entry->isEnabled() );
}

/** \brief The Main Loop button is offered only by a controller with ddspc_mode.main_loop, sends all three elements,
 * and its banner and refusal reasons follow the main-loop route.
 */
TEST_CASE( "loPredCtrlAccGUI selects the main-loop route", "[loPredCtrlAccGUI]" )
{
    // clang-format off
#ifdef LOPREDCTRLACCGUI_TEST_DOXYGEN_REF
    xqt::loPredCtrlAccPanel::sendMode;
    xqt::loPredCtrlAccPanel::refusalReason;
#endif
    // clang-format on
    Fixture fixture;
    fixture.defineAll();
    auto *mainLoop = fixture.widget<QPushButton>( "ddspc_mode.main_loop" );
    CHECK_FALSE( mainLoop->isEnabled() ); // An older controller without the element.

    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    mode.add( pcf::IndiElement( "standard", pcf::IndiElement::On ) );
    mode.add( pcf::IndiElement( "accel_assisted", pcf::IndiElement::Off ) );
    mode.add( pcf::IndiElement( "main_loop", pcf::IndiElement::Off ) );
    fixture.publish( mode, true ); // A controller with the main-loop route.
    REQUIRE( mainLoop->isEnabled() );
    mainLoop->click();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 1 );
    const auto &command = fixture.m_publisher.m_commands.back();
    CHECK( command.getName() == "ddspc_mode" );
    CHECK( command.getElements().size() == 3 );
    CHECK( command["main_loop"].getSwitchState() == pcf::IndiElement::On );
    CHECK( command["standard"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( command["accel_assisted"].getSwitchState() == pcf::IndiElement::Off );

    fixture.publishSwitch( "ddspc_mode", "standard", pcf::IndiElement::Off );
    fixture.publishSwitch( "ddspc_mode", "main_loop", pcf::IndiElement::On );
    CHECK( mainLoop->isChecked() );
    CHECK_FALSE( fixture.widget<QPushButton>( "ddspc_mode.standard" )->isChecked() );

    // The route needs no integrator parameters, so a refusal names the standard model instead.
    pcf::IndiProperty integrator( pcf::IndiProperty::Number, "loPredCtrlAcc", "integrator" );
    integrator.add( pcf::IndiElement( "enabled", 1.0 ) );
    integrator.add( pcf::IndiElement( "ready", 0.0 ) );
    fixture.publish( integrator, true );
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Alert );
    CHECK( fixture.widget<QLabel>( "warnings" )
               ->text()
               .contains( "Prediction request refused: the standard model is not ready" ) );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "predicted" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text() ==
           "PREDICTION ACTIVE: DDSPC tip/tilt through the main loop" );
}
/** \brief The two mfilt copy buttons are offered only by a controller with their ddspc_mode elements and each sends
 * its own; a refusal names the integrator parameters, never a model, and the banner names the active rate.
 */
TEST_CASE( "loPredCtrlAccGUI selects the mfilt copies", "[loPredCtrlAccGUI]" )
{
    // clang-format off
#ifdef LOPREDCTRLACCGUI_TEST_DOXYGEN_REF
    xqt::loPredCtrlAccPanel::sendMode;
    xqt::loPredCtrlAccPanel::refusalReason;
#endif
    // clang-format on
    Fixture fixture;
    fixture.defineAll();
    auto *copy8k = fixture.widget<QPushButton>( "ddspc_mode.mfilt_copy_8k" );
    auto *copy2k = fixture.widget<QPushButton>( "ddspc_mode.mfilt_copy_2k" );
    CHECK_FALSE( copy8k->isEnabled() ); // An older controller without the elements.
    CHECK_FALSE( copy2k->isEnabled() );

    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    for( const char *element : { "standard", "accel_assisted", "main_loop", "mfilt_copy_8k", "mfilt_copy_2k" } )
    {
        mode.add( pcf::IndiElement(
            element, std::strcmp( element, "standard" ) ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    }
    fixture.publish( mode, true );
    REQUIRE( copy8k->isEnabled() );
    REQUIRE( copy2k->isEnabled() );
    copy2k->click();
    copy8k->click();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 2 );
    const auto &sent2k = fixture.m_publisher.m_commands[0], &sent8k = fixture.m_publisher.m_commands[1];
    CHECK( sent2k.getElements().size() == 5 );
    CHECK( sent2k["mfilt_copy_2k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( sent2k["mfilt_copy_8k"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( sent8k["mfilt_copy_8k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( sent8k["mfilt_copy_2k"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( sent8k["standard"].getSwitchState() == pcf::IndiElement::Off );

    fixture.publishSwitch( "ddspc_mode", "standard", pcf::IndiElement::Off );
    fixture.publishSwitch( "ddspc_mode", "mfilt_copy_2k", pcf::IndiElement::On );
    CHECK( copy2k->isChecked() );
    CHECK_FALSE( copy8k->isChecked() );

    // Integrator parameters are required even with [integrator] disabled; no model is.
    pcf::IndiProperty integrator( pcf::IndiProperty::Number, "loPredCtrlAcc", "integrator" );
    integrator.add( pcf::IndiElement( "enabled", 0.0 ) );
    integrator.add( pcf::IndiElement( "ready", 0.0 ) );
    fixture.publish( integrator, true );
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Alert );
    auto warnings = [&]()
    {
        return fixture.widget<QLabel>( "warnings" )->text();
    };
    CHECK( warnings().contains(
        "Prediction request refused: the main loop's integrator parameters have not been received" ) );
    integrator["ready"].set( 1.0 );
    fixture.publish( integrator );
    CHECK_FALSE( warnings().contains( "model is not ready" ) );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "requested_mode", "mfilt_copy_2k" );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "zeroed" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text() ==
           "MFILT COPY ACTIVE (2 kHz): main-loop tip/tilt integrator on the DDSPC DM channel" );
    fixture.publishText( "ddspc_control", "requested_mode", "mfilt_copy_8k" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text() ==
           "MFILT COPY ACTIVE (8 kHz): main-loop tip/tilt integrator on the DDSPC DM channel" );
}

/** \brief The two mfilt clone buttons are offered only by a controller with their ddspc_mode elements and each sends
 * its own, with every element the controller publishes; the banner names the rate and the step-0 sync counts, which
 * also have their own status row.
 */
TEST_CASE( "loPredCtrlAccGUI selects the mfilt clones", "[loPredCtrlAccGUI]" )
{
    // clang-format off
#ifdef LOPREDCTRLACCGUI_TEST_DOXYGEN_REF
    xqt::loPredCtrlAccPanel::sendMode;
    xqt::loPredCtrlAccPanel::refresh;
#endif
    // clang-format on
    Fixture fixture;
    fixture.defineAll();
    auto *clone8k = fixture.widget<QPushButton>( "ddspc_mode.mfilt_clone_8k" );
    auto *clone2k = fixture.widget<QPushButton>( "ddspc_mode.mfilt_clone_2k" );
    CHECK_FALSE( clone8k->isEnabled() ); // An older controller without the elements.
    CHECK_FALSE( clone2k->isEnabled() );

    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    for( const char *element : { "standard",
                                 "accel_assisted",
                                 "main_loop",
                                 "mfilt_copy_8k",
                                 "mfilt_copy_2k",
                                 "mfilt_clone_8k",
                                 "mfilt_clone_2k" } )
    {
        mode.add( pcf::IndiElement(
            element, std::strcmp( element, "standard" ) ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    }
    fixture.publish( mode, true );
    REQUIRE( clone8k->isEnabled() );
    REQUIRE( clone2k->isEnabled() );
    clone2k->click();
    clone8k->click();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 2 );
    const auto &sent2k = fixture.m_publisher.m_commands[0], &sent8k = fixture.m_publisher.m_commands[1];
    CHECK( sent2k.getElements().size() == 7 );
    CHECK( sent2k["mfilt_clone_2k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( sent2k["mfilt_copy_2k"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( sent2k["mfilt_clone_8k"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( sent8k["mfilt_clone_8k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( sent8k["mfilt_clone_2k"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( sent8k["standard"].getSwitchState() == pcf::IndiElement::Off );

    fixture.publishSwitch( "ddspc_mode", "standard", pcf::IndiElement::Off );
    fixture.publishSwitch( "ddspc_mode", "mfilt_clone_2k", pcf::IndiElement::On );
    CHECK( clone2k->isChecked() );
    CHECK_FALSE( clone8k->isChecked() );

    pcf::IndiProperty clone( pcf::IndiProperty::Number, "loPredCtrlAcc", "mfilt_clone" );
    clone.add( pcf::IndiElement( "step0_in_sync", 1990.0 ) );
    clone.add( pcf::IndiElement( "step0_missed", 3.0 ) );
    fixture.publish( clone, true );
    const auto inSync = QString::fromStdString( clone["step0_in_sync"].get() );
    const auto missed = QString::fromStdString( clone["step0_missed"].get() );
    CHECK( QString::fromStdString( fixture.status( "cloneStep0" ) ) == inSync + " / " + missed );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "requested_mode", "mfilt_clone_2k" );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "zeroed" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text() ==
           "MFILT CLONE ACTIVE (2 kHz): mfilt's tip/tilt state on the DDSPC DM channel; step 0 in sync " + inSync +
               ", missed " + missed );
    fixture.publishText( "ddspc_control", "requested_mode", "mfilt_clone_8k" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text().startsWith( "MFILT CLONE ACTIVE (8 kHz)" ) );
}
/** \brief Failed and discarded DM writes, which never stop prediction, are counted and warned about while they rise;
 * the detected DM slots and ACC ratio are shown; a dm_slots mismatch or an input hold names why assisted is not
 * writing.
 */
TEST_CASE( "loPredCtrlAccGUI shows DM write failures and detected rates", "[loPredCtrlAccGUI]" )
{
    // clang-format off
#ifdef LOPREDCTRLACCGUI_TEST_DOXYGEN_REF
    xqt::loPredCtrlAccPanel::ratesText;
    xqt::loPredCtrlAccPanel::rising;
#endif
    // clang-format on
    Fixture fixture;
    fixture.defineAll();
    auto banner = [&]()
    {
        return fixture.widget<QLabel>( "prediction.banner" )->text().toStdString();
    };
    auto warnings = [&]()
    {
        return fixture.widget<QLabel>( "warnings" )->text();
    };
    auto performance = [&]( double failures, double discarded, double mismatch )
    {
        pcf::IndiProperty update( pcf::IndiProperty::Number, "loPredCtrlAcc", "ddspc_performance" );
        update.add( pcf::IndiElement( "dm_write_failures", failures ) );
        update.add( pcf::IndiElement( "transport_discarded", discarded ) );
        update.add( pcf::IndiElement( "step_mismatch", mismatch ) );
        fixture.publish( update );
    };
    auto rates = [&]( pcf::IndiProperty::PropertyStateType state )
    {
        pcf::IndiProperty update( pcf::IndiProperty::Number, "loPredCtrlAcc", "rates" );
        update.add( pcf::IndiElement( "dm_slots", 4.0 ) );
        update.add( pcf::IndiElement( "acc_ratio", 4.0 ) );
        update.add( pcf::IndiElement( "scheduler_steps", 4.0 ) );
        update.add( pcf::IndiElement( "acc_per_wfs", 3.99 ) );
        update.setState( state );
        fixture.publish( update, true );
    };

    // A controller without these readbacks.
    CHECK( fixture.status( "rates" ) == "---" );
    CHECK( fixture.status( "ddspc_performance.dm_write_failures" ) == "---" );

    // The first counts are a baseline, not a rise.
    performance( 2, 5, 0 );
    CHECK( fixture.status( "ddspc_performance.dm_write_failures" ) == "2" );
    CHECK( fixture.status( "ddspc_performance.transport_discarded" ) == "5" );
    CHECK( warnings().isEmpty() );

    // Rising counts warn; prediction continues, so nothing is a fault.
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "safety", "output_state", "write_failed_scheduler_reset" );
    performance( 3, 9, 0 );
    CHECK( warnings().contains( "DM writes failing (3 so far)" ) );
    CHECK( warnings().contains( "DM scheduler discarding DDSPC commands (9 so far)" ) );
    CHECK_FALSE( warnings().contains( "Output fault" ) );
    CHECK( fixture.status( "safety.output_state" ) ==
           "a DM write failed: the DM scheduler (dmtweeter-synco) restarted [write_failed_scheduler_reset]" );
    CHECK( fixture.status( "stopReason" ) == "running" );
    CHECK( banner() == "PREDICTION ACTIVE: DDSPC is correcting tip/tilt" );

    // Detected rates, and the detection state.
    rates( pcf::IndiProperty::Busy );
    CHECK( fixture.status( "rates" ) == "detecting: 4 / 4 (scheduler 4 steps, 3.99 ACC per WFS)" );
    CHECK( warnings().contains( "DM slots / ACC ratio not yet confirmed" ) );
    rates( pcf::IndiProperty::Ok );
    CHECK( fixture.status( "rates" ) == "4 / 4 (scheduler 4 steps, 3.99 ACC per WFS)" );
    CHECK_FALSE( warnings().contains( "DM slots / ACC ratio" ) );

    // Why accepted assisted prediction is not writing.
    performance( 3, 9, 7 );
    CHECK( warnings().contains( "ACC-assisted slots not written" ) );
    fixture.publishText( "ddspc_control", "prediction", "dm_slots_4_differ_from_scheduler_steps_1" );
    CHECK( banner() == "PREDICTION ON: ACC-assisted not writing: dm_slots 4 differs from the DM scheduler's 1 steps "
                       "(set dm_slots = 0)" );
    fixture.publishText( "ddspc_control", "prediction", "holding_acc_window" );
    CHECK( banner() == "PREDICTION ON: holding the last DM command (the ACC window was unusable)" );

    // A refused request while the rates are detected.
    rates( pcf::IndiProperty::Busy );
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Alert );
    CHECK( warnings().contains(
        "Prediction request refused: dm_slots and acc_ratio are still being detected from the running rates" ) );
}
/** \brief The ACC Assisted 2 kHz button is offered only by a controller with accel_assisted_2k, sends it with every
 * other element off, follows the readback, and the banner names it while active.
 */
TEST_CASE( "loPredCtrlAccGUI selects ACC-assisted 2 kHz", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    fixture.defineAll();
    auto *assisted2k = fixture.widget<QPushButton>( "ddspc_mode.accel_assisted_2k" );
    CHECK_FALSE( assisted2k->isEnabled() ); // An older controller without the element.

    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    for( const char *element : { "standard", "accel_assisted", "accel_assisted_2k" } )
    {
        mode.add( pcf::IndiElement(
            element, std::strcmp( element, "standard" ) ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    }
    fixture.publish( mode, true );
    REQUIRE( assisted2k->isEnabled() );
    assisted2k->click();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 1 );
    const auto &sent = fixture.m_publisher.m_commands[0];
    CHECK( sent.getElements().size() == 3 );
    CHECK( sent["accel_assisted_2k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( sent["accel_assisted"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( sent["standard"].getSwitchState() == pcf::IndiElement::Off );

    fixture.publishSwitch( "ddspc_mode", "standard", pcf::IndiElement::Off );
    fixture.publishSwitch( "ddspc_mode", "accel_assisted_2k", pcf::IndiElement::On );
    CHECK( assisted2k->isChecked() );
    CHECK_FALSE( fixture.widget<QPushButton>( "ddspc_mode.accel_assisted" )->isChecked() );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "requested_mode", "accel_assisted_2k" );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "zeroed" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text() ==
           "PREDICTION ACTIVE: ACC-assisted DDSPC, one command per WFS frame" );
}
/** \brief The ACC rate entry follows the accelerometer device's fps, independently of loPredCtrlAcc, and sends the
 * typed rate as that device's fps target.
 */
TEST_CASE( "loPredCtrlAccGUI sets the accelerometer rate", "[loPredCtrlAccGUI]" )
{
    Fixture fixture;
    auto   *entry = fixture.widget<QWidget>( "acc.fps.entry" );
    CHECK_FALSE( entry->isEnabled() );

    // As mcp3208Ctrl: CREATE_REG_INDI_NEW_NUMBERF( m_indiP_fps, "fps", 0, 10000, 1, "%d", "", "" ).
    pcf::IndiProperty fps( pcf::IndiProperty::Number, "acc000", "fps" );
    for( const char *element : { "current", "target" } )
    {
        fps.add( pcf::IndiElement( element, 8000 ) );
        fps[element].setMin( "0" );
        fps[element].setMax( "10000" );
        fps[element].setStep( "1" );
        fps[element].setFormat( "%d" );
    }
    fixture.publish( fps, true );
    REQUIRE( entry->isEnabled() ); // Without loPredCtrlAcc running.
    CHECK( fixture.m_publisher.m_commands.empty() );

    auto *edit = entry->findChild<QLineEdit *>( QStringLiteral( "value" ) );
    REQUIRE( edit );
    edit->setText( QStringLiteral( "2000" ) );
    emit edit->returnPressed();
    QApplication::processEvents();
    REQUIRE( fixture.m_publisher.m_commands.size() == 1 );
    const auto &command = fixture.m_publisher.m_commands.back();
    CHECK( command.getDevice() == "acc000" );
    CHECK( command.getName() == "fps" );
    CHECK( command["target"].get() == "2000" );

    // loPredCtrlAcc stopping or restarting leaves the accelerometer's entry: acc000 publishes fps only on a change.
    fixture.defineAll();
    pcf::IndiProperty controller( pcf::IndiProperty::Text, "loPredCtrlAcc", "" );
    fixture.m_publisher.handleDelProperty( controller );
    QApplication::processEvents();
    CHECK( entry->isEnabled() );
    CHECK( edit->text().toStdString() != "---" );

    // The accelerometer stops: the entry is disabled.
    pcf::IndiProperty gone( pcf::IndiProperty::Number, "acc000", "" );
    fixture.m_publisher.handleDelProperty( gone );
    QApplication::processEvents();
    CHECK_FALSE( entry->isEnabled() );
}
/** \brief The averaged mode sets the accelerometer to 8 kHz averaged by 4 and selects ACC Assisted 2 kHz; any other
 * mode switches the averaging off; the buttons, status row and banner follow the accelerometer's averaging.
 */
TEST_CASE( "loPredCtrlAccGUI selects ACC-assisted 2 kHz with 8 kHz averaged samples", "[loPredCtrlAccGUI]" )
{
    // clang-format off
#ifdef LOPREDCTRLACCGUI_TEST_DOXYGEN_REF
    xqt::loPredCtrlAccPanel::sendMode;
    xqt::loPredCtrlAccPanel::sendAccTarget;
#endif
    // clang-format on
    Fixture fixture;
    fixture.defineAll();
    auto *averaged   = fixture.widget<QPushButton>( "mode.accel_assisted_2k_avg" );
    auto *assisted2k = fixture.widget<QPushButton>( "ddspc_mode.accel_assisted_2k" );
    auto &commands   = fixture.m_publisher.m_commands;

    pcf::IndiProperty mode( pcf::IndiProperty::Switch, "loPredCtrlAcc", "ddspc_mode" );
    for( const char *element : { "standard", "accel_assisted", "accel_assisted_2k" } )
    {
        mode.add( pcf::IndiElement(
            element, std::strcmp( element, "standard" ) ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    }
    fixture.publish( mode, true );
    CHECK_FALSE( averaged->isEnabled() ); // No averaging accelerometer yet.
    CHECK( fixture.status( "accAverage" ) == "---" );

    auto accNumber = [&]( const char *name, double value )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Number, "acc000", name );
        property.add( pcf::IndiElement( "current", value ) );
        property.add( pcf::IndiElement( "target", value ) );
        fixture.publish( property, true );
    };
    accNumber( "fps", 2000 );
    accNumber( "average", 1 );
    accNumber( "stream_fps", 2000 );
    REQUIRE( averaged->isEnabled() );
    CHECK( fixture.status( "accAverage" ) == "off (every scan)" );

    averaged->click();
    QApplication::processEvents();
    REQUIRE( commands.size() == 3 );
    CHECK( commands[0].getDevice() == "acc000" );
    CHECK( commands[0].getName() == "fps" );
    CHECK( commands[0]["target"].get<double>() == 8000 );
    CHECK( commands[1].getName() == "average" );
    CHECK( commands[1]["target"].get<double>() == 4 );
    CHECK( commands[2].getName() == "ddspc_mode" );
    CHECK( commands[2]["accel_assisted_2k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( commands[2]["standard"].getSwitchState() == pcf::IndiElement::Off );

    // The readbacks: 8 kHz scans averaged by 4, ACC Assisted 2 kHz selected.
    accNumber( "fps", 8000 );
    accNumber( "average", 4 );
    accNumber( "stream_fps", 2000 );
    fixture.publishSwitch( "ddspc_mode", "standard", pcf::IndiElement::Off );
    fixture.publishSwitch( "ddspc_mode", "accel_assisted_2k", pcf::IndiElement::On );
    CHECK( averaged->isChecked() );
    CHECK_FALSE( assisted2k->isChecked() );
    CHECK( fixture.status( "accAverage" ) == "4 scans per sample (2000 Hz stream)" );

    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::On );
    fixture.publishText( "ddspc_control", "requested_mode", "accel_assisted_2k" );
    fixture.publishText( "ddspc_control", "prediction", "active" );
    fixture.publishText( "ddspc_control", "main_loop_tt", "zeroed" );
    CHECK( fixture.widget<QLabel>( "prediction.banner" )->text() ==
           "PREDICTION ACTIVE: ACC-assisted DDSPC, one command per WFS frame, ACC averaged 4 scans per sample" );

    // Another mode, here the plain ACC Assisted 2 kHz, switches the averaging off.
    fixture.publishSwitch( "predict", "toggle", pcf::IndiElement::Off );
    fixture.publishText( "ddspc_control", "prediction", "off" );
    REQUIRE( assisted2k->isEnabled() );
    assisted2k->click();
    QApplication::processEvents();
    REQUIRE( commands.size() == 5 );
    CHECK( commands[3].getDevice() == "acc000" );
    CHECK( commands[3].getName() == "average" );
    CHECK( commands[3]["target"].get<double>() == 1 );
    CHECK( commands[4]["accel_assisted_2k"].getSwitchState() == pcf::IndiElement::On );
}
} // namespace loPredCtrlAccGUITest
} // namespace libXWCTest

/// Run the Catch session inside a Qt application.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in] Arguments. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
