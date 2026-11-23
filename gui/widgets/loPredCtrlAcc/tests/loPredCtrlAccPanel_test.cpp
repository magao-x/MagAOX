/** \file loPredCtrlAccPanel_test.cpp
 * \brief Exercise loPredCtrlAccGUI against a socket-free INDI publisher.
 * \ingroup loPredCtrlAccGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

#include "../../../lib/tests/offlineIndiPublisher.hpp"
#include "../loPredCtrlAccPanel.hpp"

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
} // namespace loPredCtrlAccGUITest
} // namespace libXWCTest

/// Run the Catch session inside a Qt application.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in] Arguments. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
