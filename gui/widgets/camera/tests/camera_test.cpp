/** \file camera_test.cpp
 * \brief Offline discovery, lifecycle, and command tests for the camera EM-gain-off button.
 * \ingroup cameraGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include "../../../lib/tests/offlineIndiPublisher.hpp"
#include "../camera.hpp"

namespace libXWCTest
{
namespace cameraGUITest
{
/// \cond TEST_HARNESS
/// Drain queued subscription and widget creation signals without running a transport.
void events();

/// Own the real camera panel and socket-free INDI publisher.
class Widget
{
  public:
    std::string m_device{ "camera_test" }; ///< Camera device selected for all requests.

    guiTest::OfflinePublisher m_publisher; ///< Publisher capturing commands instead of sending them.

    xqt::camera m_widget; ///< Actual shared camera widget under test.

    /// Register and display the real panel with the socket-free publisher.
    Widget();

    /// Deliver one current property definition through queued dispatch.
    void publish( const pcf::IndiProperty &property /**< [in] Property definition. */ );

    /// Connect the local publisher and deliver the camera FSM.
    void ready( const std::string &state = "READY" /**< [in] Reported camera FSM state. */ );

    /// Deliver a camera FSM update.
    void state( const std::string &state /**< [in] Reported camera FSM state. */ );

    /// Advertise the standard EM-gain-off request.
    void support();

    /// Delete one camera property or the complete device.
    void erase( const std::string &name /**< [in] Property name; empty deletes the device. */ );

    /// Retrieve the optional command button, or nullptr before discovery.
    QPushButton *button();
};

void events()
{
    for( int depth = 0; depth < 4; ++depth )
        QApplication::processEvents();
}

Widget::Widget() : m_widget( m_device )
{
    m_publisher.addSubscriber( &m_widget );
    m_widget.show();
    events();
}

void Widget::publish( const pcf::IndiProperty &property )
{
    m_publisher.handleDefProperty( property );
    events();
}

void Widget::ready( const std::string &value )
{
    m_publisher.onConnect();
    events();
    state( value );
}

void Widget::state( const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_device, "fsm" );
    property.add( pcf::IndiElement( "state", value ) );
    publish( property );
}

void Widget::support()
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch, m_device, "emgain_off" );
    property.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );
    publish( property );
}

void Widget::erase( const std::string &name )
{
    pcf::IndiProperty property;
    property.setDevice( m_device );
    property.setName( name );
    m_publisher.handleDelProperty( property );
    events();
}

QPushButton *Widget::button()
{
    return m_widget.findChild<QPushButton *>( "emGainOff" );
}
/// \endcond

/// Only cameras advertising the standard request expose a button; one press sends that request without choosing gain.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera EM-gain-off discovery and command", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::handleDefProperty; xqt::camera::setup_emGainOff; xqt::camera::emGainOff;
    #endif
    // clang-format on
    Widget widget;
    widget.ready();
    REQUIRE( widget.button() == nullptr );
    widget.m_widget.emGainOff();
    REQUIRE( widget.m_publisher.m_commands.empty() );

    pcf::IndiProperty wrongType( pcf::IndiProperty::Text, widget.m_device, "emgain_off" );
    wrongType.add( pcf::IndiElement( "request", "Off" ) );
    widget.publish( wrongType );
    REQUIRE( widget.button() == nullptr );

    widget.support();
    auto *button = widget.button();
    REQUIRE( button != nullptr );
    REQUIRE( button->isVisible() );
    REQUIRE( button->isEnabled() );
    REQUIRE( button->text() == "EM gain off" );
    widget.support();
    REQUIRE( widget.m_widget.findChildren<QPushButton *>( "emGainOff" ).size() == 1 );
    REQUIRE( widget.m_publisher.m_commands.empty() );
    button->click();
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
    const auto &command = widget.m_publisher.m_commands.back();
    REQUIRE( command.getDevice() == widget.m_device );
    REQUIRE( command.getName() == "emgain_off" );
    REQUIRE( command.getType() == pcf::IndiProperty::Switch );
    REQUIRE( command.getNumElements() == 1 );
    REQUIRE( command["request"].getSwitchState() == pcf::IndiElement::On );
}

/// Property and device deletion hide the control until capability is rediscovered.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera EM-gain-off capability removal", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::handleDelProperty; xqt::camera::handleSetProperty; xqt::camera::emGainOff;
    #endif
    // clang-format on
    Widget widget;
    widget.ready();
    widget.support();
    std::string deleted;
    SECTION( "only the request property is removed" )
    {
        deleted = "emgain_off";
    }
    SECTION( "the entire camera disappears" )
    {
        deleted = "";
    }
    widget.erase( deleted );
    REQUIRE_FALSE( widget.button()->isVisible() );
    REQUIRE_FALSE( widget.button()->isEnabled() );
    widget.m_widget.emGainOff();
    REQUIRE( widget.m_publisher.m_commands.empty() );
    widget.state( "OPERATING" );
    REQUIRE_FALSE( widget.button()->isVisible() );
    widget.support();
    REQUIRE( widget.button()->isVisible() );
    REQUIRE( widget.button()->isEnabled() );
    widget.button()->click();
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
}

/// Connection loss and unavailable camera states block both button clicks and direct slot invocation.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera EM-gain-off readiness and reconnect", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::onConnect; xqt::camera::onDisconnect; xqt::camera::updateGUI; xqt::camera::emGainOff;
    #endif
    // clang-format on
    Widget widget;
    widget.ready();
    widget.support();
    for( const auto *state : { "NODEVICE", "NOTCONNECTED", "CONNECTED", "POWEROFF", "ERROR" } )
    {
        CAPTURE( state );
        widget.state( state );
        REQUIRE_FALSE( widget.button()->isEnabled() );
        widget.m_widget.emGainOff();
        REQUIRE( widget.m_publisher.m_commands.empty() );
    }
    for( const auto *state : { "READY", "OPERATING", "CONFIGURING" } )
    {
        widget.state( state );
        REQUIRE( widget.button()->isEnabled() );
    }
    widget.m_publisher.onDisconnect();
    events();
    REQUIRE_FALSE( widget.button()->isVisible() );
    widget.m_widget.emGainOff();
    REQUIRE( widget.m_publisher.m_commands.empty() );
    widget.m_publisher.onConnect();
    events();
    REQUIRE_FALSE( widget.button()->isVisible() );
    widget.support();
    REQUIRE( widget.button()->isVisible() );
    REQUIRE_FALSE( widget.button()->isEnabled() );
    widget.m_widget.emGainOff();
    REQUIRE( widget.m_publisher.m_commands.empty() );
    widget.state( "READY" );
    REQUIRE( widget.button()->isEnabled() );
    widget.button()->click();
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
}

/// The optional command occupies its own row beside the existing gain, synchronization, and averaging controls.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera EM-gain-off layout with existing controls", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::setup_emGain; xqt::camera::setup_emGainOff;
    xqt::camera::setup_synchro; xqt::camera::setup_avgTime;
    #endif
    // clang-format on
    Widget widget;
    widget.ready();
    for( const auto *name : { "temp_ccd", "emgain" } )
    {
        pcf::IndiProperty number( pcf::IndiProperty::Number, widget.m_device, name );
        const std::string value = std::string( name ) == "temp_ccd" ? "-45" : "10";
        number.add( pcf::IndiElement( "current", value ) );
        number.add( pcf::IndiElement( "target", value ) );
        widget.publish( number );
    }
    widget.support();
    pcf::IndiProperty synchro( pcf::IndiProperty::Switch, widget.m_device, "synchro" );
    synchro.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    widget.publish( synchro );
    pcf::IndiProperty average( pcf::IndiProperty::Number, widget.m_device + "-avg", "avgTime" );
    average.add( pcf::IndiElement( "current", "1" ) );
    average.add( pcf::IndiElement( "target", "1" ) );
    widget.publish( average );
    widget.m_widget.adjustSize();
    events();

    std::vector<QWidget *> controls;
    for( const auto *name : { "emgain", "emGainOff", "synchro", "avgTime" } )
    {
        auto *control = widget.m_widget.findChild<QWidget *>( name );
        REQUIRE( control != nullptr );
        REQUIRE( control->isVisible() );
        for( auto *previous : controls )
            REQUIRE_FALSE( previous->geometry().intersects( control->geometry() ) );
        controls.push_back( control );
    }
    const QString screenshot = qEnvironmentVariable( "CAMERA_GUI_SCREENSHOT" );
    if( !screenshot.isEmpty() )
        REQUIRE( widget.m_widget.grab().save( screenshot ) );
}

} // namespace cameraGUITest
} // namespace libXWCTest

/// Run the socket-free camera widget tests with Qt's offscreen backend.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Test arguments. */ )
{
    qputenv( "QT_QPA_PLATFORM", "offscreen" );
    QApplication application( argc, argv );
    return Catch::Session().run( argc, argv );
}
