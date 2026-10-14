/** \file pupilGuide_test.cpp
 * \brief Offline pupil-guide availability, recovery, and command tests.
 * \ingroup pupilGuideGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>

#include "../pupilGuide.hpp"

namespace libXWCTest
{
namespace pupilGuideGUITest
{

/// \cond TEST_HARNESS
/// Exercise the real widget without constructing an INDI client or contacting hardware.
class Widget : public xqt::pupilGuide
{
  public:
    std::vector<pcf::IndiProperty> m_sent; ///< Commands captured in memory.

    /// Capture a command instead of sending it.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Command being submitted. */ ) override;

    /// Deliver a definition to the panel and its independently subscribed children.
    void define( const pcf::IndiProperty &property /**< [in] Definition to deliver. */ );

    /// Supply a device FSM definition.
    void state( const std::string &device, /**< [in] Device name. */
                const std::string &value /**< [in] FSM state. */ );

    /// Supply numeric elements.
    void number( const std::string                                 &device, /**< [in] Device name. */
                 const std::string                                 &name,   /**< [in] Property name. */
                 const std::vector<std::pair<std::string, double>> &values /**< [in] Elements to define. */ );

    /// Deliver a device-wide or individual-property deletion.
    void erase( const std::string &device, /**< [in] Device losing a property. */
                const std::string &name = "" /**< [in] Empty deletes the whole device. */ );

    /// Find a real child widget and fail the test if the UI contract changes.
    QWidget *control( const char *name /**< [in] Qt object name. */ );

    /// Establish representative independent device states.
    void ready();
};

void Widget::sendNewProperty( const pcf::IndiProperty &property )
{
    m_sent.push_back( property );
}

void Widget::define( const pcf::IndiProperty &property )
{
    handleDefProperty( property );
    for( auto *child : findChildren<QWidget *>() )
    {
        if( auto *subscriber = dynamic_cast<multiIndiSubscriber *>( child ) )
            subscriber->handleDefProperty( property );
    }
    QApplication::processEvents();
}

void Widget::state( const std::string &device, const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text, device, "fsm" );
    property.add( pcf::IndiElement( "state", value ) );
    define( property );
}

void Widget::number( const std::string                                 &device,
                     const std::string                                 &name,
                     const std::vector<std::pair<std::string, double>> &values )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number, device, name );
    for( const auto &value : values )
    {
        property.add( pcf::IndiElement( value.first ) );
        property[value.first] = value.second;
    }
    define( property );
}

void Widget::erase( const std::string &device, const std::string &name )
{
    pcf::IndiProperty property;
    property.setDevice( device );
    property.setName( name );
    handleDelProperty( property );
    for( auto *child : findChildren<QWidget *>() )
    {
        if( auto *subscriber = dynamic_cast<multiIndiSubscriber *>( child ) )
            subscriber->handleDelProperty( property );
    }
    QApplication::processEvents();
}

QWidget *Widget::control( const char *name )
{
    auto *widget = findChild<QWidget *>( name );
    REQUIRE( widget != nullptr );
    return widget;
}

void Widget::ready()
{
    onConnect();
    state( "modwfs", "READY" );
    state( "dmwoofer", "OPERATING" );
    state( "wooferModes", "READY" );
    number( "wooferModes", "current_amps", { { "0000", 1 }, { "0001", 2 }, { "0002", 3 } } );
    state( "picomotors", "READY" );
    number( "picomotors", "picoscix_pos", { { "current", 100 } } );
    state( "twAlign-camwfs-wfs", "READY" );
    number( "twAlign-camwfs-wfs", "nPokeImages", { { "current", 5 }, { "target", 5 } } );
}
/// \endcond

/** \brief Sensor loss and recovery preserve unrelated control availability and local selections.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide sensor restart stays local", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::handleDelProperty; xqt::pupilGuide::handleDefProperty; xqt::pupilGuide::updateGUI;
#endif
    // clang-format on
    for( const std::string deleted : { "", "fsm" } )
    {
        Widget widget;
        widget.ready();
        widget.on_button_tip_scale_pressed();
        auto      *scale     = widget.findChild<QPushButton *>( "button_tip_scale" );
        const auto scaleText = scale->text();
        REQUIRE( widget.control( "button_tip_u" )->isEnabled() );
        REQUIRE( widget.control( "actAlignSensor_nImages" )->isEnabled() );
        for( int restart = 0; restart < 2; ++restart )
        {
            widget.erase( "twAlign-camwfs-wfs", deleted );
            CHECK( widget.windowTitle() == "Alignment" );
            CHECK( widget.control( "button_ttmtel" )->isEnabled() );
            CHECK( widget.control( "button_tip_u" )->isEnabled() );
            CHECK( widget.control( "picoscix_l" )->isEnabled() );
            CHECK( widget.control( "button_camera" )->isEnabled() );
            CHECK_FALSE( widget.control( "actAlignSensor_nImages" )->isEnabled() );
            widget.state( "twAlign-camwfs-wfs", "READY" );
            widget.number( "twAlign-camwfs-wfs", "nPokeImages", { { "current", 7 }, { "target", 7 } } );
            CHECK( widget.control( "actAlignSensor_nImages" )->isEnabled() );
            CHECK( scale->text() == scaleText );
            CHECK( widget.m_sent.empty() );
        }
    }
}

/** \brief Unknown and non-FSM property deletions do not erase healthy device state.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide property deletion does not disconnect the panel", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::handleDelProperty;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    for( const auto &device : { "unknown", "twAlign-camwfs-wfs", "wooferModes" } )
    {
        widget.erase( device, "unrelated" );
        CHECK( widget.control( "button_tip_u" )->isEnabled() );
        CHECK( widget.control( "button_ttmtel" )->isEnabled() );
        CHECK( widget.windowTitle() == "Alignment" );
    }
    widget.erase( "twAlign-camwfs-wfs", "nPokeImages" );
    CHECK( widget.control( "button_tip_u" )->isEnabled() );
    CHECK( widget.control( "button_ttmtel" )->isEnabled() );
}

/** \brief Startup absence and true transport loss do not reuse previous connection state.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide shared connection controls global availability", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::onConnect; xqt::pupilGuide::onDisconnect; xqt::pupilGuide::updateGUI;
#endif
    // clang-format on
    Widget widget;
    CHECK_FALSE( widget.control( "button_ttmtel" )->isEnabled() );
    widget.onConnect();
    CHECK( widget.control( "button_ttmtel" )->isEnabled() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    CHECK_FALSE( widget.control( "actAlignSensor_nImages" )->isEnabled() );
    widget.ready();
    widget.onDisconnect();
    widget.updateGUI();
    CHECK_FALSE( widget.control( "button_ttmtel" )->isEnabled() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    CHECK( widget.windowTitle() == "Alignment (disconnected)" );
    widget.onConnect();
    CHECK( widget.control( "button_ttmtel" )->isEnabled() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    CHECK_FALSE( widget.control( "actAlignSensor_nImages" )->isEnabled() );
}

/** \brief Whole-device loss clears edits belonging to that device without submitting them.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide clears affected child edits", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::resetChild; xqt::pupilGuide::handleDelProperty;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    auto *sensor = widget.findChild<xqt::statusEntry *>( "actAlignSensor_nImages" );
    auto *pico   = widget.findChild<xqt::statusEntry *>( "picoscix_pos" );
    sensor->setEditText( "99" );
    pico->setEditText( "123" );
    widget.erase( "twAlign-camwfs-wfs" );
    CHECK( sensor->findChild<xqt::statusLineEdit *>()->editText().isEmpty() );
    CHECK( pico->findChild<xqt::statusLineEdit *>()->editText() == "123" );
    CHECK( widget.m_sent.empty() );
}

} // namespace pupilGuideGUITest
} // namespace libXWCTest

/// Run offline Qt tests without an INDI connection.
int main( int argc, /**< [in] Argument count. */ char **argv /**< [in] Catch options. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
