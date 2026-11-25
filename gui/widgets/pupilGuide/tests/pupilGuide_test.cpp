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
/// Capture both panel and child commands and reproduce shared-publisher dispatch offline.
class Publisher : public multiIndiSubscriber
{
  public:
    std::vector<pcf::IndiProperty> m_sent; ///< Commands captured without transport.

    /// Detach subscribers before their Qt owners are destroyed.
    ~Publisher() noexcept;

    /// Capture an outgoing command.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Outgoing property. */ ) override;

    /// Count registrations for a property, including independent child subscriptions.
    size_t count( const std::string &key /**< [in] Device.property key. */ ) const;
};

Publisher::~Publisher() noexcept
{
    detachSubscribersRecursive();
}

void Publisher::sendNewProperty( const pcf::IndiProperty &property )
{
    m_sent.push_back( property );
}

size_t Publisher::count( const std::string &key ) const
{
    return subscribedProperties.count( key );
}

/// Exercise the real widget without constructing an INDI client or contacting hardware.
class Widget : public xqt::pupilGuide
{
  public:
    Publisher m_publisher; ///< Offline parent shared by the panel and child subscribers.

    std::vector<pcf::IndiProperty> &m_sent; ///< Alias for all captured commands.

    /// Attach the panel and children to an offline publisher.
    Widget();

    /// Report the effective mode for fallback tests.
    bool labMode() const;

    /// Capture a command instead of sending it.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Command being submitted. */ ) override;

    /// Deliver a definition to the panel and its independently subscribed children.
    void define( const pcf::IndiProperty &property, /**< [in] Property to deliver. */
                 bool                     definition = true /**< [in] False delivers a partial Set update. */ );

    /// Supply a switch capability and state.
    void toggle( const std::string &device,          /**< [in] Device name. */
                 const std::string &property,        /**< [in] Property name. */
                 bool               on      = false, /**< [in] Observed switch state. */
                 const std::string &element = "toggle" /**< [in] Switch element. */ );

    /// Define command elements whose values are initially empty.
    void capability( const std::string                  &device,   /**< [in] Device name. */
                     const std::string                  &property, /**< [in] Command property. */
                     std::initializer_list<const char *> elements /**< [in] Command elements. */ );

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

Widget::Widget() : m_sent( m_publisher.m_sent )
{
    m_publisher.addSubscriber( this );
}

bool Widget::labMode() const
{
    return m_labMode;
}

void Widget::toggle( const std::string &device, const std::string &name, bool on, const std::string &element )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch, device, name );
    property.add( pcf::IndiElement( element ) );
    property[element].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );
    define( property );
}

void Widget::capability( const std::string                  &device,
                         const std::string                  &name,
                         std::initializer_list<const char *> elements )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number, device, name );
    for( const auto *element : elements )
        property.add( pcf::IndiElement( element ) );
    define( property );
}

void Widget::sendNewProperty( const pcf::IndiProperty &property )
{
    m_sent.push_back( property );
}

void Widget::define( const pcf::IndiProperty &property, bool definition )
{
    if( definition )
        handleDefProperty( property );
    else
        handleSetProperty( property );
    for( auto *child : findChildren<QWidget *>() )
    {
        if( auto *subscriber = dynamic_cast<multiIndiSubscriber *>( child ) )
        {
            if( definition )
                subscriber->handleDefProperty( property );
            else
                subscriber->handleSetProperty( property );
        }
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
    capability( "wooferModes", "target_amps", { "0000", "0001", "0002" } );
    number( "wooferModes", "current_amps", { { "0000", 1 }, { "0001", 2 }, { "0002", 3 } } );
    state( "picomotors", "READY" );
    number( "picomotors", "picoscix_pos", { { "current", 100 }, { "target", 100 } } );
    state( "twAlign-camwfs-wfs", "READY" );
    toggle( "twAlign-camwfs-wfs", "continuous" );
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

/** \brief Missing or invalid woofer baselines block only their dependent modes, including direct slot calls.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide validates woofer values and partial updates", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::receiveProperty; xqt::pupilGuide::propertyAvailable; xqt::pupilGuide::updateAvailability;
    xqt::pupilGuide::on_button_tip_u_pressed; xqt::pupilGuide::on_button_focus_p_pressed;
#endif
    // clang-format on
    Widget widget;
    widget.onConnect();
    widget.state( "dmwoofer", "OPERATING" );
    widget.state( "wooferModes", "READY" );
    widget.capability( "wooferModes", "target_amps", { "0000", "0001", "0002" } );
    widget.on_button_tip_u_pressed();
    widget.on_button_focus_p_pressed();
    CHECK( widget.m_sent.empty() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    CHECK_FALSE( widget.control( "button_focus_p" )->isEnabled() );

    pcf::IndiProperty update( pcf::IndiProperty::Number, "wooferModes", "current_amps" );
    update.add( pcf::IndiElement( "0002", "0" ) );
    widget.define( update, false );
    REQUIRE( widget.control( "button_focus_p" )->isEnabled() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    widget.on_button_focus_p_pressed();
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back()["0002"].get<double>() == Approx( 0.02 ) );
    widget.m_sent.clear();

    update.setElements( { { "0000", pcf::IndiElement( "0000", "1" ) } } );
    widget.define( update, false );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    update.setElements( { { "0001", pcf::IndiElement( "0001", "2" ) } } );
    widget.define( update, false );
    REQUIRE( widget.control( "button_tip_u" )->isEnabled() );
    widget.on_button_tip_u_pressed();
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back().getDevice() == "wooferModes" );
    CHECK( widget.m_sent.back().getName() == "target_amps" );
    CHECK( widget.m_sent.back()["0000"].get<double>() == Approx( 0.9515195 ) );
    CHECK( widget.m_sent.back()["0001"].get<double>() == Approx( 2.0874620 ) );

    for( const std::string value : { "", "nan", "inf", "-inf", "1e999", "2bad" } )
    {
        CAPTURE( value );
        update["0001"] = value;
        widget.define( update, false );
        CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
        CHECK( widget.control( "button_focus_p" )->isEnabled() );
        widget.m_sent.clear();
        widget.on_button_tip_u_pressed();
        CHECK( widget.m_sent.empty() );
        update["0001"] = "2";
        widget.define( update, false );
        REQUIRE( widget.control( "button_tip_u" )->isEnabled() );
    }

    widget.erase( "wooferModes", "current_amps" );
    widget.on_button_tip_u_pressed();
    widget.on_button_focus_p_pressed();
    CHECK( widget.m_sent.empty() );
    widget.define( update, false );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    CHECK_FALSE( widget.control( "button_focus_p" )->isEnabled() );
    update.setElements( { { "0000", pcf::IndiElement( "0000", "0" ) } } );
    widget.define( update, false );
    CHECK( widget.control( "button_tip_u" )->isEnabled() );
    // A replacement definition does not retain omitted elements from the old vector.
    widget.define( update );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
}

/** \brief All subsets use all-required Start and any-applicable Stop, without replay or false-property dependencies.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide alignment start and stop have independent availability", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::subscribe; xqt::pupilGuide::updateAvailability;
    xqt::pupilGuide::on_button_startAlignment_pressed; xqt::pupilGuide::on_button_stopAlignment_pressed;
#endif
    // clang-format on
    const std::array<const char *, 3> devices{ "twAlign-camwfs-wfs", "twAlign-camwfs-ctrl", "camwfs-align" };
    const std::array<const char *, 3> properties{ "continuous", "loop_state", "loop_state" };
    for( bool lab : { false, true } )
    {
        for( unsigned mask = 0; mask < 8; ++mask )
        {
            CAPTURE( lab, mask );
            Widget widget;
            widget.onConnect();
            widget.state( "tcsi", "CONNECTED" );
            widget.toggle( "tcsi", "labMode", lab );
            CHECK( widget.m_publisher.count( "twAlign-camwfs-wfs.loop_state" ) == 0 );
            CHECK( widget.m_publisher.count( "twAlign-camwfs-wfs.continuous" ) == 2 );
            for( unsigned i = 0; i < 3; ++i )
            {
                if( !( mask & ( 1u << i ) ) )
                    continue;
                widget.state( devices[i], "READY" );
                widget.toggle( devices[i], properties[i] ); // Off is a known state, not a missing property.
            }
            CHECK( widget.control( "button_startAlignment" )->isEnabled() == ( mask == 7 ) );
            const unsigned stopMask = mask & ( lab ? 7u : 3u );
            CHECK( widget.control( "button_stopAlignment" )->isEnabled() == ( stopMask != 0 ) );
            widget.on_button_startAlignment_pressed();
            REQUIRE( widget.m_sent.size() == ( mask == 7 ? 3 : 0 ) );
            if( mask == 7 )
            {
                for( unsigned i = 0; i < 3; ++i )
                {
                    CHECK( widget.m_sent[i].getDevice() == devices[i] );
                    CHECK( widget.m_sent[i].getName() == properties[i] );
                    CHECK( widget.m_sent[i]["toggle"].getSwitchState() == pcf::IndiElement::On );
                }
            }
            widget.m_sent.clear();
            widget.on_button_stopAlignment_pressed();
            size_t sent = 0;
            for( unsigned i = 0; i < 3; ++i )
            {
                if( !( stopMask & ( 1u << i ) ) )
                    continue;
                REQUIRE( sent < widget.m_sent.size() );
                CHECK( widget.m_sent[sent].getDevice() == devices[i] );
                CHECK( widget.m_sent[sent].getName() == properties[i] );
                CHECK( widget.m_sent[sent++]["toggle"].getSwitchState() == pcf::IndiElement::Off );
            }
            CHECK( widget.m_sent.size() == sent );
        }
    }
    Widget widget;
    widget.onConnect();
    for( unsigned i = 0; i < 3; ++i )
    {
        widget.state( devices[i], "READY" );
        widget.toggle( devices[i], properties[i], true );
    }
    widget.state( devices[0], "ERROR" );
    CHECK_FALSE( widget.control( "button_startAlignment" )->isEnabled() );
    CHECK( widget.control( "button_stopAlignment" )->isEnabled() );
    widget.on_button_stopAlignment_pressed();
    CHECK( widget.m_sent.size() == 3 );
    widget.m_sent.clear();
    widget.erase( devices[0], properties[0] );
    widget.on_button_startAlignment_pressed();
    CHECK( widget.m_sent.empty() );
    widget.on_button_stopAlignment_pressed();
    CHECK( widget.m_sent.size() == 2 );
    widget.m_sent.clear();
    widget.state( devices[0], "READY" );
    widget.toggle( devices[0], properties[0] );
    CHECK( widget.control( "button_startAlignment" )->isEnabled() );
    CHECK( widget.m_sent.empty() );
    widget.onDisconnect();
    widget.on_button_startAlignment_pressed();
    widget.on_button_stopAlignment_pressed();
    CHECK( widget.m_sent.empty() );
}

/** \brief TCS loss enters lab mode and returning mode/state may arrive in either order.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide falls back to lab mode when TCS disappears", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::receiveProperty; xqt::pupilGuide::handleDelProperty; xqt::pupilGuide::updateAvailability;
    xqt::pupilGuide::on_button_ttmtel_pressed; xqt::pupilGuide::on_button_tip_u_pressed;
#endif
    // clang-format on
    for( bool modeFirst : { false, true } )
    {
        for( const std::string loss : { "", "fsm", "labMode", "NOTCONNECTED" } )
        {
            Widget widget;
            widget.ready();
            CHECK( widget.labMode() );
            if( modeFirst )
                widget.toggle( "tcsi", "labMode", false );
            widget.state( "tcsi", "CONNECTED" );
            if( !modeFirst )
                widget.toggle( "tcsi", "labMode", false );
            widget.capability( "tcsi", "pyrNudge", { "x", "y", "z" } );
            REQUIRE_FALSE( widget.labMode() );
            widget.on_button_ttmtel_pressed(); // woofer -> telescope
            widget.on_button_tip_u_pressed();
            REQUIRE( widget.m_sent.size() == 1 );
            CHECK( widget.m_sent.back().getDevice() == "tcsi" );
            widget.m_sent.clear();
            if( loss == "NOTCONNECTED" )
                widget.state( "tcsi", loss );
            else
                widget.erase( "tcsi", loss );
            CHECK( widget.labMode() );
            CHECK( widget.control( "button_ttmtel" )->isEnabled() );
            widget.on_button_tip_u_pressed();
            CHECK( widget.m_sent.empty() );
            CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
            widget.on_button_ttmtel_pressed(); // telescope -> TTM; never redirect a pending move
            widget.on_button_ttmtel_pressed(); // TTM -> woofer
            widget.updateGUI();
            CHECK( widget.control( "button_tip_u" )->isEnabled() );
            if( modeFirst )
                widget.toggle( "tcsi", "labMode", false );
            widget.state( "tcsi", "CONNECTED" );
            if( !modeFirst )
            {
                CHECK( widget.labMode() );
                widget.toggle( "tcsi", "labMode", false );
            }
            CHECK_FALSE( widget.labMode() );
            widget.onDisconnect();
            CHECK( widget.labMode() );
            widget.onConnect();
            CHECK( widget.labMode() );
        }
    }
}

/** \brief Sensor parameters have independent availability, and property loss cancels pending child state.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide sensor properties recover independently", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::receiveProperty; xqt::pupilGuide::resetChild; xqt::pupilGuide::updateAvailability;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    auto *slider = widget.findChild<xqt::toggleSlider *>( "actAlignSensor_slider" );
    REQUIRE( slider->isEnabled() );
    CHECK_FALSE( widget.control( "actAlignSensor_nAverage" )->isEnabled() );
    CHECK_FALSE( widget.control( "actAlignSensor_pokeAmp" )->isEnabled() );
    const auto registrations = widget.m_publisher.count( "twAlign-camwfs-wfs.continuous" );
    widget.erase( "twAlign-camwfs-wfs", "nPokeImages" );
    CHECK_FALSE( widget.control( "actAlignSensor_nImages" )->isEnabled() );
    CHECK( slider->isEnabled() );
    widget.number( "twAlign-camwfs-wfs", "nPokeImages", { { "target", 5 } } );
    CHECK_FALSE( widget.control( "actAlignSensor_nImages" )->isEnabled() );
    pcf::IndiProperty update( pcf::IndiProperty::Number, "twAlign-camwfs-wfs", "nPokeImages" );
    update.add( pcf::IndiElement( "current", "5" ) );
    widget.define( update, false );
    CHECK( widget.control( "actAlignSensor_nImages" )->isEnabled() );
    slider->on_slider_sliderPressed();
    widget.erase( "twAlign-camwfs-wfs" );
    widget.state( "twAlign-camwfs-wfs", "READY" );
    CHECK_FALSE( slider->isEnabled() );
    widget.toggle( "twAlign-camwfs-wfs", "continuous", true );
    REQUIRE( slider->isEnabled() );
    auto *control = slider->findChild<QSlider *>();
    CHECK( control->sliderPosition() == control->maximum() );
    CHECK( widget.m_sent.empty() );
    CHECK( widget.m_publisher.count( "twAlign-camwfs-wfs.continuous" ) == registrations );
}

/** \brief Partial actuator data gates axes independently and never emits half a diagonal move.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide motion waits for the axes it uses", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::on_button_pup_ul_pressed; xqt::pupilGuide::on_button_ttmPeri_l_pressed;
    xqt::pupilGuide::on_button_ttmPeri_u_pressed; xqt::pupilGuide::on_button_camlens_r_pressed;
    xqt::pupilGuide::on_button_camlens_d_pressed; xqt::pupilGuide::move_picoscix;
#endif
    // clang-format on
    Widget widget;
    widget.onConnect();
    widget.number( "ttmpupil", "pos_1", { { "current", 10 }, { "target", 10 } } );
    widget.state( "ttmpupil", "READY" );
    widget.on_button_pup_ul_pressed();
    CHECK( widget.m_sent.empty() );
    widget.number( "ttmpupil", "pos_2", { { "current", 20 }, { "target", 20 } } );
    widget.on_button_pup_ul_pressed();
    REQUIRE( widget.m_sent.size() == 2 );
    CHECK( widget.m_sent[0]["target"].get<double>() == Approx( 10 + 0.5 / std::sqrt( 2.0 ) ) );
    CHECK( widget.m_sent[1]["target"].get<double>() == Approx( 20 + 0.5 / std::sqrt( 2.0 ) ) );
    widget.m_sent.clear();
    widget.erase( "ttmpupil", "pos_2" );
    widget.on_button_pup_ul_pressed();
    CHECK( widget.m_sent.empty() );

    widget.state( "ttmperi", "OPERATING" );
    widget.number( "ttmperi", "axis1_voltage", { { "current", 10 }, { "target", 10 } } );
    widget.on_button_ttmPeri_l_pressed();
    widget.on_button_ttmPeri_u_pressed();
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back()["target"].get<double>() == Approx( 35 ) );
    widget.m_sent.clear();
    widget.number( "ttmperi", "axis2_voltage", { { "current", 20 }, { "target", 20 } } );
    widget.erase( "ttmperi", "axis1_voltage" );
    widget.on_button_ttmPeri_l_pressed();
    widget.on_button_ttmPeri_u_pressed();
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back().getName() == "axis2_voltage" );
    CHECK( widget.m_sent.back()["target"].get<double>() == Approx( 45 ) );
    widget.m_sent.clear();

    widget.state( "stagecamlensx", "READY" );
    widget.state( "stagecamlensy", "READY" );
    widget.number( "stagecamlensx", "position", { { "current", 1 }, { "target", 1 } } );
    widget.on_button_camlens_r_pressed();
    widget.on_button_camlens_d_pressed();
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back()["target"].get<double>() == Approx( 1.01 ) );
    widget.m_sent.clear();
    widget.erase( "stagecamlensx", "fsm" );
    widget.state( "stagecamlensx", "READY" );
    widget.on_button_camlens_r_pressed();
    CHECK( widget.m_sent.empty() );
    widget.number( "stagecamlensx", "position", { { "current", 2 }, { "target", 2 } } );
    widget.on_button_camlens_r_pressed();
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back()["target"].get<double>() == Approx( 2.01 ) );
    widget.m_sent.clear();

    widget.state( "picomotors", "READY" );
    widget.move_picoscix( 50 );
    CHECK( widget.m_sent.empty() );
    widget.number( "picomotors", "picoscix_pos", { { "current", 100 }, { "target", 100 } } );
    widget.move_picoscix( 50 );
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back()["target"].get<int>() == 150 );
    widget.m_sent.clear();
    pcf::IndiProperty pico( pcf::IndiProperty::Number, "picomotors", "picoscix_pos" );
    pico.add( pcf::IndiElement( "current", "1e2" ) );
    widget.define( pico, false );
    widget.move_picoscix( 50 );
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back()["target"].get<int>() == 150 );
    widget.m_sent.clear();
    pico["current"] = "1.5";
    widget.define( pico, false );
    widget.move_picoscix( 50 );
    CHECK( widget.m_sent.empty() );
    widget.number( "picomotors", "picoscix_pos", { { "current", std::numeric_limits<int>::max() }, { "target", 0 } } );
    widget.move_picoscix( 50 );
    CHECK( widget.m_sent.empty() );
}

/** \brief Each device can disappear and return without erasing another section's state.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide isolates every device section", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::handleDelProperty; xqt::pupilGuide::receiveProperty; xqt::pupilGuide::updateGUI;
#endif
    // clang-format on
    for( const std::string deleted : { "", "fsm" } )
    {
        Widget widget;
        widget.ready();
        const auto populate = [&widget]( const std::string &device )
        {
            widget.state( device, device == "ttmperi" ? "OPERATING" : "READY" );
            if( device == "modwfs" )
                widget.number( device, "modState", { { "current", 3 }, { "target", 3 } } );
            if( device == "picomotors" )
                widget.number( device, "picoscix_pos", { { "current", 100 }, { "target", 100 } } );
            if( device == "camwfs-fit" )
                widget.number( device, "threshold", { { "current", 0.5 }, { "target", 0.5 } } );
            if( device == "camwfs-avg" )
                widget.number( device, "avgTime", { { "current", 1 }, { "target", 1 } } );
            if( device == "ttmpupil" )
            {
                widget.number( device, "pos_1", { { "current", 10 }, { "target", 10 } } );
                widget.number( device, "pos_2", { { "current", 20 }, { "target", 20 } } );
            }
            if( device == "ttmperi" )
                widget.number( device, "axis1_voltage", { { "current", 10 }, { "target", 10 } } );
            if( device == "stagecamlensx" || device == "stagecamlensy" )
                widget.number( device, "position", { { "current", 1 }, { "target", 1 } } );
            if( device == "dmtweeter" || device == "dmncpc" )
                widget.toggle( device, "test_set" );
            if( device == "camwfs-align" || device == "twAlign-camwfs-ctrl" )
                widget.toggle( device, "loop_state" );
            if( device == "twAlign-camwfs-wfs" )
                widget.toggle( device, "continuous" );
        };
        const std::vector<std::pair<std::string, const char *>> sections{
            { "modwfs", "buttonMod_mod" },
            { "picomotors", "picoscix_l" },
            { "camwfs-fit", "fitThreshold" },
            { "camwfs-avg", "fitAvgTime" },
            { "ttmpupil", "button_pup_ul" },
            { "ttmperi", "button_ttmPeri_l" },
            { "stagecamlensx", "button_camlens_l" },
            { "stagecamlensy", "button_camlens_u" },
            { "dmtweeter", "buttonTweeterTest_set" },
            { "dmncpc", "buttonNCPCTest_set" },
            { "camwfs-align", "pupTrackLoop_slider" },
            { "twAlign-camwfs-ctrl", "actAlignLoop_slider" },
            { "twAlign-camwfs-wfs", "actAlignSensor_slider" } };
        for( const auto &section : sections )
            populate( section.first );
        for( const auto &section : sections )
        {
            CAPTURE( section.first, deleted );
            REQUIRE( widget.control( section.second )->isEnabled() );
            widget.erase( section.first, deleted );
            CHECK_FALSE( widget.control( section.second )->isEnabled() );
            CHECK( widget.control( "button_tip_u" )->isEnabled() );
            CHECK( widget.control( "button_ttmtel" )->isEnabled() );
            CHECK( widget.windowTitle() == "Alignment" );
            for( const auto &other : sections )
                if( other.first != section.first )
                    CHECK( widget.control( other.second )->isEnabled() );
            populate( section.first );
            CHECK( widget.control( section.second )->isEnabled() );
        }
        for( const std::string device : { "dmwoofer", "wooferModes" } )
        {
            widget.erase( device, deleted );
            CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
            CHECK( widget.control( "actAlignSensor_slider" )->isEnabled() );
            widget.state( device, device == "dmwoofer" ? "OPERATING" : "READY" );
            if( device == "wooferModes" )
            {
                CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
                widget.capability( device, "target_amps", { "0000", "0001", "0002" } );
                widget.number( device, "current_amps", { { "0000", 1 }, { "0001", 2 }, { "0002", 3 } } );
            }
            CHECK( widget.control( "button_tip_u" )->isEnabled() );
        }
        widget.number( "fxngenmodwfs", "C1ofst", { { "value", 1 } } );
        REQUIRE( widget.control( "modCh1" )->isEnabled() );
        widget.erase( "fxngenmodwfs", deleted );
        CHECK_FALSE( widget.control( "modCh1" )->isEnabled() );
        widget.number( "fxngenmodwfs", "C1ofst", { { "value", 1 } } );
        CHECK( widget.control( "modCh1" )->isEnabled() );
        widget.erase( "camwfs", deleted );
        CHECK( widget.control( "button_tip_u" )->isEnabled() );
        CHECK( widget.control( "actAlignSensor_slider" )->isEnabled() );
        CHECK( widget.m_sent.empty() );
    }
}

/** \brief Wrong protocol types and unknown switch states never count as availability.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide rejects malformed capability and value observations", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::receiveProperty; xqt::pupilGuide::on_button_tip_u_pressed;
    xqt::pupilGuide::on_button_stopAlignment_pressed;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    pcf::IndiProperty bad( pcf::IndiProperty::Text, "wooferModes", "current_amps" );
    bad.add( pcf::IndiElement( "0000", "1" ) );
    bad.add( pcf::IndiElement( "0001", "2" ) );
    widget.define( bad );
    widget.on_button_tip_u_pressed();
    CHECK( widget.m_sent.empty() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
    widget.number( "wooferModes", "current_amps", { { "0000", 1 }, { "0001", 2 }, { "0002", 0 } } );
    REQUIRE( widget.control( "button_tip_u" )->isEnabled() );
    bad = pcf::IndiProperty( pcf::IndiProperty::Switch, "twAlign-camwfs-wfs", "continuous" );
    bad.add( pcf::IndiElement( "toggle" ) ); // UnknownSwitchState
    widget.define( bad );
    CHECK_FALSE( widget.control( "button_stopAlignment" )->isEnabled() );
    CHECK_FALSE( widget.control( "actAlignSensor_slider" )->isEnabled() );
    widget.on_button_stopAlignment_pressed();
    CHECK( widget.m_sent.empty() );
    widget.toggle( "twAlign-camwfs-wfs", "continuous" );
    CHECK( widget.control( "button_stopAlignment" )->isEnabled() );
    widget.erase( "wooferModes", "fsm" );
    widget.state( "wooferModes", "READY" );
    widget.on_button_tip_u_pressed();
    CHECK( widget.m_sent.empty() );
    widget.onDisconnect();
    // A late property queued on the old connection must not establish a new command baseline.
    widget.number( "wooferModes", "current_amps", { { "0000", 1 }, { "0001", 2 }, { "0002", 0 } } );
    widget.onConnect();
    widget.state( "dmwoofer", "OPERATING" );
    widget.state( "wooferModes", "READY" );
    widget.capability( "wooferModes", "target_amps", { "0000", "0001", "0002" } );
    widget.on_button_tip_u_pressed();
    CHECK( widget.m_sent.empty() );
    CHECK_FALSE( widget.control( "button_tip_u" )->isEnabled() );
}

/** \brief Fitting readouts never turn missing inputs into apparent measured zero or stale averages.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide fitting displays require observed values", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::updateAvailability; xqt::pupilGuide::handleDelProperty;
#endif
    // clang-format on
    Widget widget;
    widget.onConnect();
    widget.state( "camwfs-fit", "READY" );
    for( int i = 1; i <= 4; ++i )
        widget.number(
            "camwfs-fit",
            "quadrant" + std::to_string( i ),
            { { "med", 10 }, { "x", 1 }, { "y", 2 }, { "D", 3 }, { "set-x", 0 }, { "set-y", 0 }, { "set-D", 0 } } );
    CHECK( widget.control( "coordAvg_x" )->isEnabled() );
    widget.erase( "camwfs-fit", "quadrant2" );
    CHECK_FALSE( widget.control( "coordLR_x" )->isEnabled() );
    CHECK( widget.findChild<QLineEdit *>( "coordLR_x" )->text().isEmpty() );
    CHECK_FALSE( widget.control( "coordAvg_x" )->isEnabled() );
    CHECK( widget.control( "coordLL_x" )->isEnabled() );
    widget.findChild<QCheckBox *>( "setDelta_pup" )->setChecked( false );
    widget.number( "camwfs-fit", "quadrant2", { { "x", 0 } } );
    CHECK( widget.control( "coordLR_x" )->isEnabled() );
    CHECK_FALSE( widget.control( "coordLR_y" )->isEnabled() );
    CHECK( widget.control( "coordAvg_x" )->isEnabled() );
    widget.findChild<QCheckBox *>( "setDelta_pup" )->setChecked( true );
    widget.updateGUI();
    CHECK_FALSE( widget.control( "coordLR_x" )->isEnabled() );
    CHECK_FALSE( widget.control( "coordAvg_x" )->isEnabled() );
}

/** \brief Every parent remote-action slot rejects absent command dependencies even when called directly.
 * \ingroup pupilGuideGUI_unit_test
 */
TEST_CASE( "Pupil-guide command slots reject unavailable inputs", "[pupilGuideGUI]" )
{
    // clang-format off
#ifdef PUPILGUIDE_GUI_TEST_DOXYGEN_REF
    xqt::pupilGuide::commandEnabled; xqt::pupilGuide::move_picoscix;
#endif
    // clang-format on
    Widget                                           widget;
    const std::vector<void ( xqt::pupilGuide::* )()> commands{
        &xqt::pupilGuide::on_buttonMod_mod_pressed,        &xqt::pupilGuide::on_buttonMod_set_pressed,
        &xqt::pupilGuide::on_buttonMod_rest_pressed,       &xqt::pupilGuide::on_button_focus_p_pressed,
        &xqt::pupilGuide::on_button_focus_m_pressed,       &xqt::pupilGuide::on_buttonTweeterTest_set_pressed,
        &xqt::pupilGuide::on_buttonNCPCTest_set_pressed,   &xqt::pupilGuide::on_buttonPup_rest_pressed,
        &xqt::pupilGuide::on_buttonPup_set_pressed,        &xqt::pupilGuide::on_button_ttmPeri_rest_pressed,
        &xqt::pupilGuide::on_button_ttmPeri_set_pressed,   &xqt::pupilGuide::on_button_startAlignment_pressed,
        &xqt::pupilGuide::on_button_stopAlignment_pressed, &xqt::pupilGuide::on_picoscix_l_pressed,
        &xqt::pupilGuide::on_picoscix_r_pressed,           &xqt::pupilGuide::on_picoscix_go_pressed,
        &xqt::pupilGuide::on_button_tip_u_pressed,         &xqt::pupilGuide::on_button_tip_ul_pressed,
        &xqt::pupilGuide::on_button_tip_l_pressed,         &xqt::pupilGuide::on_button_tip_dl_pressed,
        &xqt::pupilGuide::on_button_tip_d_pressed,         &xqt::pupilGuide::on_button_tip_dr_pressed,
        &xqt::pupilGuide::on_button_tip_r_pressed,         &xqt::pupilGuide::on_button_tip_ur_pressed,
        &xqt::pupilGuide::on_button_pup_ul_pressed,        &xqt::pupilGuide::on_button_pup_dl_pressed,
        &xqt::pupilGuide::on_button_pup_dr_pressed,        &xqt::pupilGuide::on_button_pup_ur_pressed,
        &xqt::pupilGuide::on_button_ttmPeri_l_pressed,     &xqt::pupilGuide::on_button_ttmPeri_r_pressed,
        &xqt::pupilGuide::on_button_ttmPeri_u_pressed,     &xqt::pupilGuide::on_button_ttmPeri_d_pressed,
        &xqt::pupilGuide::on_button_camlens_u_pressed,     &xqt::pupilGuide::on_button_camlens_l_pressed,
        &xqt::pupilGuide::on_button_camlens_d_pressed,     &xqt::pupilGuide::on_button_camlens_r_pressed };
    for( bool connected : { false, true } )
    {
        if( connected )
            widget.onConnect();
        for( const auto command : commands )
        {
            ( widget.*command )();
            CHECK( widget.m_sent.empty() );
        }
        widget.move_picoscix( 50 );
        CHECK( widget.m_sent.empty() );
    }
}

} // namespace pupilGuideGUITest
} // namespace libXWCTest

/// Run offline Qt tests without an INDI connection.
int main( int argc, /**< [in] Argument count. */ char **argv /**< [in] Catch options. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
