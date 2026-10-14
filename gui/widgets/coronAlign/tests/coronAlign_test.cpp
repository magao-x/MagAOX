/** \file coronAlign_test.cpp
 * \brief Offline coronagraph restart, baseline, pending-motion, and rediscovery tests.
 * \ingroup coronAlignGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include "../../../lib/tests/offlineIndiPublisher.hpp"
#include "../coronAlign.hpp"

namespace libXWCTest
{
namespace coronAlignGUITest
{
/// \cond TEST_HARNESS
/// Drain queued subscriptions through the panel, summary, dialog, and FSM layers.
void events();
void events()
{
    for( int depth = 0; depth < 4; ++depth )
        QApplication::processEvents();
}

/// Exercise the actual panel and queued publisher using no sockets.
class Widget : public xqt::coronAlign
{
  public:
    guiTest::OfflinePublisher m_publisher; ///< Socket-free publisher and captured commands.

    /// Register the panel and its nested child subscribers.
    Widget();

    /// Establish transport availability and representative inputs for every axis.
    void ready();

    /// Publish an observation through the real queued dispatch path.
    void publish( const pcf::IndiProperty &property, /**< [in] Observation. */
                  bool                     definition = true /**< [in] False publishes a partial Set. */ );

    /// Publish one device FSM.
    void state( const std::string &device, /**< [in] Device name. */
                const std::string &value = "READY" /**< [in] Observed state. */ );

    /// Publish position inputs, using an empty target to match producer capability definitions.
    void position( const std::string &device, /**< [in] Device name. */
                   const std::string &name,   /**< [in] Property name. */
                   const std::string &value = "100" /**< [in] Current position text. */ );

    /// Delete one property or a device through the publisher.
    void erase( const std::string &device, /**< [in] Device name. */
                const std::string &name = "" /**< [in] Empty deletes the device. */ );

    /// Find a real GUI control, failing if the UI contract changes.
    QWidget *control( const std::string &name /**< [in] Qt object name. */ );

    /// Change camera mapping without enabling the existing unavailable LLOWFS selector.
    void camera( int value /**< [in] FLOWFS=0, LLOWFS=1, CAMSCIS=2. */ );

    /// Invoke a real slot directly, including when its button is disabled.
    void press( const std::string &name /**< [in] Button object name. */ );

    /// Supply all numeric inputs for one device, without changing its FSM.
    void positions( const std::string &device /**< [in] Device name. */ );
};

Widget::Widget()
{
    m_publisher.addSubscriber( this );
    events();
}
void Widget::publish( const pcf::IndiProperty &property, bool definition )
{
    if( definition )
        m_publisher.handleDefProperty( property );
    else
        m_publisher.handleSetProperty( property );
    events();
}
void Widget::state( const std::string &device, const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text, device, "fsm" );
    property.add( pcf::IndiElement( "state", value ) );
    publish( property );
}
void Widget::position( const std::string &device, const std::string &name, const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number, device, name );
    property.add( pcf::IndiElement( "current", value ) );
    property.add( pcf::IndiElement( "target" ) );
    publish( property );
}
void Widget::erase( const std::string &device, const std::string &name )
{
    pcf::IndiProperty property;
    property.setDevice( device );
    property.setName( name );
    m_publisher.handleDelProperty( property );
    events();
}
QWidget *Widget::control( const std::string &name )
{
    auto *result = findChild<QWidget *>( name.c_str() );
    REQUIRE( result != nullptr );
    return result;
}
void Widget::camera( int value )
{
    m_camera = value;
    updateGUI();
}
void Widget::press( const std::string &name )
{
    REQUIRE( QMetaObject::invokeMethod( this, ( "on_" + name + "_pressed" ).c_str(), Qt::DirectConnection ) );
}
void Widget::positions( const std::string &device )
{
    if( device == "picomotors" )
        for( const auto *name : { "picopupil_pos",
                                  "picofpm_pos",
                                  "picolyot_pos",
                                  "piaa0x_pos",
                                  "piaa1x_pos",
                                  "piaa1y_pos",
                                  "ipiaa0x_pos",
                                  "ipiaa1x_pos",
                                  "ipiaa1y_pos" } )
            position( device, name );
    else
        position( device, device.find( "fw" ) == 0 ? "filter" : "position" );
}
void Widget::ready()
{
    m_publisher.onConnect();
    events();
    for( const auto *device : { "fwpupil", "fwfpm", "fwlyot", "stagepiaa", "stageipiaa", "picomotors" } )
    {
        state( device );
        positions( device );
    }
}

/// List remote motion buttons while excluding local scale and camera selectors.
std::vector<std::string> motionButtons();
std::vector<std::string> motionButtons()
{
    std::vector<std::string> result;
    for( const auto *plane : { "pupil", "focal", "lyot", "piaa", "piaa0", "piaa1", "ipiaa", "ipiaa0", "ipiaa1" } )
        for( const auto *direction : { "u", "d", "l", "r" } )
        {
            if( ( std::string( plane ) == "piaa0" || std::string( plane ) == "ipiaa0" ) &&
                ( std::string( direction ) == "u" || std::string( direction ) == "d" ) )
                continue;
            result.push_back( std::string( "button_" ) + plane + "_" + direction );
        }
    return result;
}

/// Map the default science-camera arrows to their controlling device.
std::string deviceFor( const std::string &button /**< [in] Button name. */ );
std::string deviceFor( const std::string &button )
{
    if( button.back() == 'u' || button.back() == 'd' )
    {
        if( button.find( "button_pupil_" ) == 0 )
            return "fwpupil";
        if( button.find( "button_focal_" ) == 0 )
            return "fwfpm";
        if( button.find( "button_lyot_" ) == 0 )
            return "fwlyot";
        if( button.find( "button_piaa_" ) == 0 )
            return "stagepiaa";
        if( button.find( "button_ipiaa_" ) == 0 )
            return "stageipiaa";
    }
    return "picomotors";
}
/// \endcond

/** \brief Missing startup properties and full reconnect block every direct motion slot.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph startup and reconnect require fresh motion inputs", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::onConnect;
    xqt::coronAlign::onDisconnect;
    xqt::coronAlign::on_button_pupil_u_pressed;
    xqt::coronAlign::on_button_pupil_d_pressed;
    xqt::coronAlign::on_button_pupil_l_pressed;
    xqt::coronAlign::on_button_pupil_r_pressed;
    xqt::coronAlign::on_button_focal_u_pressed;
    xqt::coronAlign::on_button_focal_d_pressed;
    xqt::coronAlign::on_button_focal_l_pressed;
    xqt::coronAlign::on_button_focal_r_pressed;
    xqt::coronAlign::on_button_lyot_u_pressed;
    xqt::coronAlign::on_button_lyot_d_pressed;
    xqt::coronAlign::on_button_lyot_l_pressed;
    xqt::coronAlign::on_button_lyot_r_pressed;
    xqt::coronAlign::on_button_piaa_l_pressed;
    xqt::coronAlign::on_button_piaa_r_pressed;
    xqt::coronAlign::on_button_piaa_u_pressed;
    xqt::coronAlign::on_button_piaa_d_pressed;
    xqt::coronAlign::on_button_piaa0_l_pressed;
    xqt::coronAlign::on_button_piaa0_r_pressed;
    xqt::coronAlign::on_button_piaa1_l_pressed;
    xqt::coronAlign::on_button_piaa1_r_pressed;
    xqt::coronAlign::on_button_piaa1_u_pressed;
    xqt::coronAlign::on_button_piaa1_d_pressed;
    xqt::coronAlign::on_button_ipiaa_l_pressed;
    xqt::coronAlign::on_button_ipiaa_r_pressed;
    xqt::coronAlign::on_button_ipiaa_u_pressed;
    xqt::coronAlign::on_button_ipiaa_d_pressed;
    xqt::coronAlign::on_button_ipiaa0_l_pressed;
    xqt::coronAlign::on_button_ipiaa0_r_pressed;
    xqt::coronAlign::on_button_ipiaa1_l_pressed;
    xqt::coronAlign::on_button_ipiaa1_r_pressed;
    xqt::coronAlign::on_button_ipiaa1_u_pressed;
    xqt::coronAlign::on_button_ipiaa1_d_pressed;
    xqt::coronAlign::commandEnabled;
    xqt::coronAlign::resetConnectionState;
#endif
    // clang-format on
    Widget widget;
    for( bool connected : { false, true } )
    {
        if( connected )
            widget.onConnect();
        for( const auto &button : motionButtons() )
        {
            REQUIRE_FALSE( widget.control( button )->isEnabled() );
            widget.press( button );
        }
        REQUIRE( widget.m_publisher.m_commands.empty() );
    }
    widget.ready();
    widget.onDisconnect();
    widget.positions( "picomotors" ); // Late queued observations while disconnected must not restore baselines.
    widget.onConnect();
    for( const auto *device : { "fwpupil", "fwfpm", "fwlyot", "stagepiaa", "stageipiaa", "picomotors" } )
        widget.state( device );
    for( const auto &button : motionButtons() )
    {
        REQUIRE_FALSE( widget.control( button )->isEnabled() );
        widget.press( button );
    }
    REQUIRE( widget.m_publisher.m_commands.empty() );
    widget.positions( "picomotors" );
    REQUIRE( widget.control( "button_pupil_l" )->isEnabled() );
    REQUIRE_FALSE( widget.control( "button_pupil_u" )->isEnabled() );
}

/** \brief Every device and FSM restart affects only that device and recovers in either input order.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph device deletion stays local", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::handleDelProperty;
    xqt::coronAlign::resetDevice;
    xqt::coronAlign::receiveProperty;
    xqt::coronAlign::updateAvailability;
#endif
    // clang-format on
    for( const auto *device : { "fwpupil", "fwfpm", "fwlyot", "stagepiaa", "stageipiaa", "picomotors" } )
        for( const auto *deleted : { "", "fsm" } )
            for( bool dataFirst : { false, true } )
            {
                Widget widget;
                widget.ready();
                widget.on_button_pupil_scale_pressed();
                auto scale = qobject_cast<QPushButton *>( widget.control( "button_pupil_scale" ) )->text();
                widget.erase( device, deleted );
                widget.erase( device, deleted );
                REQUIRE( widget.windowTitle() == "Coronagraph Alignment" );
                REQUIRE( widget.control( "checkCamflowfs" )->isEnabled() );
                for( const auto &button : motionButtons() )
                    REQUIRE( widget.control( button )->isEnabled() == ( deviceFor( button ) != device ) );
                if( dataFirst )
                    widget.positions( device );
                else
                    widget.state( device );
                for( const auto &button : motionButtons() )
                    if( deviceFor( button ) == device )
                        REQUIRE_FALSE( widget.control( button )->isEnabled() );
                if( dataFirst )
                    widget.state( device );
                else
                    widget.positions( device );
                if( std::string( device ) != "picomotors" )
                {
                    auto *summary = widget.findChild<xqt::stageStatus *>( device );
                    REQUIRE( summary != nullptr );
                    REQUIRE( summary->ctrlWidget()->windowTitle() == device );
                }
                for( const auto &button : motionButtons() )
                    REQUIRE( widget.control( button )->isEnabled() );
                REQUIRE( qobject_cast<QPushButton *>( widget.control( "button_pupil_scale" ) )->text() == scale );
                REQUIRE( widget.m_publisher.m_commands.empty() );
            }
}

/** \brief Named deletion and partial updates preserve independent axes and require full paired-move inputs.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph property recovery preserves independent axes", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::handleDefProperty;
    xqt::coronAlign::handleSetProperty;
    xqt::coronAlign::motionAvailable;
    xqt::coronAlign::on_button_piaa_l_pressed;
    xqt::coronAlign::on_button_ipiaa_l_pressed;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    widget.erase( "unknown" );
    widget.erase( "fwpupil", "unknown" );
    for( const auto &button : motionButtons() )
        REQUIRE( widget.control( button )->isEnabled() );
    for( const auto *prefix : { "piaa", "ipiaa" } )
    {
        const std::string name   = std::string( prefix ) + "1x_pos";
        const std::string button = std::string( "button_" ) + prefix;
        widget.erase( "picomotors", name );
        REQUIRE_FALSE( widget.control( button + "_l" )->isEnabled() );
        REQUIRE_FALSE( widget.control( button + "1_l" )->isEnabled() );
        REQUIRE( widget.control( button + "0_l" )->isEnabled() );
        REQUIRE( widget.control( button + "1_u" )->isEnabled() );
        widget.press( button + "_l" );
        REQUIRE( widget.m_publisher.m_commands.empty() );
        pcf::IndiProperty partial( pcf::IndiProperty::Number, "picomotors", name );
        partial.add( pcf::IndiElement( "current", "0" ) );
        widget.publish( partial, false );
        REQUIRE_FALSE( widget.control( button + "_l" )->isEnabled() );
        partial = pcf::IndiProperty( pcf::IndiProperty::Number, "picomotors", name );
        partial.add( pcf::IndiElement( "target" ) );
        widget.publish( partial, false );
        REQUIRE( widget.control( button + "_l" )->isEnabled() );
        widget.publish( partial ); // Def replaces, rather than merging, the old current element.
        REQUIRE_FALSE( widget.control( button + "_l" )->isEnabled() );
        widget.position( "picomotors", name );
    }
    widget.erase( "fwpupil", "filter" );
    REQUIRE_FALSE( widget.control( "button_pupil_u" )->isEnabled() );
    REQUIRE( widget.control( "button_pupil_l" )->isEnabled() );
    widget.position( "fwpupil", "filter" );
    REQUIRE( widget.control( "button_pupil_u" )->isEnabled() );
}

/** \brief Motion commands retain existing camera transforms, step sizes, and paired ordering.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph movement mappings retain their command values", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::on_button_pupil_u_pressed;
    xqt::coronAlign::on_button_pupil_l_pressed;
    xqt::coronAlign::on_button_piaa_l_pressed;
    xqt::coronAlign::on_button_ipiaa_l_pressed;
#endif
    // clang-format on
    for( int camera : { 0, 1, 2 } )
        for( const auto *direction : { "u", "l" } )
        {
            Widget widget;
            widget.ready();
            widget.camera( camera );
            widget.press( std::string( "button_pupil_" ) + direction );
            REQUIRE( widget.m_publisher.m_commands.size() == 1 );
            const auto &command = widget.m_publisher.m_commands.front();
            bool        pico    = ( camera == 0 ) == ( std::string( direction ) == "u" );
            REQUIRE( command.getDevice() == ( pico ? "picomotors" : "fwpupil" ) );
            REQUIRE( command["target"].get<double>() == Approx( pico ? 10100.0 : 100.01 ) );
        }
    for( const auto *prefix : { "piaa", "ipiaa" } )
    {
        Widget widget;
        widget.ready();
        widget.press( std::string( "button_" ) + prefix + "_l" );
        REQUIRE( widget.m_publisher.m_commands.size() == 2 );
        REQUIRE( widget.m_publisher.m_commands[0].getName() == std::string( prefix ) + "0x_pos" );
        REQUIRE( widget.m_publisher.m_commands[1].getName() == std::string( prefix ) + "1x_pos" );
        for( const auto &command : widget.m_publisher.m_commands )
            REQUIRE( command["target"].get<double>() == ( std::string( prefix ) == "piaa" ? 80 : 120 ) );
        widget.press( std::string( "button_" ) + prefix + "_l" );
        REQUIRE( widget.m_publisher.m_commands.size() == 2 ); // Existing pending latch blocks repeated motion.
    }
}

/** \brief Malformed and out-of-range position inputs cannot restore motion; valid scientific integers can.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph motion rejects unusable position observations", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::receiveProperty;
    xqt::coronAlign::on_button_pupil_l_pressed;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    for( const auto *value : { "", "garbage", "nan", "inf", "2x", "3.5", "1e100", "9223372036854775808" } )
    {
        widget.position( "picomotors", "picopupil_pos", value );
        REQUIRE_FALSE( widget.control( "button_pupil_l" )->isEnabled() );
        REQUIRE( widget.control( "button_focal_l" )->isEnabled() );
        widget.press( "button_pupil_l" );
        REQUIRE( widget.m_publisher.m_commands.empty() );
    }
    widget.position( "picomotors", "picopupil_pos", "1e3" );
    widget.press( "button_pupil_l" );
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
    REQUIRE( widget.m_publisher.m_commands.back()["target"].get<double>() == 11000 );
}

/** \brief Device loss cancels affected pending motion and child edits without disturbing another device.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph recovery preserves unrelated pending state", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::handleDelProperty;
    xqt::coronAlign::resetDevice;
    xqt::coronAlign::expirePendingCommands;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    auto *pupil = widget.findChild<xqt::stageStatus *>( "fwpupil" );
    auto *focal = widget.findChild<xqt::stageStatus *>( "fwfpm" );
    REQUIRE( pupil != nullptr );
    REQUIRE( focal != nullptr );
    auto *pupilEdit = pupil->ctrlWidget()->findChild<QComboBox *>( "setPoint" );
    auto *focalEdit = focal->ctrlWidget()->findChild<QComboBox *>( "setPoint" );
    REQUIRE( pupilEdit != nullptr );
    REQUIRE( focalEdit != nullptr );
    for( const auto *device : { "fwpupil", "fwfpm" } )
    {
        pcf::IndiProperty presets( pcf::IndiProperty::Switch, device, "filterName" );
        presets.add( pcf::IndiElement( "initial", pcf::IndiElement::On ) );
        presets.add( pcf::IndiElement( "pending", pcf::IndiElement::Off ) );
        widget.publish( presets );
    }
    pupilEdit->setCurrentIndex( pupilEdit->findText( "pending" ) );
    focalEdit->setCurrentIndex( focalEdit->findText( "pending" ) );
    REQUIRE( pupilEdit->currentText() == "pending" );
    REQUIRE( focalEdit->currentText() == "pending" );
    REQUIRE( QMetaObject::invokeMethod(
        pupil->ctrlWidget(), "on_setPoint_activated", Q_ARG( int, pupilEdit->currentIndex() ) ) );
    REQUIRE( QMetaObject::invokeMethod(
        focal->ctrlWidget(), "on_setPoint_activated", Q_ARG( int, focalEdit->currentIndex() ) ) );
    widget.press( "button_pupil_u" );
    widget.erase( "fwfpm" );
    REQUIRE_FALSE( widget.control( "button_pupil_u" )->isEnabled() );
    REQUIRE( pupilEdit->currentText() == "pending" );
    REQUIRE( focalEdit->currentText().isEmpty() );
    widget.erase( "fwpupil" );
    REQUIRE( pupilEdit->currentText().isEmpty() );
    widget.state( "fwpupil" );
    widget.position( "fwpupil", "filter" );
    REQUIRE( widget.control( "button_pupil_u" )->isEnabled() );
    widget.expirePendingCommands();
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
}

/** \brief A silent late producer is queried through shared backoff and restores the real GUI without reconnect.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph late properties recover through shared rediscovery", "[coronAlignGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::coronAlign::handleDefProperty;
    xqt::coronAlign::updateGUI;
    multiIndiPublisher::update;
    multiIndiPublisher::handleDefProperty;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    widget.erase( "fwpupil" );
    widget.m_publisher.m_queries.clear();
    widget.m_publisher.tick( std::chrono::seconds( 1 ) );
    bool queriedState = false, queriedFilter = false;
    for( const auto &query : widget.m_publisher.m_queries )
    {
        if( query.getDevice() != "fwpupil" )
            continue;
        queriedState |= query.getName() == "fsm";
        queriedFilter |= query.getName() == "filter";
    }
    REQUIRE( queriedState );
    REQUIRE( queriedFilter );
    widget.state( "fwpupil" );
    widget.position( "fwpupil", "filter" );
    REQUIRE( widget.control( "button_pupil_u" )->isEnabled() );
    widget.m_publisher.m_queries.clear();
    widget.m_publisher.tick( std::chrono::seconds( 60 ) );
    for( const auto &query : widget.m_publisher.m_queries )
        REQUIRE_FALSE(
            ( query.getDevice() == "fwpupil" && ( query.getName() == "fsm" || query.getName() == "filter" ) ) );
    REQUIRE( widget.m_publisher.m_commands.empty() );
}

/** \brief Embedded stage dialogs preserve independent property state and never move from missing numeric inputs.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph stage dialogs recover named properties independently", "[coronAlignGUI][stageGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::stage::handleDefProperty;
    xqt::stage::handleDelProperty;
    xqt::stage::handleSetProperty;
    xqt::stage::motionReady;
    xqt::stage::readNumber;
    xqt::stage::updateGUI;
    xqt::stage::on_posMinus_pressed;
    xqt::stage::on_posPlus_pressed;
    xqt::stage::on_position_returnPressed;
    xqt::stage::on_positionSlider_sliderReleased;
#endif
    // clang-format on
    for( const auto *device : { "fwpupil", "stagepiaa" } )
    {
        Widget widget;
        widget.ready();
        auto *summary = widget.findChild<xqt::stageStatus *>( device );
        REQUIRE( summary != nullptr );
        auto *dialog = dynamic_cast<xqt::stage *>( summary->ctrlWidget() );
        REQUIRE( dialog != nullptr );
        auto *plus   = dialog->findChild<QPushButton *>( "posPlus" );
        auto *slider = dialog->findChild<QWidget *>( "positionSlider" );
        REQUIRE( plus != nullptr );
        REQUIRE( slider != nullptr );
        const std::string motion = std::string( device ) == "fwpupil" ? "filter" : "position";
        REQUIRE( plus->isEnabled() );
        REQUIRE( dialog->windowTitle() == device );
        widget.erase( device, motion );
        REQUIRE_FALSE( plus->isEnabled() );
        REQUIRE( dialog->findChild<QPushButton *>( "home" )->isEnabled() );
        dialog->on_posMinus_pressed();
        dialog->on_posPlus_pressed();
        dialog->on_position_returnPressed();
        dialog->on_positionSlider_sliderReleased();
        REQUIRE( widget.m_publisher.m_commands.empty() );
        widget.position( device, motion ); // No FSM resend is needed for named recovery.
        REQUIRE( plus->isEnabled() );
        REQUIRE( dialog->windowTitle() == device );
        widget.erase( device, motion == "filter" ? "position" : "filter" );
        REQUIRE( plus->isEnabled() ); // Inactive alternative subscription is unrelated.
        pcf::IndiProperty maximum( pcf::IndiProperty::Number, device, "maxpos" );
        maximum.add( pcf::IndiElement( "value", "1000" ) );
        widget.publish( maximum );
        REQUIRE( slider->isEnabled() );
        widget.erase( device, "maxpos" );
        REQUIRE_FALSE( slider->isEnabled() );
        REQUIRE( plus->isEnabled() );
        widget.publish( maximum );
        REQUIRE( slider->isEnabled() );
        pcf::IndiProperty partial( pcf::IndiProperty::Number, device, motion );
        partial.add( pcf::IndiElement( "target" ) );
        widget.publish( partial );
        REQUIRE_FALSE( plus->isEnabled() );
        partial = pcf::IndiProperty( pcf::IndiProperty::Number, device, motion );
        partial.add( pcf::IndiElement( "current", "0" ) );
        widget.publish( partial, false );
        REQUIRE( plus->isEnabled() );
        dialog->on_posPlus_pressed();
        REQUIRE( widget.m_publisher.m_commands.size() == 1 );
        REQUIRE( widget.m_publisher.m_commands[0].getName() == motion );
        REQUIRE( widget.m_publisher.m_commands[0]["target"].get<double>() == 1 );
    }
}

/** \brief Summary readouts and preset choices recover without losing the independent device FSM.
 * \ingroup coronAlignGUI_unit_test
 */
TEST_CASE( "Coronagraph stage summaries recover only deleted properties", "[coronAlignGUI][stageGUI]" )
{
    // clang-format off
#ifdef CORONALIGN_GUI_TEST_DOXYGEN_REF
    xqt::stageStatus::handleDefProperty;
    xqt::stageStatus::handleDelProperty;
    xqt::stageStatus::handleSetProperty;
    xqt::stageStatus::formatValue;
    xqt::statusCombo::handleDelProperty;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    auto *summary = widget.findChild<xqt::stageStatus *>( "fwpupil" );
    REQUIRE( summary != nullptr );
    auto *combo = summary->findChild<QComboBox *>( "status" );
    REQUIRE( combo != nullptr );
    REQUIRE( summary->formatValue() == "100.0000" );
    widget.erase( "fwpupil", "filter" );
    REQUIRE( summary->formatValue() == "---" );
    widget.position( "fwpupil", "filter" );
    REQUIRE( summary->formatValue() == "100.0000" );
    widget.erase( "fwpupil", "position" );
    REQUIRE( summary->formatValue() == "100.0000" );
    pcf::IndiProperty presets( pcf::IndiProperty::Switch, "fwpupil", "filterName" );
    presets.add( pcf::IndiElement( "initial", pcf::IndiElement::On ) );
    presets.add( pcf::IndiElement( "alternate", pcf::IndiElement::Off ) );
    widget.publish( presets );
    REQUIRE( combo->placeholderText() == "initial" );
    widget.erase( "fwpupil", "filterName" );
    REQUIRE( combo->count() == 0 );
    REQUIRE( combo->placeholderText() == "100.0000" );
    widget.publish( presets );
    REQUIRE( combo->count() == 2 );
    REQUIRE( combo->placeholderText() == "initial" );
    widget.erase( "fwpupil", "filterName" );
    for( const auto *invalid : { "", "bad", "nan", "inf", "1e999" } )
    {
        widget.position( "fwpupil", "filter", invalid );
        REQUIRE( summary->formatValue() == "---" );
        REQUIRE_FALSE( widget.control( "button_pupil_u" )->isEnabled() );
        REQUIRE_FALSE( summary->ctrlWidget()->findChild<QPushButton *>( "posPlus" )->isEnabled() );
    }
    widget.position( "fwpupil", "filter", "0" );
    REQUIRE( combo->placeholderText() == "0.0000" );
    REQUIRE( widget.m_publisher.m_commands.empty() );
}

} // namespace coronAlignGUITest
} // namespace libXWCTest

/// Run Qt widget tests with captured transport only.
int main( int    argc, /**< [in] Argument count. */
          char **argv /**< [in] Arguments for Qt and Catch. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
