/** \file camera_test.cpp
 * rief Offline discovery, lifecycle, and command tests for camera controls and the star-acquisition panel.
 * \ingroup cameraGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <cstdlib>
#include <initializer_list>
#include <QApplication>
#include <QKeyEvent>
#include <QLineEdit>
#include <QTemporaryFile>
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

    /// Configure, register, and display the real panel with the socket-free publisher.
    Widget( const std::string &stageNames = "" /**< [in] Optional comma-separated list of associated stages. */ );

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
    {
        QApplication::processEvents();
    }
}

Widget::Widget( const std::string &stageNames ) : m_widget( m_device )
{
    if( !stageNames.empty() )
    {
        QTemporaryFile configFile;
        REQUIRE( configFile.open() );
        REQUIRE( configFile.write( QByteArray::fromStdString( "[camera]\nstages=" + stageNames + "\n" ) ) > 0 );
        configFile.close();

        mx::app::appConfigurator config;
        m_widget.setupConfig( config );
        REQUIRE( config.readConfig( configFile.fileName().toStdString() ) == 0 );
        m_widget.loadConfig( config );
    }

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
    REQUIRE( button->text() == "off" );
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

/// Pressing Return in Avg. Time sends a numeric target to the camera's averager.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera averaging time sends a numeric target", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::setup_avgTime;
    xqt::statusEntry::handleSetProperty;
    xqt::statusEntry::on_value_returnPressed;
    #endif
    // clang-format on
    Widget widget;
    widget.ready();

    pcf::IndiProperty average( pcf::IndiProperty::Number, widget.m_device + "-avg", "avgTime" );
    average.add( pcf::IndiElement( "current", 1.0 ) );
    average.add( pcf::IndiElement( "target", 1.0 ) );
    widget.publish( average );

    auto *entry = widget.m_widget.findChild<xqt::statusEntry *>( "avgTime" );
    REQUIRE( entry != nullptr );

    // The entry subscribes when created. Deliver the averager's reply to that subscription.
    widget.publish( average );
    REQUIRE( entry->formattedValue().toDouble() == 1.0 );
    REQUIRE( entry->isEnabled() );
    REQUIRE_FALSE( entry->readOnly() );

    auto *editor = entry->findChild<QLineEdit *>( "value" );
    REQUIRE( editor != nullptr );
    editor->setFocus();
    entry->setEditText( "2.75" );
    QKeyEvent submit( QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier );
    QApplication::sendEvent( editor, &submit );

    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
    const auto &command = widget.m_publisher.m_commands.back();
    CHECK( command.getDevice() == average.getDevice() );
    CHECK( command.getName() == average.getName() );
    CHECK( command.getType() == pcf::IndiProperty::Number );
    REQUIRE( command.getNumElements() == 1 );
    CHECK( command["target"].get<double>() == 2.75 );

    average["current"].set( 2.75 );
    average["target"].set( 2.75 );
    widget.m_publisher.handleSetProperty( average );
    events();
    CHECK( entry->inputText().toDouble() == 2.75 );
}

/// The optional command sits beside the gain entry regardless of property discovery order.
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

    const bool requestFirst = GENERATE( false, true );
    if( requestFirst )
    {
        widget.support();
    }

    for( const auto *name : { "temp_ccd", "emgain" } )
    {
        pcf::IndiProperty number( pcf::IndiProperty::Number, widget.m_device, name );
        const std::string value = std::string( name ) == "temp_ccd" ? "-45" : "10";
        number.add( pcf::IndiElement( "current", value ) );
        number.add( pcf::IndiElement( "target", value ) );
        widget.publish( number );
    }

    if( !requestFirst )
    {
        widget.support();
    }

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
        {
            REQUIRE_FALSE( previous->geometry().intersects( control->geometry() ) );
        }

        controls.push_back( control );
    }

    const QRect gain   = controls[0]->geometry();
    const QRect button = controls[1]->geometry();
    CHECK( gain.right() < button.left() );
    CHECK( std::abs( gain.center().y() - button.center().y() ) <= 1 );

    auto *fsm = widget.m_widget.findChild<xqt::fsmDisplay *>( "fsmState" );
    REQUIRE( fsm != nullptr );
    CHECK( fsm->geometry().bottom() == controls.back()->geometry().bottom() );

    const QString screenshot = qEnvironmentVariable( "CAMERA_GUI_SCREENSHOT" );

    if( !screenshot.isEmpty() )
    {
        REQUIRE( widget.m_widget.grab().save( screenshot ) );
    }
}

/// Temperature controls stay at the top and ordered settings align with the bottom of the stage column.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera packs optional controls in discovery-independent order", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::setup_vshiftSpd; xqt::camera::setup_expTime; xqt::camera::setup_fps;
    xqt::camera::setup_emGain; xqt::camera::setup_synchro;
    xqt::camera::setup_modes; xqt::camera::setup_roiStatus;
    xqt::camera::setup_temp_ccd; xqt::camera::setup_tempStatus;
    #endif
    // clang-format on
    Widget widget( "stage1,stage2,stage3,stage4,stage5,stage6,stage7" );
    widget.ready();
    REQUIRE( widget.m_widget.findChildren<xqt::stageStatus *>().size() == 7 );

    const auto checkRows = [&]( std::initializer_list<const char *> names )
    {
        widget.m_widget.adjustSize();
        events();

        QWidget *previous = nullptr;
        for( const char *name : names )
        {
            CAPTURE( name );
            auto *control = widget.m_widget.findChild<QWidget *>( name );
            REQUIRE( control != nullptr );
            REQUIRE( control->isVisible() );

            if( previous )
            {
                CHECK( previous->geometry().bottom() + 1 == control->geometry().top() );
            }

            previous = control;
        }
    };

    pcf::IndiProperty synchro( pcf::IndiProperty::Switch, widget.m_device, "synchro" );
    synchro.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    widget.publish( synchro );
    widget.support();

    for( const auto *name : { "emgain", "exptime" } )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Number, widget.m_device, name );
        property.add( pcf::IndiElement( "current", 1.0 ) );
        widget.publish( property );
    }

    pcf::IndiProperty vshift( pcf::IndiProperty::Switch, widget.m_device, "vshift_speed" );
    vshift.add( pcf::IndiElement( "fast", pcf::IndiElement::On ) );
    widget.publish( vshift );

    checkRows( { "vshiftSpd", "expTime", "emgain", "synchro" } );

    auto *heading      = widget.m_widget.findChild<QWidget *>( "lab_camName" );
    auto *lastStage    = widget.m_widget.findChild<xqt::stageStatus *>( "stage7" );
    auto *firstSetting = widget.m_widget.findChild<QWidget *>( "vshiftSpd" );
    auto *lastSetting  = widget.m_widget.findChild<QWidget *>( "synchro" );
    REQUIRE( heading != nullptr );
    REQUIRE( lastStage != nullptr );
    REQUIRE( firstSetting != nullptr );
    REQUIRE( lastSetting != nullptr );
    CHECK( firstSetting->geometry().top() > heading->geometry().top() );
    CHECK( lastSetting->geometry().bottom() == lastStage->geometry().bottom() );

    auto *exposure = widget.m_widget.findChild<xqt::statusEntry *>( "expTime" );
    REQUIRE( exposure != nullptr );
    auto *field = exposure->findChild<QLineEdit *>( "value" );
    REQUIRE( field != nullptr );
    CHECK( field->height() == field->sizeHint().height() );

    widget.m_widget.resize( widget.m_widget.width(), widget.m_widget.height() + 200 );
    events();
    CHECK( field->height() == field->sizeHint().height() );
    CHECK( lastSetting->geometry().bottom() == lastStage->geometry().bottom() );

    pcf::IndiProperty tempControl( pcf::IndiProperty::Text, widget.m_device, "temp_control" );
    tempControl.add( pcf::IndiElement( "status", "READY" ) );
    widget.publish( tempControl );
    pcf::IndiProperty tempCCD( pcf::IndiProperty::Number, widget.m_device, "temp_ccd" );
    tempCCD.add( pcf::IndiElement( "current", -45.0 ) );
    widget.publish( tempCCD );

    checkRows( { "tempCCD", "tempStatus" } );
    auto *temperature       = widget.m_widget.findChild<QWidget *>( "tempCCD" );
    auto *temperatureStatus = widget.m_widget.findChild<QWidget *>( "tempStatus" );
    REQUIRE( temperature != nullptr );
    REQUIRE( temperatureStatus != nullptr );
    CHECK( temperature->geometry().top() == heading->geometry().top() );
    CHECK( temperatureStatus->geometry().bottom() + 1 < firstSetting->geometry().top() );
    CHECK( lastSetting->geometry().bottom() == lastStage->geometry().bottom() );

    pcf::IndiProperty fps( pcf::IndiProperty::Number, widget.m_device, "fps" );
    fps.add( pcf::IndiElement( "current", 1.0 ) );
    widget.publish( fps );
    pcf::IndiProperty mode( pcf::IndiProperty::Switch, widget.m_device, "mode" );
    mode.add( pcf::IndiElement( "full", pcf::IndiElement::On ) );
    widget.publish( mode );
    pcf::IndiProperty roi( pcf::IndiProperty::Switch, widget.m_device, "roi_set" );
    roi.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );
    widget.publish( roi );

    checkRows( { "roiStatus", "modes", "vshiftSpd", "expTime", "fps", "emgain", "synchro" } );
    CHECK( field->height() == field->sizeHint().height() );
    CHECK( lastSetting->geometry().bottom() == lastStage->geometry().bottom() );
}

/// Parked stage and wheel readouts survive power transitions while every motion input is blocked.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Parked camera stages and wheels are read-only through power cycles", "[cameraGUI][stageGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::setEnableDisable;
    xqt::stageStatus::selectionEnabled; xqt::statusCombo::updateGUI;
    xqt::statusCombo::on_buttonGo_pressed; xqt::statusCombo::on_buttonCtrl_pressed;
    xqt::stage::updateGUI; xqt::stage::on_setPointGo_pressed;
    xqt::stage::on_position_returnPressed; xqt::stage::on_home_pressed; xqt::stage::on_stop_pressed;
    #endif
    // clang-format on
    const bool        wheel  = GENERATE( false, true );
    const std::string device = wheel ? "wheel_test" : "stage_test";
    Widget            widget( device );
    widget.ready( "POWEROFF" );

    const auto state = [&]( const std::string &value )
    {
        pcf::IndiProperty fsm( pcf::IndiProperty::Text, device, "fsm" );
        fsm.add( pcf::IndiElement( "state", value ) );
        widget.publish( fsm );
    };
    state( "POWEROFF" );

    pcf::IndiProperty position( pcf::IndiProperty::Number, device, wheel ? "filter" : "position" );
    position.add( pcf::IndiElement( "current", 2.5 ) );
    position.add( pcf::IndiElement( "target", 2.5 ) );
    widget.publish( position );
    pcf::IndiProperty parked( pcf::IndiProperty::Number, device, "parked" );
    parked.add( pcf::IndiElement( "current", 1 ) );
    widget.publish( parked );
    pcf::IndiProperty presets( pcf::IndiProperty::Switch, device, wheel ? "filterName" : "presetName" );
    presets.add( pcf::IndiElement( "current", pcf::IndiElement::On ) );
    presets.add( pcf::IndiElement( "other", pcf::IndiElement::Off ) );
    widget.publish( presets );

    auto *summary = widget.m_widget.findChild<xqt::stageStatus *>( QString::fromStdString( device ) );
    REQUIRE( summary != nullptr );
    auto *dialog = qobject_cast<xqt::stage *>( summary->ctrlWidget() );
    REQUIRE( dialog != nullptr );
    auto *summaryChoice = summary->findChild<QComboBox *>( "status" );
    auto *summaryGo     = summary->findChild<QAbstractButton *>( "buttonGo" );
    auto *launcher      = summary->findChild<QAbstractButton *>( "buttonCtrl" );
    auto *choice        = dialog->findChild<QComboBox *>( "setPoint" );
    auto *field         = dialog->findChild<xqt::statusLineEdit *>( "position" );
    REQUIRE( summaryChoice != nullptr );
    REQUIRE( summaryGo != nullptr );
    REQUIRE( launcher != nullptr );
    REQUIRE( choice != nullptr );
    REQUIRE( field != nullptr );
    REQUIRE( launcher->isEnabled() );
    REQUIRE_FALSE( summaryChoice->isEnabled() );
    REQUIRE( summaryChoice->placeholderText() == "current" );
    REQUIRE( field->text() == "2.5" );

    state( "READY" );
    REQUIRE( summaryChoice->isEnabled() );
    REQUIRE( choice->isEnabled() );
    REQUIRE_FALSE( field->isReadOnly() );

    summaryChoice->setCurrentText( "other" );
    summary->on_status_activated( summaryChoice->currentIndex() );
    choice->setCurrentText( "other" );
    dialog->on_setPoint_activated( choice->currentIndex() );
    field->setEditText( "99" );

    for( const auto *transition : { "POWEROFF", "POWERON", "NODEVICE", "NOTCONNECTED", "CONNECTED" } )
    {
        CAPTURE( transition );
        state( transition );
        widget.state( transition );
        REQUIRE( launcher->isEnabled() );
        launcher->click();
        REQUIRE( dialog->isVisible() );
        REQUIRE_FALSE( summaryChoice->isEnabled() );
        REQUIRE_FALSE( summaryGo->isEnabled() );
        REQUIRE( summaryChoice->placeholderText() == "current" );
        REQUIRE( summaryChoice->currentIndex() == -1 );
        REQUIRE_FALSE( choice->isEnabled() );
        REQUIRE( choice->currentText() == "current" );
        REQUIRE( field->isReadOnly() );
        REQUIRE( field->isEnabled() );
        REQUIRE( field->text() == "2.5" );

        for( const auto *name : { "setPointGo",
                                  "positionSlider",
                                  "stepSize",
                                  "posMinus",
                                  "posPlus",
                                  "posStepMulTen",
                                  "posStepDivTen",
                                  "home",
                                  "stop" } )
        {
            auto *control = dialog->findChild<QWidget *>( name );
            REQUIRE( control != nullptr );
            CHECK_FALSE( control->isEnabled() );
        }

        dialog->on_setPointGo_pressed();
        dialog->on_position_returnPressed();
        dialog->on_positionSlider_sliderReleased();
        dialog->on_posPlus_pressed();
        dialog->on_posMinus_pressed();
        dialog->on_home_pressed();
        dialog->on_stop_pressed();
        REQUIRE( widget.m_publisher.m_commands.empty() );
    }

    state( "POWEROFF" );
    summaryChoice->setCurrentText( "other" );
    summary->on_buttonGo_pressed();
    REQUIRE( widget.m_publisher.m_commands.empty() );
    presets["current"].setSwitchState( pcf::IndiElement::Off );
    widget.publish( presets );
    REQUIRE( summaryChoice->placeholderText() == "2.5000" );
    REQUIRE( choice->currentIndex() == -1 );
    REQUIRE( field->text() == "2.5" );
    parked["current"].set( 0 );
    widget.publish( parked );
    REQUIRE( summaryChoice->placeholderText() == "POWEROFF" );
    REQUIRE_FALSE( summaryChoice->isEnabled() );

    state( "READY" );
    REQUIRE( summaryChoice->isEnabled() );
    REQUIRE( choice->isEnabled() );
    REQUIRE_FALSE( field->isReadOnly() );
    REQUIRE( field->editText().isEmpty() );
    REQUIRE( widget.m_publisher.m_commands.empty() );
    summaryChoice->setCurrentText( "other" );
    summaryGo->click();
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
    REQUIRE( widget.m_publisher.m_commands.back().getDevice() == device );
    REQUIRE( widget.m_publisher.m_commands.back().getName() == presets.getName() );
}

/// A psfAcq helper advertising acquire_star adds the star-acquisition panel below the existing left-column controls.
/** \ingroup cameraGUI_unit_test */
TEST_CASE( "Camera star-acquisition panel discovery and command", "[cameraGUI]" )
{
    // clang-format off
    #ifdef CAMERA_GUI_TEST_DOXYGEN_REF
    xqt::camera::handleSetProperty; xqt::camera::setup_psfAcq; xqt::camera::layoutLeftColumnControls;
    xqt::psfAcq::on_button_acquire_pressed;
    #endif
    // clang-format on
    Widget widget;
    widget.ready();
    const std::string acqName = widget.m_device + "-acq";

    pcf::IndiProperty unrelated( pcf::IndiProperty::Number, acqName, "num_stars" );
    unrelated.add( pcf::IndiElement( "current", "0" ) );
    widget.publish( unrelated );
    REQUIRE( widget.m_widget.findChild<xqt::psfAcq *>( "psfAcq" ) == nullptr );

    pcf::IndiProperty dark( pcf::IndiProperty::Switch, widget.m_device + "-dark", "start" );
    dark.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    widget.publish( dark );

    pcf::IndiProperty acquire( pcf::IndiProperty::Number, acqName, "acquire_star" );
    acquire.add( pcf::IndiElement( "current", "-1" ) );
    acquire.add( pcf::IndiElement( "target", "-1" ) );
    widget.publish( acquire );
    widget.publish( acquire );
    REQUIRE( widget.m_widget.findChildren<xqt::psfAcq *>( "psfAcq" ).size() == 1 );
    auto *panel = widget.m_widget.findChild<xqt::psfAcq *>( "psfAcq" );
    REQUIRE( panel->isVisible() );

    pcf::IndiProperty fsm( pcf::IndiProperty::Text, acqName, "fsm" );
    fsm.add( pcf::IndiElement( "state", "OPERATING" ) );
    widget.publish( fsm );
    pcf::IndiProperty star( pcf::IndiProperty::Number, acqName, "star_0" );
    for( const auto *name : { "x", "y", "peak", "fwhm", "seeing" } )
    {
        star.add( pcf::IndiElement( name, "0.9" ) );
    }
    star.add( pcf::IndiElement( "id", "7" ) );
    widget.publish( star );
    panel->updateGUI();

    auto *button = panel->findChild<QPushButton *>( "button_acquire" );
    REQUIRE( button->isEnabled() );
    REQUIRE( panel->findChild<QLabel *>( "val_seeing" )->text() == "0.90\"" );

    widget.m_widget.adjustSize();
    events();
    auto *takeDarks = widget.m_widget.findChild<QWidget *>( "takeDarks" );
    REQUIRE( takeDarks != nullptr );
    REQUIRE( takeDarks->geometry().bottom() < panel->geometry().top() );

    button->click();
    REQUIRE( widget.m_publisher.m_commands.size() == 1 );
    REQUIRE( widget.m_publisher.m_commands.back().getDevice() == acqName );
    REQUIRE( widget.m_publisher.m_commands.back().getName() == "acquire_star" );
    REQUIRE( widget.m_publisher.m_commands.back()["target"].get<int>() == 7 );

    const QString screenshot = qEnvironmentVariable( "CAMERA_GUI_ACQ_SCREENSHOT" );
    if( !screenshot.isEmpty() )
    {
        REQUIRE( widget.m_widget.grab().save( screenshot ) );
    }
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
