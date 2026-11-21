/** \file timingPanel_test.cpp
 * \brief Exercise timingGUI against a socket-free INDI publisher.
 * \ingroup timingGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>

#include "../../../lib/tests/offlineIndiPublisher.hpp"
#include "../timingPanel.hpp"

namespace libXWCTest
{
namespace timingGUITest
{
/// \cond TEST_HARNESS
/// Bind the real Qt panel to the shared offline publisher.
class Fixture
{
  public:
    guiTest::OfflinePublisher m_publisher; ///< Queued property delivery and captured commands.
    xqt::timingPanel          m_panel;     ///< Real operator widgets.

    /// Register panel subscriptions before any property is sent.
    Fixture();

    /// Send a definition or a partial update through the real publisher routing.
    void publish( const pcf::IndiProperty &property, /**< [in] INDI property to deliver. */
                  bool                     definition = false /**< [in] Deliver a definition instead of an update. */ );

    /// Look up a named Qt control or display.
    template <class Widget>
    Widget *widget( const char *name /**< [in] Object name of the widget. */ );
};

Fixture::Fixture()
{
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
/// \endcond

/** \brief Convert timing readbacks, retaining partial updates and gating synco on its enable readback. */
TEST_CASE( "timingGUI shows FPS and microseconds from the correct INDI elements", "[timingGUI]" )
{
    Fixture     fixture;
    const auto *cameraRate   = fixture.widget<QLabel>( "camwfs.value" );
    const auto *cameraJitter = fixture.widget<QLabel>( "camwfs.jitter" );
    const auto *syncoRate    = fixture.widget<QLabel>( "dmtweeter-synco.value" );
    const auto *syncoStatus  = fixture.widget<QLabel>( "dmtweeter-synco.status" );
    const auto *dmRate       = fixture.widget<QLabel>( "dmtweeter.value" );
    const auto *dmJitter     = fixture.widget<QLabel>( "dmtweeter.jitter" );
    const auto *wma          = fixture.widget<QLabel>( "holoop-time.value" );
    const auto *wmaJitter    = fixture.widget<QLabel>( "holoop-time.jitter" );
    CHECK( cameraRate->text().toStdString() == "---" );
    CHECK( syncoStatus->text().toStdString() == "---" );

    pcf::IndiProperty camera( pcf::IndiProperty::Number, "camwfs", "fg_timing" );
    camera.add( pcf::IndiElement( "acq_fps", 2000.25 ) );
    camera.add( pcf::IndiElement( "acq_jitter", 4.5e-6 ) );
    fixture.publish( camera, true );
    CHECK( cameraRate->text().toStdString() == "2000.25" );
    CHECK( cameraJitter->text().toStdString() == "4.50" );

    pcf::IndiProperty cameraUpdate( pcf::IndiProperty::Number, "camwfs", "fg_timing" );
    cameraUpdate.add( pcf::IndiElement( "acq_jitter", 0.0 ) );
    fixture.publish( cameraUpdate );
    CHECK( cameraRate->text().toStdString() == "2000.25" );
    CHECK( cameraJitter->text().toStdString() == "0.00" );

    pcf::IndiProperty delivered( pcf::IndiProperty::Number, "dmtweeter-synco", "timing" );
    delivered.add( pcf::IndiElement( "deliveredHz", 7997.25 ) );
    fixture.publish( delivered );
    CHECK( syncoRate->text().toStdString() == "---" );

    pcf::IndiProperty synco( pcf::IndiProperty::Switch, "dmtweeter-synco", "enable" );
    synco.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    synco.setState( pcf::IndiProperty::Busy );
    fixture.publish( synco );
    CHECK( syncoRate->text().toStdString() == "7997.25" );
    pcf::IndiProperty health( pcf::IndiProperty::Text, "dmtweeter-synco", "health" );
    health.add( pcf::IndiElement( "reason", "production clock/ownership warming" ) );
    fixture.publish( health, true );
    CHECK( syncoStatus->text().toStdString() == "Production clock/ownership warming" );

    health["reason"] = "production active";
    fixture.publish( health );
    pcf::IndiProperty steps( pcf::IndiProperty::Number, "dmtweeter-synco", "timing" );
    steps.add( pcf::IndiElement( "active", 4.0 ) );
    fixture.publish( steps );
    CHECK( syncoStatus->text().toStdString() == "Production active - 4 steps" );
    CHECK( syncoRate->text().toStdString() == "7997.25" );

    delivered["deliveredHz"] = 0.0;
    fixture.publish( delivered );
    CHECK( syncoRate->text().toStdString() == "0.00" );
    delivered["deliveredHz"] = 7980.0;
    fixture.publish( delivered );
    CHECK( syncoRate->text().toStdString() == "7980.00" );
    CHECK( syncoStatus->text().toStdString() == "Production active - 4 steps" );
    synco["toggle"].setSwitchState( pcf::IndiElement::Off );
    synco.setState( pcf::IndiProperty::Idle );
    fixture.publish( synco );
    CHECK( syncoRate->text().toStdString() == "---" );
    CHECK( syncoStatus->text().toStdString() == "Disabled" );
    fixture.m_publisher.handleDelProperty( synco );
    QApplication::processEvents();
    CHECK( syncoStatus->text().toStdString() == "---" );

    pcf::IndiProperty dm( pcf::IndiProperty::Number, "dmtweeter", "dm_timing" );
    dm.add( pcf::IndiElement( "receipt_fps", 2000.0 ) );
    dm.add( pcf::IndiElement( "send_fps", 7985.0 ) );
    dm.add( pcf::IndiElement( "send_jitter", 3.25e-6 ) );
    fixture.publish( dm );
    CHECK( dmRate->text().toStdString() == "7985.00" );
    CHECK( dmJitter->text().toStdString() == "3.25" );

    pcf::IndiProperty loop( pcf::IndiProperty::Number, "holoop-time", "loop_timing" );
    loop.add( pcf::IndiElement( "mean", 275.5e-6 ) );
    loop.add( pcf::IndiElement( "stddev", 12e-6 ) );
    loop.add( pcf::IndiElement( "valid", 0 ) );
    fixture.publish( loop );
    CHECK( wma->text().toStdString() == "---" );
    loop["valid"] = 1;
    fixture.publish( loop );
    CHECK( wma->text().toStdString() == "275.50" );
    CHECK( wmaJitter->text().toStdString() == "12.00" );
    loop["valid"] = 0;
    fixture.publish( loop );
    CHECK( wma->text().toStdString() == "---" );
    CHECK( wmaJitter->text().toStdString() == "---" );

    pcf::IndiProperty deleted;
    deleted.setDevice( "camwfs" );
    fixture.m_publisher.handleDelProperty( deleted );
    QApplication::processEvents();
    CHECK( cameraRate->text().toStdString() == "---" );
    CHECK( cameraJitter->text().toStdString() == "---" );
    CHECK( dmRate->text().toStdString() == "7985.00" );

    pcf::IndiProperty dmUpdate( pcf::IndiProperty::Number, "dmtweeter", "dm_timing" );
    dmUpdate.add( pcf::IndiElement( "send_fps", -1.0 ) );
    fixture.publish( dmUpdate );
    CHECK( dmRate->text().toStdString() == "---" );
    CHECK( dmJitter->text().toStdString() == "---" );
    dmUpdate["send_fps"] = 8000.0;
    fixture.publish( dmUpdate );
    CHECK( dmRate->text().toStdString() == "8000.00" );
    CHECK( dmJitter->text().toStdString() == "---" );
    CHECK( fixture.m_publisher.m_commands.empty() );

    // clang-format off
#ifdef TIMINGGUI_TEST_DOXYGEN_REF
    xqt::timingPanel::handleSetProperty;
    xqt::timingPanel::handleDelProperty;
#endif
    // clang-format on
}

/** \brief Verify standard controls route operator commands and recover across rediscovery. */
TEST_CASE( "timingGUI routes synco and waveform controls to the named devices", "[timingGUI]" )
{
    Fixture fixture;
    auto   *enable = fixture.widget<QWidget>( "dmtweeter-synco.enable" );
    auto   *slider = enable->findChild<QSlider *>();
    REQUIRE( slider );
    CHECK_FALSE( enable->isEnabled() );

    pcf::IndiProperty toggle( pcf::IndiProperty::Switch, "dmtweeter-synco", "enable" );
    toggle.setState( pcf::IndiProperty::Busy );
    toggle.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    fixture.publish( toggle );
    CHECK( enable->isEnabled() );
    CHECK( slider->sliderPosition() == slider->maximum() );
    CHECK( fixture.m_publisher.m_commands.empty() );

    slider->setSliderPosition( slider->minimum() );
    REQUIRE( QMetaObject::invokeMethod( enable, "on_slider_sliderReleased", Qt::DirectConnection ) );
    REQUIRE( fixture.m_publisher.m_commands.size() == 1 );
    CHECK( fixture.m_publisher.m_commands.back().createUniqueKey() == toggle.createUniqueKey() );
    CHECK( fixture.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::Off );

    auto *reacquire = fixture.widget<QPushButton>( "dmtweeter-synco.reacquire" );
    CHECK_FALSE( reacquire->isEnabled() );
    pcf::IndiProperty request( pcf::IndiProperty::Switch, "dmtweeter-synco", "reacquire" );
    request.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );
    fixture.publish( request, true );
    reacquire->click();
    REQUIRE( fixture.m_publisher.m_commands.size() == 2 );
    CHECK( fixture.m_publisher.m_commands.back().createUniqueKey() == request.createUniqueKey() );
    CHECK( fixture.m_publisher.m_commands.back()["request"].getSwitchState() == pcf::IndiElement::On );

    fixture.m_publisher.detachAllSubscribers();
    fixture.m_panel.onDisconnect();
    CHECK_FALSE( reacquire->isEnabled() );
    fixture.m_publisher.addSubscriber( &fixture.m_panel );
    QApplication::processEvents();

    for( const std::string device : { "tweeterSpeck", "tweeterMod" } )
    {
        pcf::IndiProperty modulating( pcf::IndiProperty::Switch, device, "modulating" );
        modulating.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
        fixture.publish( modulating );
        auto *mod       = fixture.widget<QWidget>( ( device + ".modulating" ).c_str() );
        auto *modSlider = mod->findChild<QSlider *>();
        REQUIRE( modSlider );
        REQUIRE( mod->isEnabled() );
        modSlider->setSliderPosition( modSlider->maximum() );
        REQUIRE( QMetaObject::invokeMethod( mod, "on_slider_sliderReleased", Qt::DirectConnection ) );
        CHECK( fixture.m_publisher.m_commands.back().createUniqueKey() == modulating.createUniqueKey() );
        CHECK( fixture.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::On );

        const bool        sparkle   = device == "tweeterSpeck";
        const char       *parameter = sparkle ? "separation" : "frequency";
        const auto        current   = sparkle ? 15.0 : 2000.0;
        const auto        target    = sparkle ? 17.5 : 2100.0;
        pcf::IndiProperty setting( pcf::IndiProperty::Number, device, parameter );
        setting.add( pcf::IndiElement( "current", current ) );
        setting.add( pcf::IndiElement( "target", current ) );
        fixture.publish( setting );

        auto *entry = fixture.widget<QWidget>( ( device + "." + parameter ).c_str() );
        auto *line  = entry->findChild<QLineEdit *>();
        REQUIRE( line );
        REQUIRE( entry->isEnabled() );
        CHECK( line->text().toDouble() == current );
        line->setText( QString::number( target ) );
        REQUIRE( QMetaObject::invokeMethod( entry, "on_value_returnPressed", Qt::DirectConnection ) );
        CHECK( fixture.m_publisher.m_commands.back().createUniqueKey() == setting.createUniqueKey() );
        CHECK( fixture.m_publisher.m_commands.back()["target"].get<double>() == target );
    }

    CHECK( fixture.m_panel.findChild<QWidget *>( "tweeterSpeck.frequency" ) == nullptr );
    CHECK( fixture.m_panel.findChild<QWidget *>( "tweeterMod.separation" ) == nullptr );

    pcf::IndiProperty angle( pcf::IndiProperty::Number, "tweeterSpeck", "angle" );
    angle.add( pcf::IndiElement( "current", 15.0 ) );
    angle.add( pcf::IndiElement( "target", 15.0 ) );
    fixture.publish( angle );
    auto *angleEntry = fixture.widget<QWidget>( "tweeterSpeck.angle" );
    auto *angleInput = angleEntry->findChild<QLineEdit *>();
    REQUIRE( angleInput );
    REQUIRE( angleEntry->isEnabled() );
    angleInput->setText( "27.5" );
    REQUIRE( QMetaObject::invokeMethod( angleEntry, "on_value_returnPressed", Qt::DirectConnection ) );
    CHECK( fixture.m_publisher.m_commands.back().createUniqueKey() == angle.createUniqueKey() );
    CHECK( fixture.m_publisher.m_commands.back()["target"].get<double>() == 27.5 );

    auto *fileLabel = fixture.widget<QLabel>( "tweeterMod.file" );
    CHECK( fileLabel->text().toStdString() == "---" );
    pcf::IndiProperty file( pcf::IndiProperty::Text, "tweeterMod", "file" );
    file.add( pcf::IndiElement( "current", "/data/fourier_mode_cube.fits" ) );
    file.add( pcf::IndiElement( "target", "another_cube.fits" ) );
    fixture.publish( file, true );
    CHECK( fileLabel->text().toStdString() == "fourier_mode_cube.fits" );
    CHECK( fileLabel->toolTip().toStdString() == "/data/fourier_mode_cube.fits" );
    fixture.m_publisher.handleDelProperty( file );
    QApplication::processEvents();
    CHECK( fileLabel->text().toStdString() == "---" );
    CHECK( fileLabel->toolTip().isEmpty() );

    CHECK( fixture.m_publisher.m_commands.size() == 7 );

    // clang-format off
#ifdef TIMINGGUI_TEST_DOXYGEN_REF
    xqt::timingPanel::subscribe;
    xqt::timingPanel::onDisconnect;
    xqt::timingPanel::handleDefProperty;
#endif
    // clang-format on
}

} // namespace timingGUITest
} // namespace libXWCTest

/// Run the socket-free Qt regression suite.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in,out] Qt and Catch command-line arguments. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
