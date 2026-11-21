/** \file polarimetry_test.cpp
 * \brief Offline polarimetry GUI subscription, control availability, status, and command tests.
 * \ingroup polarimetryGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include <QLabel>
#include <QPushButton>

#include "../polarimetry.hpp"

namespace libXWCTest
{
namespace polarimetryGUITest
{

/// \cond TEST_HARNESS
/// Capture commands and subscriptions without an INDI connection.
class Publisher : public multiIndiSubscriber
{
  public:
    std::vector<pcf::IndiProperty> m_sent; ///< Commands captured without transport.

    /// Detach subscribers before their Qt owners are destroyed.
    ~Publisher() noexcept;

    /// Capture an outgoing command.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Outgoing property. */ ) override;

    /// Count registrations for a property.
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
class Widget : public xqt::polarimetry
{
  public:
    Publisher m_publisher; ///< Offline parent shared by the panel and its child subscribers.

    std::vector<pcf::IndiProperty> &m_sent; ///< Alias for all captured commands.

    /// Attach the panel and children to an offline publisher.
    Widget();

    /// Capture a command instead of sending it.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Command being submitted. */ ) override;

    /// Deliver a property to the panel and its child subscribers.
    void define( const pcf::IndiProperty &property /**< [in] Property to deliver. */ );

    /// Supply a device FSM state.
    void state( const std::string &device, /**< [in] Device name. */
                const std::string &value /**< [in] FSM state. */ );

    /// Supply a toggle switch state.
    void toggle( const std::string &device,   /**< [in] Device name. */
                 const std::string &property, /**< [in] Property name. */
                 bool               on /**< [in] Switch state. */ );

    /// Supply numeric elements.
    void number( const std::string                                 &device, /**< [in] Device name. */
                 const std::string                                 &name,   /**< [in] Property name. */
                 const std::vector<std::pair<std::string, double>> &values /**< [in] Elements to define. */ );

    /// Supply the hwpSequencer positions list.
    void positions( const std::string &value /**< [in] Comma-separated positions. */ );

    /// Supply the hwpSequencer status.
    void status( int posIndex, /**< [in] 0-based position index. */
                 int cycle,    /**< [in] Current cycle. */
                 int frames /**< [in] Frames landed at the last position. */ );

    /// Find a real child widget and fail the test if the UI contract changes.
    QWidget *control( const char *name /**< [in] Qt object name. */ );

    /// Find a real child label.
    QLabel *label( const char *name /**< [in] Qt object name. */ );

    /// Connect with all devices ready and the sequencer idle.
    void ready();
};

Widget::Widget() : m_sent( m_publisher.m_sent )
{
    m_publisher.addSubscriber( this );
}

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
        {
            subscriber->handleDefProperty( property );
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

void Widget::toggle( const std::string &device, const std::string &name, bool on )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch, device, name );
    property.add( pcf::IndiElement( "toggle" ) );
    property["toggle"].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );
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

void Widget::positions( const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text, "hwpsequence", "positions" );
    property.add( pcf::IndiElement( "value", value ) );
    define( property );
}

void Widget::status( int posIndex, int cycle, int frames )
{
    number( "hwpsequence",
            "status",
            { { "pos_index", posIndex }, { "cycle", cycle }, { "target", 0 }, { "frames", frames } } );
}

QWidget *Widget::control( const char *name )
{
    auto *widget = findChild<QWidget *>( name );
    REQUIRE( widget != nullptr );
    return widget;
}

QLabel *Widget::label( const char *name )
{
    auto *widget = findChild<QLabel *>( name );
    REQUIRE( widget != nullptr );
    return widget;
}

void Widget::ready()
{
    onConnect();
    state( "hwptrack", "READY" );
    state( "hwpsequence", "READY" );
    toggle( "hwpsequence", "sequence", false );
    toggle( "hwpsequence", "abort", false );
    number( "hwpsequence", "frames_per_pos", { { "current", 0 }, { "target", 0 } } );
    number( "hwpsequence", "num_cycles", { { "current", 0 }, { "target", 0 } } );
}
/// \endcond

/** \brief The panel and its children subscribe to the current hwpSequencer and stage property names, not the removed
 * HWP tracking ones.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry subscribes to current property names", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::subscribe;
#endif
    // clang-format on
    Widget widget;

    for( const auto *key : { "hwptrack.fsm",
                             "hwptrack.hwp_position",
                             "hwptrack.hwp_position_name",
                             "hwpsequence.sequence",
                             "hwpsequence.last_cycle",
                             "hwpsequence.abort",
                             "hwpsequence.num_cycles",
                             "hwpsequence.frames_per_pos",
                             "hwpsequence.time_per_pos",
                             "hwpsequence.status",
                             "hwpsequence.positions",
                             "stagepollin.presetName",
                             "stageqwplin.presetName",
                             "qwptrack.fsm",
                             "qwptrack.tracking",
                             "qwptrack.qwp1",
                             "qwptrack.qwp2",
                             "qwppupil.fsm",
                             "qwppupil.tracking",
                             "qwppupil.offset" } )
    {
        INFO( key );
        CHECK( widget.m_publisher.count( key ) >= 1 );
    }

    for( const auto *key : { "hwpsequence.numCycles",
                             "hwpsequence.timePerPos",
                             "hwpsequence.lastCycle",
                             "hwpsequence.hwpPosIndex",
                             "hwpsequence.curCycle",
                             "stagepollin.",
                             "stageqwplin.",
                             "hwptrack.tracking",
                             "hwptrack.hwp_tracking_offset",
                             "hwptrack.hwp_position_actual",
                             "stagepolrot.fsm" } )
    {
        INFO( key );
        CHECK( widget.m_publisher.count( key ) == 0 );
    }

    CHECK( widget.findChild<QWidget *>( "qwpPupilFsm" ) != nullptr );
    CHECK( widget.findChild<QWidget *>( "sliderHwpTracking" ) == nullptr );
    CHECK( widget.findChild<QWidget *>( "stagePolRotFsm" ) == nullptr );
}

/** \brief While idle, the time entry is disabled in frames mode and the sequencing buttons are hidden.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry idle controls follow frames mode", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::handleSetProperty; xqt::polarimetry::updateGUI;
#endif
    // clang-format on
    Widget widget;
    widget.ready();

    CHECK_FALSE( widget.control( "buttonStartSequence" )->isHidden() );
    CHECK( widget.control( "buttonStartSequence" )->isEnabled() );
    CHECK( widget.control( "buttonStopSequence" )->isHidden() );
    CHECK( widget.control( "buttonLastCycle" )->isHidden() );
    CHECK( widget.control( "buttonAbortSequence" )->isHidden() );
    CHECK( widget.control( "framesStatus" )->isHidden() );

    CHECK( widget.control( "entryNumCycles" )->isEnabled() );
    CHECK( widget.control( "entryFramesPerPos" )->isEnabled() );
    CHECK( widget.control( "entryTimePerPos" )->isEnabled() );
    CHECK( widget.control( "labelTimeHint" )->isEnabled() );
    CHECK( widget.label( "labelNumCyclesHint" )->text() == "(0 for infinite)" );

    widget.number( "hwpsequence", "frames_per_pos", { { "current", 4 }, { "target", 4 } } );

    CHECK( widget.control( "entryFramesPerPos" )->isEnabled() );
    CHECK_FALSE( widget.control( "entryTimePerPos" )->isEnabled() );
    CHECK_FALSE( widget.control( "labelTimeHint" )->isEnabled() );

    widget.number( "hwpsequence", "frames_per_pos", { { "current", 0 }, { "target", 0 } } );

    CHECK( widget.control( "entryTimePerPos" )->isEnabled() );
}

/** \brief While sequencing, only num_cycles is editable and abort is offered until the app acknowledges it.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry sequencing controls", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::handleSetProperty; xqt::polarimetry::updateGUI;
#endif
    // clang-format on
    Widget widget;
    widget.ready();
    widget.toggle( "hwpsequence", "sequence", true );

    CHECK( widget.control( "buttonStartSequence" )->isHidden() );
    CHECK_FALSE( widget.control( "buttonStopSequence" )->isHidden() );
    CHECK_FALSE( widget.control( "buttonLastCycle" )->isHidden() );
    CHECK_FALSE( widget.control( "buttonAbortSequence" )->isHidden() );
    CHECK( widget.control( "buttonAbortSequence" )->isEnabled() );

    CHECK( widget.control( "entryNumCycles" )->isEnabled() );
    CHECK_FALSE( widget.control( "entryFramesPerPos" )->isEnabled() );
    CHECK_FALSE( widget.control( "entryTimePerPos" )->isEnabled() );
    CHECK_FALSE( widget.control( "comboHwpLin" )->isEnabled() );
    CHECK_FALSE( widget.control( "comboQwpLin" )->isEnabled() );

    widget.toggle( "hwpsequence", "abort", true );

    CHECK_FALSE( widget.control( "buttonAbortSequence" )->isHidden() );
    CHECK_FALSE( widget.control( "buttonAbortSequence" )->isEnabled() );

    widget.toggle( "hwpsequence", "abort", false );
    widget.toggle( "hwpsequence", "sequence", false );

    CHECK( widget.control( "buttonAbortSequence" )->isHidden() );
    CHECK( widget.control( "comboHwpLin" )->isEnabled() );
    CHECK( widget.control( "entryFramesPerPos" )->isEnabled() );
}

/** \brief Position and frames status text come from the status and positions properties.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry status text", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::countPositions; xqt::polarimetry::updateGUI;
#endif
    // clang-format on
    CHECK( xqt::polarimetry::countPositions( "0,45,22.5,67.5" ) == 4 );
    CHECK( xqt::polarimetry::countPositions( "0" ) == 1 );
    CHECK( xqt::polarimetry::countPositions( "" ) == 0 );
    CHECK( xqt::polarimetry::countPositions( "0,abc" ) == 0 );

    Widget widget;
    widget.ready();
    widget.toggle( "hwpsequence", "sequence", true );

    SECTION( "position total from positions, with fallback" )
    {
        widget.status( 1, 1, 0 );
        CHECK( widget.label( "hwpPosIndex" )->text() == "2" );

        widget.positions( "0,45,22.5,67.5" );
        CHECK( widget.label( "hwpPosIndex" )->text() == "2 / 4" );

        widget.positions( "" );
        CHECK( widget.label( "hwpPosIndex" )->text() == "2" );
    }

    SECTION( "frames row only in frames mode" )
    {
        widget.status( 1, 1, 3 );
        CHECK( widget.control( "framesStatus" )->isHidden() );

        widget.number( "hwpsequence", "frames_per_pos", { { "current", 4 }, { "target", 4 } } );
        CHECK_FALSE( widget.control( "framesStatus" )->isHidden() );
        CHECK_FALSE( widget.control( "labelFramesStatus" )->isHidden() );
    }

    SECTION( "frames count, stale first position and mismatch warning" )
    {
        widget.number( "hwpsequence", "frames_per_pos", { { "current", 4 }, { "target", 4 } } );

        widget.status( 0, 1, 3 );
        CHECK( widget.label( "framesStatus" )->text() == "---" );
        CHECK_FALSE( widget.label( "framesStatus" )->property( "isWarning" ).toBool() );

        widget.status( 1, 1, 4 );
        CHECK( widget.label( "framesStatus" )->text() == "4 / 4" );
        CHECK_FALSE( widget.label( "framesStatus" )->property( "isWarning" ).toBool() );

        widget.status( 2, 1, 3 );
        CHECK( widget.label( "framesStatus" )->text() == "3 / 4" );
        CHECK( widget.label( "framesStatus" )->property( "isWarning" ).toBool() );

        // first position of a later cycle has a valid count from the previous cycle's last position
        widget.status( 0, 2, 4 );
        CHECK( widget.label( "framesStatus" )->text() == "4 / 4" );
        CHECK_FALSE( widget.label( "framesStatus" )->property( "isWarning" ).toBool() );
    }
}

/** \brief The QWP angles show beside the tracking slider only while tracking, and the slider locks while sequencing.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry QWP tracking", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::handleSetProperty; xqt::polarimetry::updateGUI; xqt::polarimetry::lockSlider;
#endif
    // clang-format on
    Widget widget;
    widget.ready();

    QLabel  *label  = widget.label( "labelQwpTracking" );
    QWidget *slider = widget.control( "sliderQwpTracking" );

    // qwptrack not yet READY
    CHECK( slider->isEnabled() );
    CHECK( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );

    widget.state( "qwptrack", "READY" );
    widget.number( "qwptrack", "qwp1", { { "current", 12.4 }, { "target", 12.6 } } );
    widget.number( "qwptrack", "qwp2", { { "current", 135.0 }, { "target", 135.0 } } );

    CHECK_FALSE( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );
    CHECK( label->isHidden() );

    widget.toggle( "qwptrack", "tracking", true );

    CHECK_FALSE( label->isHidden() );
    CHECK( label->isEnabled() );

    // current, not target, padded to three digits with a figure space
    CHECK( label->text() == QString::fromUtf8( "QWP1: \u2007"
                                               "12°  QWP2: 135°" ) );

    widget.number( "qwptrack", "qwp1", { { "current", 5.0 } } );
    CHECK( label->text() == QString::fromUtf8( "QWP1: \u2007\u2007"
                                               "5°  QWP2: 135°" ) );

    widget.toggle( "hwpsequence", "sequence", true );

    CHECK( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );
    CHECK_FALSE( label->isHidden() );
    CHECK_FALSE( label->isEnabled() );

    widget.toggle( "hwpsequence", "sequence", false );

    CHECK_FALSE( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );

    widget.state( "qwptrack", "NOTCONNECTED" );

    CHECK( label->text() == "---" );
    CHECK( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );

    widget.state( "qwptrack", "READY" );
    widget.toggle( "qwptrack", "tracking", false );

    CHECK( label->isHidden() );
}

/** \brief The ttmperi offsets show beside the QWP pupil tracking slider only while tracking, and the
 * slider locks only when `qwppupil` is not READY/OPERATING, not while sequencing.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry QWP pupil tracking", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::handleSetProperty; xqt::polarimetry::updateGUI; xqt::polarimetry::pupilOffsetText;
#endif
    // clang-format on
    // one decimal, explicit sign, padded with figure spaces to a sign and three digits
    CHECK( xqt::polarimetry::pupilOffsetText( 1.23, -12.45 ) == QString::fromUtf8( "ΔV1:   +1.2 V  ΔV2:  −12.4 V" ) );
    CHECK( xqt::polarimetry::pupilOffsetText( -123.4, 0 ) == QString::fromUtf8( "ΔV1: −123.4 V  ΔV2:   +0.0 V" ) );
    CHECK( xqt::polarimetry::pupilOffsetText( -0.04, 0.04 ) == QString::fromUtf8( "ΔV1:   +0.0 V  ΔV2:   +0.0 V" ) );

    Widget widget;
    widget.ready();

    QLabel  *label  = widget.label( "labelQwpPupilTracking" );
    QWidget *slider = widget.control( "sliderQwpPupilTracking" );

    // qwppupil not yet READY
    CHECK( slider->isEnabled() );
    CHECK( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );
    CHECK( label->isHidden() );

    widget.state( "qwppupil", "READY" );
    widget.number( "qwppupil", "offset", { { "dx", 0.0 }, { "dy", 0.0 }, { "v1", 0.0 }, { "v2", 0.0 } } );

    CHECK_FALSE( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );
    CHECK( label->isHidden() );

    widget.toggle( "qwppupil", "tracking", true );
    widget.state( "qwppupil", "OPERATING" );
    widget.number( "qwppupil", "offset", { { "dx", 0.02 }, { "dy", -0.2 }, { "v1", 2.5 }, { "v2", -12.0 } } );

    CHECK_FALSE( label->isHidden() );
    CHECK( label->text() == xqt::polarimetry::pupilOffsetText( 2.5, -12.0 ) );

    // sequencing neither locks the slider nor greys the label
    widget.toggle( "hwpsequence", "sequence", true );

    CHECK_FALSE( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );
    CHECK( label->isEnabled() );

    widget.toggle( "qwppupil", "tracking", false );
    widget.state( "qwppupil", "READY" );

    CHECK( label->isHidden() );
    CHECK_FALSE( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );

    widget.toggle( "hwpsequence", "sequence", false );
    widget.state( "qwppupil", "NOTCONNECTED" );

    CHECK( slider->testAttribute( Qt::WA_TransparentForMouseEvents ) );

    widget.toggle( "qwppupil", "tracking", true );

    CHECK( label->text() == "---" );
}

/** \brief Buttons send the current hwpSequencer property names.
 * \ingroup polarimetryGUI_unit_test
 */
TEST_CASE( "Polarimetry button commands", "[polarimetryGUI]" )
{
    // clang-format off
#ifdef POLARIMETRY_GUI_TEST_DOXYGEN_REF
    xqt::polarimetry::on_buttonStartSequence_pressed; xqt::polarimetry::on_buttonStopSequence_pressed;
    xqt::polarimetry::on_buttonLastCycle_clicked; xqt::polarimetry::on_buttonAbortSequence_pressed;
#endif
    // clang-format on
    Widget widget;
    widget.ready();

    widget.on_buttonStartSequence_pressed();
    widget.on_buttonStopSequence_pressed();
    widget.on_buttonLastCycle_clicked( true );
    widget.on_buttonAbortSequence_pressed();

    REQUIRE( widget.m_sent.size() == 4 );

    const std::vector<std::pair<std::string, pcf::IndiElement::SwitchStateType>> expected = {
        { "sequence", pcf::IndiElement::On },
        { "sequence", pcf::IndiElement::Off },
        { "last_cycle", pcf::IndiElement::On },
        { "abort", pcf::IndiElement::On } };

    for( size_t n = 0; n < expected.size(); ++n )
    {
        INFO( n );
        CHECK( widget.m_sent[n].getDevice() == "hwpsequence" );
        CHECK( widget.m_sent[n].getName() == expected[n].first );
        REQUIRE( widget.m_sent[n].find( "toggle" ) );
        CHECK( widget.m_sent[n]["toggle"].getSwitchState() == expected[n].second );
    }
}

} // namespace polarimetryGUITest
} // namespace libXWCTest

/// Run offline Qt tests without an INDI connection, using Qt's offscreen backend.
int main( int argc, /**< [in] Argument count. */ char **argv /**< [in] Catch options. */ )
{
    qputenv( "QT_QPA_PLATFORM", "offscreen" );
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
