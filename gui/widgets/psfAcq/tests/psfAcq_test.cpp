/** \file psfAcq_test.cpp
 * \brief Offline star list, seeing display, seeing-star selection, and acquire command tests for the psfAcq GUI widget.
 * \ingroup psfAcqGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"
#include <QApplication>
#include "../psfAcq.hpp"

namespace libXWCTest
{
namespace psfAcqGUITest
{
/// \cond TEST_HARNESS
/// Capture widget commands locally without an INDI publisher.
class Widget : public xqt::psfAcq
{
  public:
    std::vector<pcf::IndiProperty> m_sent; ///< Typed outgoing commands.

    /// Create an offline psfAcq widget.
    explicit Widget( std::string &name /**< [in] psfAcq device. */ );

    /// Capture one command in memory.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Outgoing vector. */ ) override;

    /// Deliver the device FSM state.
    void fsm( const std::string &state /**< [in] FSM state string. */ );

    /// Define one `star_N` property with a seeing value and refresh the widget.
    void star( int n /**< [in] Star index. */, double seeing /**< [in] Seeing in arcsec. */ );

    /// Deliver the star the device uses for its published seeing and refresh the widget.
    void seeingStar( int n /**< [in] Star index, or -1 for none. */ );

    /// Delete one `star_N` property and refresh the widget.
    void delStar( int n /**< [in] Star index. */ );
};

Widget::Widget( std::string &name ) : psfAcq( name )
{
}

void Widget::sendNewProperty( const pcf::IndiProperty &property )
{
    m_sent.push_back( property );
}

void Widget::fsm( const std::string &state )
{
    pcf::IndiProperty p( pcf::IndiProperty::Text, "camacq-acq", "fsm" );
    p.add( pcf::IndiElement( "state", state ) );
    handleSetProperty( p );
    updateGUI();
}

void Widget::star( int n, double seeing )
{
    pcf::IndiProperty p( pcf::IndiProperty::Number, "camacq-acq", "star_" + std::to_string( n ) );
    for( const auto &el : { "x", "y", "peak", "fwhm" } )
    {
        p.add( pcf::IndiElement( el ) );
        p[el] = 1.0;
    }
    p.add( pcf::IndiElement( "seeing" ) );
    p["seeing"] = seeing;
    handleDefProperty( p );
    updateGUI();
}

void Widget::seeingStar( int n )
{
    pcf::IndiProperty p( pcf::IndiProperty::Number, "camacq-acq", "seeing_star" );
    p.add( pcf::IndiElement( "current" ) );
    p["current"] = n;
    handleSetProperty( p );
    updateGUI();
}

void Widget::delStar( int n )
{
    pcf::IndiProperty p( pcf::IndiProperty::Number, "camacq-acq", "star_" + std::to_string( n ) );
    handleDelProperty( p );
    updateGUI();
}
/// \endcond

/** \brief Only `star_N` names within the subscribed range map to a star index.
 * \ingroup psfAcqGUI_unit_test
 */
TEST_CASE( "psfAcq widget parses star property names", "[psfAcqGUI]" )
{
    CHECK( xqt::psfAcq::starIndex( "star_0" ) == 0 );
    CHECK( xqt::psfAcq::starIndex( "star_19" ) == 19 );
    CHECK( xqt::psfAcq::starIndex( "star_20" ) == -1 );
    CHECK( xqt::psfAcq::starIndex( "star_" ) == -1 );
    CHECK( xqt::psfAcq::starIndex( "star_1x" ) == -1 );
    CHECK( xqt::psfAcq::starIndex( "seeing_star" ) == -1 );
}

/** \brief Detected stars populate the list, star 0 is selected by default with its seeing shown, and selections are
 * sent.
 * \ingroup psfAcqGUI_unit_test
 */
TEST_CASE( "psfAcq widget selects a star, shows its seeing, and acquires it", "[psfAcqGUI]" )
{
    // clang-format off
#ifdef PSFACQ_GUI_TEST_DOXYGEN_REF
    xqt::psfAcq::handleSetProperty; xqt::psfAcq::handleDelProperty; xqt::psfAcq::updateGUI;
    xqt::psfAcq::on_button_acquire_pressed; xqt::psfAcq::on_combo_star_activated;
#endif
    // clang-format on
    std::string name = "camacq-acq";
    Widget      widget( name );

    auto *combo   = widget.findChild<QComboBox *>( "combo_star" );
    auto *seeing  = widget.findChild<QLabel *>( "val_seeing" );
    auto *acquire = widget.findChild<QPushButton *>( "button_acquire" );
    REQUIRE( combo );
    REQUIRE( seeing );
    REQUIRE( acquire );

    widget.star( 1, 1.25 );
    CHECK( combo->currentData().toInt() == 1 );

    widget.star( 0, 0.85 );
    CHECK( combo->count() == 2 );
    CHECK( combo->currentData().toInt() == 0 );
    CHECK_FALSE( acquire->isEnabled() );
    CHECK( seeing->text() == "0.85\"" );

    widget.fsm( "OPERATING" );
    CHECK( acquire->isEnabled() );
    CHECK( seeing->text() == "0.85\"" );
    CHECK( widget.m_sent.empty() );

    combo->setCurrentIndex( 1 );
    emit combo->activated( 1 );
    widget.updateGUI();
    CHECK( seeing->text() == "1.25\"" );
    REQUIRE( widget.m_sent.size() == 1 );
    CHECK( widget.m_sent.back().getName() == "seeing_star" );
    CHECK( widget.m_sent.back()["target"].get<int>() == 1 );
    widget.seeingStar( 1 );

    widget.star( 2, 2.0 );
    CHECK( combo->count() == 3 );
    CHECK( combo->currentData().toInt() == 1 );

    widget.on_button_acquire_pressed();
    REQUIRE( widget.m_sent.size() == 2 );
    CHECK( widget.m_sent.back().getDevice() == "camacq-acq" );
    CHECK( widget.m_sent.back().getName() == "acquire_star" );
    CHECK( widget.m_sent.back()["target"].get<int>() == 1 );

    widget.seeingStar( 0 );
    widget.delStar( 1 );
    CHECK( combo->count() == 2 );
    CHECK( combo->currentData().toInt() == 0 );
    CHECK( seeing->text() == "0.85\"" );

    widget.onDisconnect();
    CHECK( combo->count() == 0 );
    CHECK_FALSE( acquire->isEnabled() );
    widget.on_button_acquire_pressed();
    CHECK( widget.m_sent.size() == 2 );
}
} // namespace psfAcqGUITest
} // namespace libXWCTest

/// Run the offline widget tests with the Qt platform selected by the caller.
int main( int argc, /**< [in] Argument count. */ char **argv /**< [in] Catch options. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
