/** \file pwr_test.cpp
 * \brief Offline power GUI state, availability, and timeout tests.
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/catch2/catch.hpp"

#include <QApplication>

#include "../pwrDevice.hpp"

/** \defgroup pwrGUI_unit_test Power GUI Unit Tests
 * \ingroup application_unit_test
 */
namespace libXWCTest
{
namespace pwrGUITest
{

/** \brief Unknown and unrecognized Text states disable only their channel until a recognized state arrives.
 * \ingroup pwrGUI_unit_test
 */
TEST_CASE( "Power GUI decodes unknown channel states", "[pwrGUI]" )
{
    xqt::pwrDevice device;
    device.deviceName( "vpdu0" );
    device.setChannels( { "camera", "lamp" } );

    pcf::IndiProperty update( pcf::IndiProperty::Text );
    update.setDevice( "vpdu0" );
    update.setName( "lamp" );
    update.add( pcf::IndiElement( "state", "On" ) );
    device.handleSetProperty( update );
    REQUIRE( device.channel( 1 )->channelSwitch()->isEnabled() );

    update.setName( "camera" );
    auto *slider   = device.channel( 0 )->channelSwitch();
    int   commands = 0;
    QObject::connect( &device, &xqt::pwrDevice::chChange, [&commands]( pcf::IndiProperty & ) { ++commands; } );

    for( const auto &state : { "Unk", "unexpected", "", "ON" } )
    {
        update["state"] = "On";
        device.handleSetProperty( update );
        REQUIRE( slider->isEnabled() );
        REQUIRE( slider->sliderPosition() == slider->maximum() );

        update["state"] = state;
        device.handleSetProperty( update );
        CHECK_FALSE( slider->isEnabled() );
        CHECK( slider->sliderPosition() == slider->maximum() );
        CHECK( device.channel( 1 )->channelSwitch()->isEnabled() );

        pcf::IndiProperty target( pcf::IndiProperty::Text );
        target.setDevice( "vpdu0" );
        target.setName( "camera" );
        target.add( pcf::IndiElement( "target", "Off" ) );
        device.handleSetProperty( target );
        CHECK_FALSE( slider->isEnabled() );

        device.channel( 0 )->timeOut();
        device.channel( 0 )->sliderReleased();
        CHECK_FALSE( slider->isEnabled() );
        CHECK( commands == 0 );

        for( const auto &known : { "Int", "Off", "On" } )
        {
            update["state"] = state;
            device.handleSetProperty( update );
            update["state"] = known;
            device.handleSetProperty( update );
            REQUIRE( slider->isEnabled() );
            CHECK( slider->sliderPosition() == ( std::string( known ) == "On"    ? 10
                                                 : std::string( known ) == "Int" ? 5
                                                                                 : 0 ) );
        }
    }
}

/** \brief Unknown enum values keep the last displayed position disabled without confirming a target.
 * \ingroup pwrGUI_unit_test
 */
TEST_CASE( "Power sliders reject unknown enum states", "[pwrGUI]" )
{
    xqt::pwrChannel channel;
    auto           *slider  = channel.channelSwitch();
    int             reached = 0;
    QObject::connect( &channel, &xqt::pwrChannel::switchTargetReached, [&reached]() { ++reached; } );

    for( auto invalid :
         { xqt::pwrChState::Unk, static_cast<xqt::pwrChState>( -1 ), static_cast<xqt::pwrChState>( 99 ) } )
    {
        for( auto known : { xqt::pwrChState::Off, xqt::pwrChState::On, xqt::pwrChState::Int } )
        {
            channel.switchState( known );
            REQUIRE( slider->isEnabled() );
            int position = slider->sliderPosition();
            int previous = reached;
            channel.switchState( invalid );
            CHECK_FALSE( slider->isEnabled() );
            CHECK( slider->sliderPosition() == position );
            CHECK( reached == previous );
            channel.timeOut();
            CHECK_FALSE( slider->isEnabled() );
            CHECK( reached == previous );
        }
    }
}

/** \brief Losing observed state cancels a pending timeout; ordinary known-state transitions retain their wait behavior.
 * \ingroup pwrGUI_unit_test
 */
TEST_CASE( "Power sliders remain disabled after losing state during a command", "[pwrGUI]" )
{
    xqt::pwrChannel channel;
    auto           *slider   = channel.channelSwitch();
    int             commands = 0;
    QObject::connect( &channel, &xqt::pwrChannel::switchOn, [&commands]( const std::string & ) { ++commands; } );
    channel.switchState( xqt::pwrChState::Off );
    channel.switchTarget( xqt::pwrChState::On );
    slider->setSliderPosition( slider->maximum() );
    channel.sliderReleased();
    REQUIRE( commands == 1 );
    REQUIRE( channel.changing() );
    REQUIRE_FALSE( slider->isEnabled() );
    auto *timer = channel.findChild<QTimer *>();
    REQUIRE( timer != nullptr );
    REQUIRE( timer->isActive() );

    channel.switchState( xqt::pwrChState::Off );
    CHECK_FALSE( slider->isEnabled() );
    channel.switchState( xqt::pwrChState::Int );
    CHECK_FALSE( slider->isEnabled() );

    channel.switchState( xqt::pwrChState::Unk );
    CHECK_FALSE( channel.changing() );
    CHECK_FALSE( timer->isActive() );
    CHECK_FALSE( slider->isEnabled() );
    channel.timeOut();
    CHECK_FALSE( slider->isEnabled() );
    channel.sliderReleased();
    CHECK( commands == 1 );

    channel.switchState( xqt::pwrChState::Int );
    CHECK( slider->isEnabled() );
    channel.switchState( xqt::pwrChState::On );
    CHECK( slider->isEnabled() );
    CHECK_FALSE( channel.changing() );
}

/** \brief Recognized On/Off transitions still emit one command and wait for the matching observed target.
 * \ingroup pwrGUI_unit_test
 */
TEST_CASE( "Power sliders retain normal command completion", "[pwrGUI]" )
{
    xqt::pwrChannel channel;
    auto           *slider      = channel.channelSwitch();
    int             onCommands  = 0;
    int             offCommands = 0;
    QObject::connect( &channel, &xqt::pwrChannel::switchOn, [&onCommands]( const std::string & ) { ++onCommands; } );
    QObject::connect( &channel, &xqt::pwrChannel::switchOff, [&offCommands]( const std::string & ) { ++offCommands; } );
    channel.switchState( xqt::pwrChState::Off );

    for( auto target : { xqt::pwrChState::On, xqt::pwrChState::Off } )
    {
        channel.switchTarget( target );
        slider->setSliderPosition( target == xqt::pwrChState::On ? slider->maximum() : slider->minimum() );
        channel.sliderReleased();
        REQUIRE( channel.changing() );
        REQUIRE_FALSE( slider->isEnabled() );
        channel.switchState( xqt::pwrChState::Int );
        CHECK( channel.changing() );
        CHECK_FALSE( slider->isEnabled() );
        channel.switchState( target );
        CHECK_FALSE( channel.changing() );
        CHECK( slider->isEnabled() );
    }
    CHECK( onCommands == 1 );
    CHECK( offCommands == 1 );
}

/** \brief Initial and disconnected channels cannot be enabled by a timeout without a recognized observation.
 * \ingroup pwrGUI_unit_test
 */
TEST_CASE( "Power sliders wait for observed state on startup and reconnect", "[pwrGUI]" )
{
    xqt::pwrChannel channel;
    auto           *slider = channel.channelSwitch();
    CHECK_FALSE( slider->isEnabled() );
    channel.timeOut();
    CHECK_FALSE( slider->isEnabled() );
    channel.switchState( xqt::pwrChState::On );
    REQUIRE( slider->isEnabled() );
    channel.onDisconnect();
    CHECK_FALSE( slider->isEnabled() );
    channel.timeOut();
    CHECK_FALSE( slider->isEnabled() );
    channel.switchState( xqt::pwrChState::Int );
    CHECK( slider->isEnabled() );
}

} // namespace pwrGUITest
} // namespace libXWCTest

/// Run widget tests with a local Qt event loop and no INDI connection.
int main( int    argc, /**< [in] Number of command-line arguments. */
          char **argv /**< [in] Qt and Catch test arguments. */ )
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
