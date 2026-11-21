/** \file elliptecDevice_test.cpp
 * \brief Catch2 unit tests for the per-address Elliptec device state.
 * \author Miles Lucas
 *
 * \ingroup elliptecLowLevel_files
 */

/** \defgroup elliptecLowLevel_unit_test elliptecLowLevel Unit Tests
 * \brief Unit tests for the elliptecLowLevel application and its device state.
 *
 * \ingroup application_unit_test
 */

#include "../../../tests/catch2/catch.hpp"

#include <thread>

#include "../elliptecDevice.hpp"

namespace libXWCTest
{
namespace elliptecLowLevelTest
{

using MagAOX::app::elliptecDevice;
namespace elliptec = MagAOX::app::elliptec;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace
{

/// A valid ELL21 info payload with the given serial.
std::string infoPayload( const std::string &serial = "12100070" )
{
    return "15" + serial + "2025" + "13" + "01" + "0168" + "00010000";
}

/// A device identified as an ELL21 at address 2.
elliptecDevice identifiedDevice()
{
    elliptecDevice dev;
    dev.name( "stage" );
    dev.address( '2' );
    REQUIRE( dev.applyInfo( infoPayload() ) == elliptecDevice::InfoResult::Ok );
    return dev;
}

/// Hex position payload.
std::string posPayload( int32_t p )
{
    char buf[9];
    std::snprintf( buf, sizeof( buf ), "%08X", static_cast<uint32_t>( p ) );
    return buf;
}

/// Report a fully answered poll with a position and status.
elliptecDevice::Event answer( elliptecDevice &dev, int32_t pos, uint8_t status )
{
    dev.applyPosition( posPayload( pos ) );
    dev.applyStatus( status );
    return dev.pollAnswered( 0.5 );
}

/// Wait past the minimum completion time.
void waitCompletionGuard()
{
    std::this_thread::sleep_for(
        std::chrono::milliseconds( static_cast<int>( elliptecDevice::s_minCompletionSec * 1000 ) + 50 ) );
}

} // namespace
/// \endcond

/// Protocol tables, angle helpers, and hex parsing.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptec helpers: status codes, velocity floors, angles, hex", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptec::statusMeaning( 0 );
    MagAOX::app::elliptec::minVelocityForModel( 21 );
    MagAOX::app::elliptec::wrapDeg( 0 );
    MagAOX::app::elliptec::circularDistanceDeg( 0, 0 );
    MagAOX::app::elliptec::parseHex( "", 0, 0, v );
    // clang-format on
#endif

    for( int code = 0; code <= 14; ++code )
    {
        REQUIRE_FALSE( elliptec::statusMeaning( static_cast<uint8_t>( code ) ).empty() );
    }
    REQUIRE( elliptec::statusMeaning( 2 ) == "mechanical time out" );
    REQUIRE( elliptec::statusMeaning( 200 ) == "reserved status 200" );

    REQUIRE( elliptec::minVelocityForModel( 21 ) == 50 );
    REQUIRE( elliptec::minVelocityForModel( 16 ) == 50 );
    REQUIRE( elliptec::minVelocityForModel( 22 ) == 50 );
    REQUIRE( elliptec::minVelocityForModel( 14 ) == 0 );

    REQUIRE( elliptec::wrapDeg( -45.0 ) == Approx( 315.0 ) );
    REQUIRE( elliptec::wrapDeg( 720.0 ) == Approx( 0.0 ) );
    REQUIRE( elliptec::wrapDeg( -1e-14 ) < 360.0 );
    REQUIRE( elliptec::circularDistanceDeg( 359.9, 0.05 ) == Approx( 0.15 ) );
    REQUIRE( elliptec::circularDistanceDeg( 10.0, 190.0 ) == Approx( 180.0 ) );

    uint32_t v = 0;
    REQUIRE( elliptec::parseHex( "0aF3", 0, 4, v ) );
    REQUIRE( v == 0x0AF3 );
    REQUIRE_FALSE( elliptec::parseHex( "0G", 0, 2, v ) );
    REQUIRE_FALSE( elliptec::parseHex( "12", 1, 2, v ) );
    REQUIRE_FALSE( elliptec::parseHex( "123456789", 0, 9, v ) );
    REQUIRE_FALSE( elliptec::parseHex( "12", 0, 0, v ) );
}

/// Configuration accessors: address validation and the velocity floor.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice configuration and velocity floor", "[elliptecLowLevel]" )
{
    elliptecDevice dev;
    dev.name( "qwp1" );
    REQUIRE( dev.name() == "qwp1" );

    REQUIRE( dev.address( 'b' ) == 0 );
    REQUIRE( dev.address() == 'B' );
    REQUIRE( dev.address( '"' ) == -1 );
    REQUIRE( dev.address() == 'B' );

    dev.expectedSerial( "12100070" );
    REQUIRE( dev.expectedSerial() == "12100070" );

    // Before identification no model minimum applies; clamped to [0, 100].
    REQUIRE( dev.velPercent( 40 ) == 40 );
    REQUIRE( dev.velPercent( 140 ) == 100 );
    REQUIRE( dev.velPercent( -5 ) == 0 );

    // Identifying an ELL21 raises the stored velocity to its 50% minimum.
    REQUIRE( dev.applyInfo( infoPayload() ) == elliptecDevice::InfoResult::Ok );
    REQUIRE( dev.velPercent() == 50 );
    REQUIRE( dev.velPercent( 10 ) == 50 );
    REQUIRE( dev.velPercent( 80 ) == 80 );
}

/// The info reply is validated in full; truncated replies from a shared bus are rejected.
/** Regression for the hardware trace where `2IN15121000702000` was parsed as 7348224 pulses/rev.
 *
 * \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice rejects malformed info and checks the serial", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecDevice::applyInfo( "" );
    // clang-format on
#endif

    elliptecDevice dev;
    REQUIRE( dev.applyInfo( "15121000702000" ) == elliptecDevice::InfoResult::Malformed );
    REQUIRE( dev.applyInfo( "151210007020251301016800010000X" ) == elliptecDevice::InfoResult::Malformed );
    REQUIRE( dev.applyInfo( "1512100070202513010168000100ZZ" ) == elliptecDevice::InfoResult::Malformed );
    REQUIRE( dev.applyInfo( "151210007020251301016800000000" ) == elliptecDevice::InfoResult::Malformed ); // 0 ppr
    REQUIRE_FALSE( dev.identified() );

    REQUIRE( dev.applyInfo( "151210007020251301016800010000" ) == elliptecDevice::InfoResult::Ok );
    REQUIRE( dev.identified() );
    REQUIRE( dev.model() == 21 );
    REQUIRE( dev.serial() == "12100070" );
    REQUIRE( dev.pulsesPerRev() == 65536 );
    REQUIRE( dev.stateString() == "NOTHOMED" );

    elliptecDevice other;
    other.expectedSerial( "12100017" );
    REQUIRE( other.applyInfo( infoPayload( "12100070" ) ) == elliptecDevice::InfoResult::SerialMismatch );
    REQUIRE( other.wrongDevice() );
    REQUIRE_FALSE( other.identified() );
    REQUIRE( other.stateString() == "NODEVICE" );
    REQUIRE_FALSE( other.motionAllowed() );

    // Power-off clears the mismatch so the bus is checked again.
    other.powerOff();
    REQUIRE_FALSE( other.wrongDevice() );
    REQUIRE( other.stateString() == "NOTCONNECTED" );
}

/// Position parsing, wrapping, and degree conversion.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice position and conversions", "[elliptecLowLevel]" )
{
    elliptecDevice dev = identifiedDevice();

    REQUIRE( dev.applyPosition( "00004000" ) );
    REQUIRE( dev.positionDeg() == Approx( 90.0 ) );

    REQUIRE( dev.applyPosition( "FFFFFFCE" ) ); // -50 pulses
    REQUIRE( dev.positionPulses() == -50 );
    REQUIRE( dev.positionDeg() == Approx( 359.725 ).margin( 0.001 ) );

    REQUIRE( dev.applyPosition( "00010400" ) ); // more than one turn
    REQUIRE( dev.positionDeg() == Approx( 5.625 ) );

    REQUIRE_FALSE( dev.applyPosition( "00Z00000" ) );
    REQUIRE_FALSE( dev.applyPosition( "123" ) );
    REQUIRE( dev.positionPulses() == 0x10400 );

    REQUIRE( dev.degToPulses( 90.0 ) == 0x4000 );
    REQUIRE( dev.degToPulses( -90.0 ) == 0xC000 );
    REQUIRE( dev.degToPulses( 359.9999999 ) == 0 );

    elliptecDevice unknown;
    REQUIRE( unknown.positionDeg() == Approx( 0.0 ) );
}

/// Moves complete only when idle and holding still at the target.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice completes moves by position, not status alone", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecDevice::pollAnswered( 0.5 );
    MagAOX::app::elliptecDevice::beginMove( 0 );
    // clang-format on
#endif

    SECTION( "a false idle mid-move does not complete it" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginMove( 0x4000 );
        REQUIRE( dev.stateString() == "OPERATING" );
        REQUIRE( dev.moving() == 1 );

        waitCompletionGuard();
        REQUIRE( answer( dev, 0x1000, 0 ) == elliptecDevice::Event::None ); // moving, falsely idle
        REQUIRE( answer( dev, 0x2000, 0 ) == elliptecDevice::Event::None ); // still moving
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::None ); // arrived, first idle-at-rest
        REQUIRE( answer( dev, 0x4001, 0 ) == elliptecDevice::Event::Completed );
        REQUIRE( dev.eventText() == "move complete" );
        REQUIRE( dev.pending() == elliptecDevice::Pending::None );
        REQUIRE( dev.stateString() == "NOTHOMED" );
    }

    SECTION( "idle and still right after the command waits for the start guard" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginMove( 0x4000 );
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::None );
        waitCompletionGuard();
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::Completed );
    }

    SECTION( "busy status resets the stillness count" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginMove( 0x4000 );
        waitCompletionGuard();
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0x4000, 9 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0x4000, 0 ) == elliptecDevice::Event::Completed );
    }

    SECTION( "stopping away from the target fails, unless a stop was requested" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginMove( 0x4000 );
        waitCompletionGuard();
        REQUIRE( answer( dev, 0x0100, 0 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0x0100, 0 ) == elliptecDevice::Event::Failed );
        REQUIRE( dev.eventText() == "move failed: stopped away from target" );
        REQUIRE( dev.lastError() == "move failed (stopped away from target)" );

        dev.beginMove( 0x4000 );
        REQUIRE( dev.lastError().empty() );
        REQUIRE( dev.stopRequested( 0 ) == false );
        waitCompletionGuard();
        REQUIRE( answer( dev, 0x0100, 0 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0x0100, 0 ) == elliptecDevice::Event::Completed );
        REQUIRE( dev.eventText() == "move stopped by request before reaching its target" );
    }

    SECTION( "an error status fails the move and keeps the homed flag" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginHome();
        waitCompletionGuard();
        answer( dev, 0, 0 );
        REQUIRE( answer( dev, 0, 0 ) == elliptecDevice::Event::Completed );
        REQUIRE( dev.homed() );
        REQUIRE( dev.eventText() == "homing complete" );

        dev.beginMove( 0x4000 );
        REQUIRE( answer( dev, 0x1000, 0x0C ) == elliptecDevice::Event::Failed );
        REQUIRE( dev.eventText() == "move failed: out of range" );
        REQUIRE( dev.homed() );
        REQUIRE( dev.stateString() == "READY" );
    }
}

/// Homes and optimizes: targets, failures, and the homed flag.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice homing and optimize", "[elliptecLowLevel]" )
{
    SECTION( "a home targets position 0 and a failed home leaves the stage not homed" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginHome();
        REQUIRE( dev.hasTarget() );
        REQUIRE( dev.targetPulses() == 0 );
        REQUIRE( dev.stateString() == "HOMING" );
        REQUIRE( dev.moving() == 2 );
        REQUIRE_FALSE( dev.motionAllowed() );
        REQUIRE( dev.eventText() == "homing in progress" );

        REQUIRE( answer( dev, 0x3000, 0x02 ) == elliptecDevice::Event::Failed );
        REQUIRE( dev.lastError() == "home failed (mechanical time out)" );
        REQUIRE_FALSE( dev.homed() );
        REQUIRE( dev.stateString() == "NOTHOMED" );
    }

    SECTION( "optimize completes on idle status and requires a re-home" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginHome();
        waitCompletionGuard();
        answer( dev, 0, 0 );
        answer( dev, 0, 0 );
        REQUIRE( dev.homed() );

        dev.beginOptimize();
        REQUIRE_FALSE( dev.hasTarget() );
        REQUIRE_FALSE( dev.motionAllowed() );
        REQUIRE( dev.eventText() == "motor optimization in progress" );
        REQUIRE( answer( dev, 0, 0 ) == elliptecDevice::Event::None ); // too soon, no busy seen
        REQUIRE( answer( dev, 0, 9 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0, 0 ) == elliptecDevice::Event::Completed );
        REQUIRE_FALSE( dev.homed() );
        REQUIRE( dev.eventText() == "motor optimization complete; the stage must be re-homed" );
    }

    SECTION( "an unsupported stop is reported" )
    {
        elliptecDevice dev = identifiedDevice();
        REQUIRE( dev.stopRequested( 3 ) ); // idle: nothing to stop, but the reply is still reported
        dev.beginMove( 0x4000 );
        REQUIRE( dev.stopRequested( 9 ) == false );
        REQUIRE( dev.stopRequested( 3 ) == true );
        REQUIRE( dev.pending() == elliptecDevice::Pending::Move );
    }
}

/// Missed replies: tolerated while an operation is in progress, budgeted while idle.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice missed replies and timeouts", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecDevice::pollMissed( 0, 0, 0 );
    // clang-format on
#endif

    SECTION( "silence during a move is tolerated until the motion timeout" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginMove( 0x4000 );
        for( int n = 0; n < 10; ++n )
        {
            REQUIRE( dev.pollMissed( 30, 600, 1 ) == elliptecDevice::Event::None );
        }
        REQUIRE( dev.pending() == elliptecDevice::Pending::Move );

        REQUIRE( dev.pollMissed( 0, 600, 1 ) == elliptecDevice::Event::Failed );
        REQUIRE( dev.lastError() == "move failed (no response from the device during motion)" );
        REQUIRE( dev.pollMissed( 0, 600, 1 ) == elliptecDevice::Event::Lost );
        REQUIRE_FALSE( dev.identified() );
    }

    SECTION( "silence during optimize uses the optimize timeout" )
    {
        elliptecDevice dev = identifiedDevice();
        dev.beginOptimize();
        REQUIRE( dev.pollMissed( 0, 600, 1 ) == elliptecDevice::Event::None );
        REQUIRE( dev.pollMissed( 600, 0, 1 ) == elliptecDevice::Event::Failed );
        REQUIRE( dev.eventText() == "optimize failed: motor optimization timed out" );
    }

    SECTION( "idle misses are budgeted, and an answered poll resets the count" )
    {
        elliptecDevice dev = identifiedDevice();
        REQUIRE( dev.pollMissed( 30, 600, 2 ) == elliptecDevice::Event::None );
        REQUIRE( dev.pollMissed( 30, 600, 2 ) == elliptecDevice::Event::None );
        REQUIRE( answer( dev, 0, 0 ) == elliptecDevice::Event::None );
        REQUIRE( dev.pollMissed( 30, 600, 2 ) == elliptecDevice::Event::None );
        REQUIRE( dev.pollMissed( 30, 600, 2 ) == elliptecDevice::Event::None );
        REQUIRE( dev.pollMissed( 30, 600, 2 ) == elliptecDevice::Event::Lost );
        REQUIRE( dev.eventText() == "device stopped answering" );
        REQUIRE( dev.stateString() == "NOTCONNECTED" );
    }
}

/// Status with nothing commanded: external motion and unsolicited errors.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice idle status reports", "[elliptecLowLevel]" )
{
    elliptecDevice dev = identifiedDevice();

    REQUIRE( answer( dev, 0, 9 ) == elliptecDevice::Event::None );
    REQUIRE( dev.stateString() == "OPERATING" ); // moved by something else
    REQUIRE( answer( dev, 0, 0 ) == elliptecDevice::Event::None );
    REQUIRE( dev.stateString() == "NOTHOMED" );

    REQUIRE( answer( dev, 0, 0x08 ) == elliptecDevice::Event::Failed );
    REQUIRE( dev.lastError() == "device reported thermal error" );
    REQUIRE( answer( dev, 0, 0x08 ) == elliptecDevice::Event::None ); // reported once

    dev.disconnect();
    REQUIRE_FALSE( dev.identified() );
    REQUIRE( dev.lastError() == "device reported thermal error" );
    dev.powerOff();
    REQUIRE( dev.lastError().empty() );
}

/// A reconnect keeps the homed flag; power-off clears it.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecDevice homed flag across disconnect and power-off", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecDevice::disconnect();
    MagAOX::app::elliptecDevice::powerOff();
    // clang-format on
#endif

    elliptecDevice dev = identifiedDevice();
    dev.beginHome();
    waitCompletionGuard();
    answer( dev, 0, 0 );
    answer( dev, 0, 0 );
    REQUIRE( dev.homed() );

    dev.beginMove( 0x4000 );
    dev.disconnect();
    REQUIRE( dev.pending() == elliptecDevice::Pending::None );
    REQUIRE( dev.homed() );

    REQUIRE( dev.applyInfo( infoPayload() ) == elliptecDevice::InfoResult::Ok );
    REQUIRE( dev.stateString() == "READY" );

    dev.powerOff();
    REQUIRE_FALSE( dev.homed() );
    REQUIRE( dev.stateString() == "NOTCONNECTED" );
}

} // namespace elliptecLowLevelTest
} // namespace libXWCTest
