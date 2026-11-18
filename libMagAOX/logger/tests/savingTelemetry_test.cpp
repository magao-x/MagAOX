/** \file savingTelemetry_test.cpp
 * \brief Saving-state snapshot formatting and start/stop event compatibility tests.
 * \ingroup logger_files
 */
#include "../../../tests/testXWC.hpp"

#include "../generated/logStdFormat.hpp"
#include "../generated/logVerify.hpp"

namespace libXWCTest
{
namespace loggerTest
{
/** \addtogroup logTypes_unit_test
 * @{ */
namespace savingTelemetryTest
{
using namespace MagAOX::logger;

/** \brief Format active and inactive snapshots through the real flatlog event dispatch.
 * \ingroup logTypes_unit_test */
TEST_CASE( "saving telemetry reports activity without implying a transition", "[savingTelemetry]" )
{
    const int16_t                state = GENERATE( 0, 1 );
    const uint64_t               frame = GENERATE( uint64_t{ 0 }, uint64_t{ 1761 }, ( uint64_t{ 1 } << 40 ) + 1761 );
    telem_saving_state::messageT message( state, frame );
    const std::string            expected = std::string( "Saving state: " ) + ( state == 0 ? "inactive" : "active" ) +
                                 "; save-window end frame: " + std::to_string( frame );
    REQUIRE( telem_saving_state::msgString( message.builder.GetBufferPointer(), message.builder.GetSize() ) ==
             expected );

    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_saving_state>(
                 buffer, { 1, 0 }, message, flatlogs::logPrio::LOG_TELEM ) == 0 );
    REQUIRE( logVerify( telem_saving_state::eventCode, buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    const auto *payload = GetSaving_state_change_fb( flatlogs::logHeader::messageBuffer( buffer ) );
    REQUIRE( payload->state() == state );
    REQUIRE( payload->frameNo() == frame );

    std::ostringstream text;
    logStdFormat( text, buffer );
    REQUIRE( text.str().find( " TELM " + expected ) != std::string::npos );
    REQUIRE( text.str().find( "started" ) == std::string::npos );
    REQUIRE( text.str().find( "stopped" ) == std::string::npos );
}

/** \brief Keep the NOTICE event wording for actual start and stop transitions.
 * \ingroup logTypes_unit_test */
TEST_CASE( "saving events retain their start and stop transition messages", "[savingTelemetry]" )
{
    saving_start::messageT start( 1, 1685 );
    saving_stop::messageT  stop( 0, 1761 );
    flatlogs::bufferPtrT   buffer;
    REQUIRE( flatlogs::logHeader::createLog<saving_start>( buffer, { 1, 0 }, start, flatlogs::logPrio::LOG_NOTICE ) ==
             0 );
    std::ostringstream starting;
    logStdFormat( starting, buffer );
    REQUIRE( starting.str().find( " NOTE Saving started at frame number 1685" ) != std::string::npos );

    REQUIRE( flatlogs::logHeader::createLog<saving_stop>( buffer, { 2, 0 }, stop, flatlogs::logPrio::LOG_NOTICE ) ==
             0 );
    std::ostringstream stopping;
    logStdFormat( stopping, buffer );
    REQUIRE( stopping.str().find( " NOTE Saving stopped at frame number 1761" ) != std::string::npos );
}
} // namespace savingTelemetryTest
/** @} */
} // namespace loggerTest
} // namespace libXWCTest
