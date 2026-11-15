/** \file irqTelemetry_test.cpp
 * \brief IRQ telemetry identity, validity, timestamp and dispatch verification.
 * \ingroup irqTelemetry_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{
using namespace MagAOX::logger;

/// Nested IRQ snapshots preserve logical identities, failure state and actual check times through standard dispatch.
/** \ingroup irqTelemetry_unit_test */
TEST_CASE( "IRQ telemetry preserves identified placement snapshots", "[irqTelemetry]" )
{
    std::vector<telem_irq::groupT> groups{ { "camera", "34", "ok", false, 100.25, { { 405, "34", "34" } } },
                                           { "network",
                                             "58-59",
                                             "IRQ706: effective affinity mismatch",
                                             true,
                                             101.5,
                                             { { 705, "58-59", "58" }, { 706, "58-59", "60" } } },
                                           { "absent", "", "required device missing", true, 102, {} } };
    telem_irq::messageT            message( groups );
    auto                           payload = message.builder.GetBufferPointer();
    auto                           decoded = telem_irq::groups( payload );
    REQUIRE( decoded == groups );
    REQUIRE( telem_irq::checkTimes( payload ) == std::vector<double>{ 100.25, 101.5, 102 } );
    REQUIRE( decoded[1].m_checkTime == groups[1].m_checkTime );
    REQUIRE( telem_irq::names( payload ) == std::vector<std::string>{ "camera", "network", "absent" } );
    REQUIRE( telem_irq::expected( payload ) == std::vector<std::string>{ "34", "58-59", "" } );
    REQUIRE( telem_irq::statuses( payload )[2] == "required device missing" );
    REQUIRE( telem_irq::warning( payload ) );
    auto text = telem_irq::msgString( payload, message.builder.GetSize() );
    REQUIRE( text.find( "network expected=58-59" ) != std::string::npos );
    REQUIRE( text.find( "IRQ706 requested=58-59 effective=60" ) != std::string::npos );

    flatlogs::bufferPtrT envelope;
    REQUIRE( flatlogs::logHeader::createLog<telem_irq>( envelope, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    REQUIRE( telem_irq::verify( envelope, flatlogs::logHeader::msgLen( envelope ) ) );
    REQUIRE( verifyLogEntry( telem_irq::eventCode, envelope.get() ) );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "names" ).valType == logMeta::valTypes::Vector_String );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "checkTimes" ).valType == logMeta::valTypes::Vector_Double );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "warning" ).accessor ==
             reinterpret_cast<void *>( &telem_irq::warning ) );
    REQUIRE( telem_irq::getAccessor( "unknown" ).accessor == nullptr );

    auto json = telem_irq::msgJSON(
        payload, message.builder.GetSize(), reinterpret_cast<const uint8_t *>( telem_irq_bfbs ), telem_irq_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_irq_bfbs ), telem_irq_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    REQUIRE( telem_irq::groups( parser.builder_.GetBufferPointer() ) == groups );
    REQUIRE( telem_irq::checkTimes( parser.builder_.GetBufferPointer() )[1] == 101.5 );

    REQUIRE_FALSE( telem_irq::verify( envelope, 4 ) );
    auto raw = static_cast<unsigned char *>( flatlogs::logHeader::messageBuffer( envelope ) );
    raw[0]   = 0xff;
    raw[1]   = 0xff;
    raw[2]   = 0xff;
    raw[3]   = 0x7f;
    REQUIRE_FALSE( verifyLogEntry( telem_irq::eventCode, envelope.get() ) );
}

/// State comparisons suppress unchanged checks but retain IRQ identity and validity changes.
/** \ingroup irqTelemetry_unit_test */
TEST_CASE( "IRQ telemetry state comparison excludes check time only", "[irqTelemetry]" )
{
    telem_irq::groupT first{ "camera", "34", "ok", false, 100, { { 405, "34", "34" } } };
    auto              later = first;
    later.m_checkTime       = 200;
    REQUIRE( first == later );
    later.m_irqs[0].m_number = 505;
    REQUIRE_FALSE( first == later );
    later           = first;
    later.m_warning = true;
    REQUIRE_FALSE( first == later );
    later          = first;
    later.m_status = "unverified";
    REQUIRE_FALSE( first == later );

    telem_irq::messageT empty( {} );
    auto                payload = empty.builder.GetBufferPointer();
    REQUIRE( telem_irq::groups( payload ).empty() );
    REQUIRE_FALSE( telem_irq::warning( payload ) );
    REQUIRE( telem_irq::msgString( payload, empty.builder.GetSize() ) == "[irq]" );
}

} // namespace loggerTest
} // namespace libXWCTest
