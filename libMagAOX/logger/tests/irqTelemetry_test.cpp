/** \file irqTelemetry_test.cpp
 * \brief Compact IRQ flags, actual check times and legacy-record compatibility.
 * \ingroup irqTelemetry_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

#include <bit>
#include <cstdint>
#include <limits>

namespace libXWCTest
{
namespace loggerTest
{
using namespace MagAOX::logger;

/// \cond TEST_HARNESS
/// Build the original field-zero-only root layout independently of the current message constructor.
telem_irq::messageT legacyMessage( const std::vector<telem_irq::groupT> &groups /**< [in] legacy placement records */ )
{
    telem_irq::messageT message( {}, 0 );
    auto               &builder = message.builder;
    builder.Clear();
    std::vector<flatbuffers::Offset<IrqGroup>> records;

    for( const auto &group : groups )
    {
        std::vector<flatbuffers::Offset<IrqAffinity>> irqs;

        for( const auto &irq : group.m_irqs )
        {
            auto requested = builder.CreateString( irq.m_requested );
            auto effective = builder.CreateString( irq.m_effective );
            irqs.push_back( CreateIrqAffinity( builder, irq.m_number, requested, effective ) );
        }

        auto name     = builder.CreateString( group.m_name );
        auto expected = builder.CreateString( group.m_expected );
        auto status   = builder.CreateString( group.m_status );
        auto vectors  = builder.CreateVector( irqs );
        records.push_back(
            CreateIrqGroup( builder, name, expected, status, group.m_warning, group.m_checkTime, vectors ) );
    }

    auto vector = builder.CreateVector( records );
    auto start  = builder.StartTable();
    builder.AddOffset( 4, vector ); // Original schema slot zero; never repurpose it for flags.
    builder.Finish( flatbuffers::Offset<Telem_irq_fb>( builder.EndTable( start ) ) );
    return message;
}

/// Wrap a diagnostic payload in its real logger envelope for verification dispatch.
flatlogs::bufferPtrT envelope( const telem_irq::messageT &message /**< [in] serialized IRQ message */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_irq>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}
/// \endcond

/// Compact records contain only ordered policy flags and the actual check time.
/** \ingroup irqTelemetry_unit_test */
TEST_CASE( "IRQ telemetry stores compact flags without placement metadata", "[irqTelemetry]" )
{
    telem_irq::messageT message( { 1, 0, 1 }, 1234.5 );
    auto                payload = message.builder.GetBufferPointer();
    const auto         *record  = GetTelem_irq_fb( payload );
    REQUIRE( record->groups() == nullptr );
    REQUIRE( record->ok() != nullptr );
    REQUIRE( telem_irq::ok( payload ) == std::vector<unsigned>{ 1, 0, 1 } );
    REQUIRE( telem_irq::checkTime( payload ) == 1234.5 );
    REQUIRE( telem_irq::checkTimes( payload ) == std::vector<double>{ 1234.5, 1234.5, 1234.5 } );
    REQUIRE( telem_irq::warning( payload ) );
    REQUIRE( telem_irq::groups( payload ).empty() );
    REQUIRE( telem_irq::names( payload ).empty() );
    REQUIRE( telem_irq::statuses( payload ).empty() );
    REQUIRE( telem_irq::expected( payload ).empty() );
    REQUIRE( telem_irq::msgString( payload, message.builder.GetSize() ) == "[irq] ok=1,0,1 checked=1234.500000" );
    auto buffer = envelope( message );
    REQUIRE( verifyLogEntry( telem_irq::eventCode, buffer.get() ) );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "ok" ).valType == logMeta::valTypes::Vector_UInt );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "ok" ).accessor == reinterpret_cast<void *>( &telem_irq::ok ) );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "checkTime" ).valType == logMeta::valTypes::Double );
    REQUIRE( logMemberAccessor( telem_irq::eventCode, "checkTime" ).accessor ==
             reinterpret_cast<void *>( &telem_irq::checkTime ) );

    auto json = telem_irq::msgJSON(
        payload, message.builder.GetSize(), reinterpret_cast<const uint8_t *>( telem_irq_bfbs ), telem_irq_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_irq_bfbs ), telem_irq_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    REQUIRE( telem_irq::ok( parser.builder_.GetBufferPointer() ) == std::vector<unsigned>{ 1, 0, 1 } );
    REQUIRE( telem_irq::checkTime( parser.builder_.GetBufferPointer() ) == 1234.5 );

    telem_irq::messageT healthy( { 1, 1 }, 1235 );
    REQUIRE_FALSE( telem_irq::warning( healthy.builder.GetBufferPointer() ) );
}

/// Nested IRQ snapshots preserve logical identities, failure state and actual check times through standard dispatch.
/** \ingroup irqTelemetry_unit_test */
TEST_CASE( "IRQ telemetry still decodes the original identified record layout", "[irqTelemetry]" )
{
    std::vector<telem_irq::groupT> groups{ { "camera", "34", "ok", false, 100.25, { { 405, "34", "34" } } },
                                           { "network",
                                             "58-59",
                                             "IRQ706: effective affinity mismatch",
                                             true,
                                             101.5,
                                             { { 705, "58-59", "58" }, { 706, "58-59", "60" } } },
                                           { "absent", "", "required device missing", true, 102, {} } };
    auto                           message = legacyMessage( groups );
    auto                           payload = message.builder.GetBufferPointer();
    auto                           decoded = telem_irq::groups( payload );
    REQUIRE( decoded == groups );
    REQUIRE( GetTelem_irq_fb( payload )->ok() == nullptr );
    REQUIRE( telem_irq::ok( payload ) == std::vector<unsigned>{ 1, 0, 0 } );
    REQUIRE( telem_irq::checkTime( payload ) == 102 );
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

/// Compact flags must be boolean, the representation must be unambiguous, and check times must be finite.
/** \ingroup irqTelemetry_unit_test */
TEST_CASE( "IRQ telemetry rejects malformed compact records", "[irqTelemetry]" )
{
    telem_irq::messageT message( { 2 }, 1 );
    auto                buffer = envelope( message );
    REQUIRE_FALSE( verifyLogEntry( telem_irq::eventCode, buffer.get() ) );

    for( double time : { -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        INFO( "Invalid check time=" << time << " bits=" << std::bit_cast<std::uint64_t>( time ) );
        telem_irq::messageT invalidTime( {}, 0 );
        invalidTime.builder.Clear();
        // Preserve NaN bits instead of relying on FlatBuffers' default-value comparison under fast math.
        invalidTime.builder.ForceDefaults( true );
        auto flags = invalidTime.builder.CreateVector( std::vector<uint8_t>{ 1 } );
        invalidTime.builder.Finish( CreateTelem_irq_fb( invalidTime.builder, 0, flags, time ) );
        REQUIRE( std::bit_cast<std::uint64_t>( telem_irq::checkTime( invalidTime.builder.GetBufferPointer() ) ) ==
                 std::bit_cast<std::uint64_t>( time ) );
        buffer = envelope( invalidTime );
        REQUIRE_FALSE( verifyLogEntry( telem_irq::eventCode, buffer.get() ) );
    }

    message.builder.Clear();
    message.builder.Finish( CreateTelem_irq_fb( message.builder ) );
    buffer = envelope( message );
    REQUIRE_FALSE( verifyLogEntry( telem_irq::eventCode, buffer.get() ) );

    message.builder.Clear();
    auto groups = message.builder.CreateVector( std::vector<flatbuffers::Offset<IrqGroup>>{} );
    auto flags  = message.builder.CreateVector( std::vector<uint8_t>{ 1 } );
    message.builder.Finish( CreateTelem_irq_fb( message.builder, groups, flags, 1 ) );
    buffer = envelope( message );
    REQUIRE_FALSE( verifyLogEntry( telem_irq::eventCode, buffer.get() ) );
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

    telem_irq::messageT empty( {}, 0 );
    auto                payload = empty.builder.GetBufferPointer();
    REQUIRE( telem_irq::groups( payload ).empty() );
    REQUIRE_FALSE( telem_irq::warning( payload ) );
    REQUIRE( telem_irq::msgString( payload, empty.builder.GetSize() ) == "[irq] ok= checked=0.000000" );
    auto buffer = envelope( empty );
    REQUIRE( verifyLogEntry( telem_irq::eventCode, buffer.get() ) );
}

} // namespace loggerTest
} // namespace libXWCTest
