/** \file raidTelemetry_test.cpp
 * \brief Compact RAID health, semantic verification, metadata, and logger dispatch tests.
 * \ingroup raidTelemetry_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../libMagAOX.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

#include <bit>
#include <cstdint>
#include <cstring>

namespace libXWCTest
{
namespace loggerTest
{
using namespace MagAOX::logger;

/// \cond RAID_TELEMETRY_TEST_HARNESS
/// Wrap a RAID payload in its real logger envelope for verification dispatch.
flatlogs::bufferPtrT raidEnvelope( const telem_raid::messageT &message /**< [in] serialized RAID result */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_raid>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Append a serialized RAID result to an in-memory device log for FITS metadata lookup.
void appendRAID( logMap<>                   &map /**< [in,out] device log map */,
                 const telem_raid::messageT &message /**< [in] serialized RAID result */,
                 unsigned                    seconds /**< [in] log timestamp */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_raid>(
                 buffer, { seconds, 0 }, message, flatlogs::logPrio::LOG_TELEM ) == 0 );
    map.m_appToFileMap["sysMon"].emplace( "sysMon_20260101000000000000000.bintel" );
    auto      &memory = map.m_appToBufferMap["sysMon"];
    const auto size   = flatlogs::logHeader::totalSize( buffer );
    memory.m_memory.insert( memory.m_memory.end(), buffer.get(), buffer.get() + size );
    memory.m_startTime = flatlogs::timespecX( 0, 0 );
    memory.m_endTime   = flatlogs::timespecX( 100, 0 );
}
/// \endcond

/// Round-trip supported health states and collection time without repeated array identity strings.
/** \ingroup raidTelemetry_unit_test */
TEST_CASE( "RAID telemetry stores indexed numeric health", "[raidTelemetry]" )
{
    using raid = telem_raid;
    const std::vector<raid::arrayT> arrays{ { 0, raid::Valid | raid::OK | raid::Redundant, 2, 2 },
                                            { 2, raid::Valid | raid::Redundant | raid::Degraded | raid::Faulty, 4, 3 },
                                            { 3, raid::Valid | raid::Redundant | raid::Faulty, 2, 2 },
                                            { 4, raid::Valid | raid::Inactive, 0, 0 },
                                            { 5, raid::Valid | raid::OK, 0, 0 },
                                            { 6, raid::Valid | raid::OK | raid::Container, 0, 0 },
                                            { 7, raid::Missing, 0, 0 },
                                            { 8, 0, 0, 0 },
                                            { 9, raid::Valid | raid::Container | raid::Faulty, 0, 0 } };
    raid::messageT                  message( raid::Ready, arrays, 1234.5 );
    auto                           *payload = message.builder.GetBufferPointer();
    REQUIRE( sizeof( Raid_array_fb ) == 16 );
    REQUIRE( raid::status( payload ) == raid::Ready );
    REQUIRE( raid::checkTime( payload ) == 1234.5 );
    REQUIRE( raid::arrays( payload ) == arrays );
    REQUIRE( raid::valid( payload ) == std::vector<unsigned>{ 1, 1, 1, 1, 1, 1, 0, 0, 1 } );
    REQUIRE( raid::ok( payload ) == std::vector<unsigned>{ 1, 0, 0, 0, 1, 1, 0, 0, 0 } );
    REQUIRE( raid::degraded( payload ) == std::vector<unsigned>{ 0, 1, 0, 0, 0, 0, 0, 0, 0 } );
    auto buffer = raidEnvelope( message );
    REQUIRE( raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( raid::eventCode, buffer.get() ) );
    const auto text = raid::msgString( payload, message.builder.GetSize() );
    REQUIRE( text.find( "[raid] status=1 checked=1234.5" ) == 0 );
    REQUIRE( text.find( "index=2 flags=165 active=3/4" ) != std::string::npos );
    REQUIRE( text.find( "index=7 flags=8 active=0/0" ) != std::string::npos );

    auto json = raid::msgJSON(
        payload, message.builder.GetSize(), reinterpret_cast<const uint8_t *>( telem_raid_bfbs ), telem_raid_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_raid_bfbs ), telem_raid_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    REQUIRE( raid::arrays( parser.builder_.GetBufferPointer() ) == arrays );
    REQUIRE( raid::checkTime( parser.builder_.GetBufferPointer() ) == 1234.5 );
}

/// Collection outcomes remain distinguishable even when no arrays have been discovered.
/** \ingroup raidTelemetry_unit_test */
TEST_CASE( "RAID telemetry preserves empty inventories and collection outcomes", "[raidTelemetry]" )
{
    for( const auto status : { telem_raid::Unchecked,
                               telem_raid::Ready,
                               telem_raid::ReadError,
                               telem_raid::ParseError,
                               telem_raid::Disabled,
                               telem_raid::NotPresent } )
    {
        telem_raid::messageT message( status, {}, status == telem_raid::Unchecked ? 0 : 12 );
        auto                *payload = message.builder.GetBufferPointer();
        REQUIRE( GetTelem_raid_fb( payload )->arrays() != nullptr );
        REQUIRE( telem_raid::arrays( payload ).empty() );
        REQUIRE( telem_raid::status( payload ) == status );
        REQUIRE( telem_raid::valid( payload ).empty() );
        REQUIRE( telem_raid::ok( payload ).empty() );
        REQUIRE( telem_raid::degraded( payload ).empty() );
        auto buffer = raidEnvelope( message );
        REQUIRE( telem_raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    }
}

/// Reject identities, flags, and member counts that cannot represent the documented health states.
/** \ingroup raidTelemetry_unit_test */
TEST_CASE( "RAID telemetry rejects inconsistent array health", "[raidTelemetry]" )
{
    using raid = telem_raid;
    const std::vector<raid::arrayT> invalid{ { raid::UnknownIndex, raid::Valid | raid::OK, 0, 0 },
                                             { 0, 256, 0, 0 },
                                             { 0, raid::OK, 0, 0 },
                                             { 0, raid::Faulty, 0, 0 },
                                             { 0, raid::Redundant, 0, 0 },
                                             { 0, 0, 1, 0 },
                                             { 0, raid::Missing, 2, 0 },
                                             { 0, raid::Missing | raid::Valid, 0, 0 },
                                             { 0, raid::Valid | raid::OK | raid::Inactive, 0, 0 },
                                             { 0, raid::Valid | raid::OK | raid::Faulty, 0, 0 },
                                             { 0, raid::Valid | raid::Container, 0, 0 },
                                             { 0, raid::Valid | raid::OK | raid::Container | raid::Redundant, 2, 2 },
                                             { 0, raid::Valid | raid::OK | raid::Redundant | raid::Degraded, 2, 1 },
                                             { 0, raid::Valid | raid::Redundant, 0, 0 },
                                             { 0, raid::Valid | raid::Redundant, 2, 3 },
                                             { 0, raid::Valid | raid::Redundant, 2, 1 },
                                             { 0, raid::Valid | raid::Redundant | raid::Degraded, 2, 2 },
                                             { 0, raid::Valid | raid::Degraded, 0, 0 },
                                             { 0, raid::Valid | raid::OK, 2, 2 } };

    for( const auto &array : invalid )
    {
        INFO( array.m_flags );
        INFO( array.m_expected );
        INFO( array.m_active );
        raid::messageT message( raid::Ready, { array }, 10 );
        auto           buffer = raidEnvelope( message );
        REQUIRE_FALSE( raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    }

    raid::messageT duplicate( raid::Ready, { { 0, raid::Valid | raid::OK, 0, 0 }, { 0, raid::Missing, 0, 0 } }, 10 );
    auto           buffer = raidEnvelope( duplicate );
    REQUIRE_FALSE( raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
}

/// Reject malformed payloads, unknown collection codes, and invalid actual-check timestamps.
/** \ingroup raidTelemetry_unit_test */
TEST_CASE( "RAID telemetry rejects invalid collection envelopes", "[raidTelemetry]" )
{
    for( const auto value : { -1.0,
                              std::bit_cast<double>( uint64_t{ 0x7ff0000000000000 } ),
                              std::bit_cast<double>( uint64_t{ 0xfff0000000000000 } ),
                              std::bit_cast<double>( uint64_t{ 0x7ff8000000000000 } ) } )
    {
        telem_raid::messageT message( telem_raid::Ready, {}, 0 );
        auto                &builder = message.builder;
        builder.Clear();
        // Preserve invalid floating-point bits instead of comparing them with defaults under fast math.
        builder.ForceDefaults( true );
        const auto arrays = builder.CreateVectorOfStructs( std::vector<Raid_array_fb>{} );
        builder.Finish( CreateTelem_raid_fb( builder, telem_raid::Ready, arrays, value ) );
        REQUIRE( std::bit_cast<uint64_t>( telem_raid::checkTime( builder.GetBufferPointer() ) ) ==
                 std::bit_cast<uint64_t>( value ) );
        auto buffer = raidEnvelope( message );
        REQUIRE_FALSE( telem_raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    }

    telem_raid::messageT invalidStatus( 6, {}, 10 );
    auto                 buffer = raidEnvelope( invalidStatus );
    REQUIRE_FALSE( telem_raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );

    telem_raid::messageT valid( telem_raid::Ready, {}, 10 );
    buffer = raidEnvelope( valid );
    REQUIRE_FALSE( telem_raid::verify( buffer, 4 ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( telem_raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );

    auto &builder = valid.builder;
    builder.Clear();
    const auto start = builder.StartTable();
    builder.Finish( flatbuffers::Offset<Telem_raid_fb>( builder.EndTable( start ) ) );
    buffer = raidEnvelope( valid );
    REQUIRE_FALSE( telem_raid::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
}

/// Numeric accessors expose aligned indices, policy bits, counts, and actual check time through standard dispatch.
/** \ingroup raidTelemetry_unit_test */
TEST_CASE( "RAID telemetry metadata dispatch preserves aligned numeric health", "[raidTelemetry]" )
{
    telem_raid::messageT                                             message( telem_raid::Ready,
                                                                              { { 0, telem_raid::Valid | telem_raid::OK | telem_raid::Redundant, 2, 2 },
                                                                                { 5, telem_raid::Valid | telem_raid::Degraded | telem_raid::Redundant, 3, 2 },
                                                                                { 8, telem_raid::Missing, 0, 0 } },
                                  1234.5 );
    auto                                                            *payload = message.builder.GetBufferPointer();
    const std::vector<std::pair<std::string, std::vector<unsigned>>> fields{ { "index", { 0, 5, 8 } },
                                                                             { "flags", { 35, 37, 8 } },
                                                                             { "expected", { 2, 3, 0 } },
                                                                             { "active", { 2, 2, 0 } },
                                                                             { "valid", { 1, 1, 0 } },
                                                                             { "ok", { 1, 0, 0 } },
                                                                             { "degraded", { 0, 1, 0 } } };

    for( const auto &[name, expected] : fields )
    {
        const auto metadata = logMemberAccessor( telem_raid::eventCode, name );
        REQUIRE( metadata.valType == logMeta::Vector_UInt );
        REQUIRE( metadata.metaType == logMeta::State );
        REQUIRE( metadata.accessor != nullptr );
        auto getter = reinterpret_cast<std::vector<unsigned> ( * )( void * )>( metadata.accessor );
        REQUIRE( getter( payload ) == expected );
    }

    const auto state = logMemberAccessor( telem_raid::eventCode, "status" );
    REQUIRE( state.valType == logMeta::UInt );
    REQUIRE( state.accessor == reinterpret_cast<void *>( &telem_raid::status ) );
    const auto time = logMemberAccessor( telem_raid::eventCode, "checkTime" );
    REQUIRE( time.valType == logMeta::Double );
    REQUIRE( time.accessor == reinterpret_cast<void *>( &telem_raid::checkTime ) );
    REQUIRE( telem_raid::getAccessor( "unknown" ).accessor == nullptr );

    logMap<> map;
    appendRAID( map, message, 1 );
    appendRAID( map, message, 20 );

    for( const auto &[member, expected] :
         std::vector<std::pair<std::string, std::string>>{ { "index", "0,5,8" },
                                                           { "ok", "1,0,0" },
                                                           { "degraded", "0,1,0" },
                                                           { "active", "2,2,0" },
                                                           { "status", "1" },
                                                           { "checkTime", "1234.5" } } )
    {
        logMeta metadata( { "sysMon", telem_raid::eventCode, member } );
        INFO( member );
        REQUIRE( metadata.card( map, { 5, 0 }, { 10, 0 } ).valueStr() == expected );
    }
}

} // namespace loggerTest
} // namespace libXWCTest
