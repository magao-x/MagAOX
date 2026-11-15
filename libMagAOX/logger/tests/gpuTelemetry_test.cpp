/** \file gpuTelemetry_test.cpp
 * \brief GPU telemetry indices, omitted identity, legacy decoding, validity, and metadata tests.
 * \ingroup flatlogs_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../libMagAOX.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{

/// \cond GPU_TELEMETRY_TEST_HARNESS
/// Append a valid serialized GPU sample to an in-memory device log for actual FITS metadata lookups.
void appendGPU( MagAOX::logger::logMap<>                  &map /**< [in,out] device log map */,
                const MagAOX::logger::telem_gpu::messageT &message /**< [in] serialized sample */,
                unsigned                                   seconds /**< [in] log timestamp */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<MagAOX::logger::telem_gpu>(
                 buffer, { seconds, 0 }, message, flatlogs::logPrio::LOG_TELEM ) == 0 );
    map.m_appToFileMap["sysMon"].emplace( "sysMon_20260101000000000000000.bintel" );
    auto      &memory = map.m_appToBufferMap["sysMon"];
    const auto size   = flatlogs::logHeader::totalSize( buffer );
    memory.m_memory.insert( memory.m_memory.end(), buffer.get(), buffer.get() + size );
    memory.m_startTime = flatlogs::timespecX( 0, 0 );
    memory.m_endTime   = flatlogs::timespecX( 100, 0 );
}
/// \endcond

/// Round-trip indexed measurements and retained invalid values while omitting static identity strings.
/** \ingroup flatlogs_unit_test */
TEST_CASE( "GPU telemetry retains indexed measurements and omits static identity", "[gpuTelemetry]" )
{
    using MagAOX::logger::telem_gpu;
    std::vector<telem_gpu::gpuT> gpus{ { "GPU-01234567-89ab-cdef-0123-456789abcdef",
                                         "0000:01:00.0",
                                         "ok",
                                         65,
                                         0.75,
                                         0.25,
                                         256,
                                         8192,
                                         1400,
                                         1410,
                                         7000,
                                         telem_gpu::AllValid,
                                         1234567890.25,
                                         0 },
                                       { "GPU-11234567-89ab-cdef-0123-456789abcdef",
                                         "0000:02:00.0",
                                         "missing",
                                         40,
                                         0.1,
                                         0.5,
                                         512,
                                         16384,
                                         500,
                                         510,
                                         2000,
                                         0,
                                         1234567880.5,
                                         7 } };
    telem_gpu::messageT          message( "partial", gpus );
    auto                        *payload = message.builder.GetBufferPointer();
    REQUIRE( telem_gpu::status( payload ) == "partial" );
    const auto decoded  = telem_gpu::gpus( payload );
    auto       expected = gpus;

    for( auto &gpu : expected )
    {
        gpu.m_uuid.clear();
        gpu.m_pci.clear();
    }

    REQUIRE( decoded == expected );
    REQUIRE( decoded[0].m_sampleTime == gpus[0].m_sampleTime );
    REQUIRE( decoded[1].m_sampleTime == gpus[1].m_sampleTime );
    REQUIRE( decoded[0].m_index == 0 );
    REQUIRE( decoded[1].m_index == 7 );
    const auto        text = telem_gpu::msgString( payload, message.builder.GetSize() );
    const std::string bytes( reinterpret_cast<const char *>( payload ), message.builder.GetSize() );

    for( size_t n = 0; n < gpus.size(); ++n )
    {
        const auto *sample = MagAOX::logger::GetTelem_gpu_fb( payload )->gpus()->Get( n );
        REQUIRE( sample->uuid() == nullptr );
        REQUIRE( sample->pci() == nullptr );
        REQUIRE( bytes.find( gpus[n].m_uuid ) == std::string::npos );
        REQUIRE( bytes.find( gpus[n].m_pci ) == std::string::npos );
        REQUIRE( text.find( gpus[n].m_uuid ) == std::string::npos );
        REQUIRE( text.find( gpus[n].m_pci ) == std::string::npos );
    }

    REQUIRE( text.find( "clocks=1400/1410/7000 MHz" ) != std::string::npos );
    REQUIRE( text.find( "memory=256/8192 MiB" ) != std::string::npos );
    REQUIRE( text.find( "valid=0" ) != std::string::npos );
    REQUIRE( text.find( "index=0" ) != std::string::npos );
    REQUIRE( text.find( "index=7" ) != std::string::npos );
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_gpu>( buffer, { 1, 0 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    REQUIRE( telem_gpu::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( MagAOX::logger::verifyLogEntry( telem_gpu::eventCode, buffer.get() ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( telem_gpu::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );

    gpus[0].m_valid = 256;
    telem_gpu::messageT badMask( "ok", gpus );
    REQUIRE( flatlogs::logHeader::createLog<telem_gpu>( buffer, { 1, 0 }, badMask, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    REQUIRE_FALSE( telem_gpu::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
}

/// Legacy identity strings remain readable, and omitted indices must never be mistaken for device zero.
/** \ingroup flatlogs_unit_test */
TEST_CASE( "GPU telemetry preserves legacy identities and reads absent indices as unknown", "[gpuTelemetry]" )
{
    using namespace MagAOX::logger;
    REQUIRE( telem_gpu::gpuT{}.m_index == telem_gpu::UnknownIndex );
    telem_gpu::messageT message( "legacy", {} );
    auto               &builder = message.builder;
    builder.Clear();
    const auto           uuid   = builder.CreateString( "GPU-01234567-89ab-cdef-0123-456789abcdef" );
    const auto           pci    = builder.CreateString( "0000:01:00.0" );
    const auto           status = builder.CreateString( "ok" );
    Gpu_sample_fbBuilder legacy( builder );
    legacy.add_uuid( uuid );
    legacy.add_pci( pci );
    legacy.add_status( status );
    legacy.add_sample_time( 1234567890.25 );
    const auto sample  = legacy.Finish();
    const auto samples = builder.CreateVector( std::vector<flatbuffers::Offset<Gpu_sample_fb>>{ sample } );
    builder.Finish( CreateTelem_gpu_fb( builder, status, samples ) );
    auto      *payload = builder.GetBufferPointer();
    const auto decoded = telem_gpu::gpus( payload );
    REQUIRE( decoded.size() == 1 );
    REQUIRE( decoded[0].m_uuid == "GPU-01234567-89ab-cdef-0123-456789abcdef" );
    REQUIRE( decoded[0].m_pci == "0000:01:00.0" );
    REQUIRE( decoded[0].m_index == telem_gpu::UnknownIndex );
    REQUIRE( decoded[0].m_sampleTime == 1234567890.25 );
    const auto text = telem_gpu::msgString( payload, builder.GetSize() );
    REQUIRE( text.find( "index=unknown" ) != std::string::npos );
    REQUIRE( text.find( "uuid=" + decoded[0].m_uuid ) != std::string::npos );
    REQUIRE( text.find( "pci=" + decoded[0].m_pci ) != std::string::npos );

    for( const auto &[member, expected] : std::vector<std::pair<std::string, std::string>>{
             { "uuid", decoded[0].m_uuid }, { "pci", decoded[0].m_pci } } )
    {
        auto metadata = telem_gpu::getAccessor( member );
        auto getter   = reinterpret_cast<std::vector<std::string> ( * )( void * )>( metadata.accessor );
        REQUIRE( getter( payload ) == std::vector<std::string>{ expected } );
    }

    auto metadata = telem_gpu::getAccessor( "index" );
    auto indices  = reinterpret_cast<std::vector<unsigned> ( * )( void * )>( metadata.accessor );
    REQUIRE( indices( payload ) == std::vector<unsigned>{ telem_gpu::UnknownIndex } );
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_gpu>( buffer, { 1, 0 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    REQUIRE( telem_gpu::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
}

/// Empty inventories still communicate explicit overall query or disabled states.
/** \ingroup flatlogs_unit_test */
TEST_CASE( "GPU telemetry records empty and unavailable inventories", "[gpuTelemetry]" )
{
    using MagAOX::logger::telem_gpu;

    for( const auto *state : { "disabled", "no_gpus", "query_failed", "timeout" } )
    {
        telem_gpu::messageT message( state, {} );
        auto               *payload = message.builder.GetBufferPointer();
        REQUIRE( telem_gpu::gpus( payload ).empty() );
        REQUIRE( telem_gpu::status( payload ) == state );
        flatlogs::bufferPtrT buffer;
        REQUIRE( flatlogs::logHeader::createLog<telem_gpu>( buffer, { 1, 0 }, message, flatlogs::logPrio::LOG_TELEM ) ==
                 0 );
        REQUIRE( telem_gpu::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    }
}

/// Metadata vectors expose matching indices and validity, with empty placeholders for omitted identity fields.
/** \ingroup flatlogs_unit_test */
TEST_CASE( "GPU metadata dispatch exposes units and validity alongside every field", "[gpuTelemetry]" )
{
    using namespace MagAOX::logger;
    std::vector<telem_gpu::gpuT>                      gpus{ { "GPU-01234567-89ab-cdef-0123-456789abcdef",
                                                              "0000:01:00.0",
                                                              "partial",
                                                              65,
                                                              0.75,
                                                              0,
                                                              256,
                                                              8192,
                                                              1400,
                                                              1410,
                                                              7000,
                                                              telem_gpu::AllValid ^ telem_gpu::MemoryLoad,
                                                              10,
                                                              3 } };
    telem_gpu::messageT                               message( "partial", gpus );
    auto                                             *payload = message.builder.GetBufferPointer();
    const std::vector<std::pair<std::string, double>> fields{ { "temperature", 65 },
                                                              { "load", 0.75 },
                                                              { "memory_load", 0 },
                                                              { "memory_used", 256 },
                                                              { "memory_total", 8192 },
                                                              { "clock_graphics", 1400 },
                                                              { "clock_sm", 1410 },
                                                              { "clock_memory", 7000 },
                                                              { "sample_time", 10 } };

    for( const auto &[name, value] : fields )
    {
        auto metadata = logMemberAccessor( telem_gpu::eventCode, name );
        REQUIRE( metadata.accessor != nullptr );
        REQUIRE( metadata.valType == logMeta::Vector_Double );
        REQUIRE( metadata.metaType == logMeta::State );
        auto getter = reinterpret_cast<std::vector<double> ( * )( void * )>( metadata.accessor );
        REQUIRE( getter( payload ) == std::vector<double>{ value } );
    }

    auto validity = telem_gpu::getAccessor( "valid" );
    REQUIRE( validity.valType == logMeta::Vector_UInt );
    auto getValidity = reinterpret_cast<std::vector<unsigned> ( * )( void * )>( validity.accessor );
    REQUIRE( getValidity( payload ) == std::vector<unsigned>{ gpus[0].m_valid } );

    auto indices = logMemberAccessor( telem_gpu::eventCode, "index" );
    REQUIRE( indices.valType == logMeta::Vector_UInt );
    REQUIRE( indices.metaType == logMeta::State );
    auto getIndices = reinterpret_cast<std::vector<unsigned> ( * )( void * )>( indices.accessor );
    REQUIRE( getIndices( payload ) == std::vector<unsigned>{ 3 } );

    for( const auto &[name, value] : std::vector<std::pair<std::string, std::string>>{
             { "uuid", "" }, { "pci", "" }, { "device_status", "partial" } } )
    {
        auto metadata = telem_gpu::getAccessor( name );
        REQUIRE( metadata.valType == logMeta::Vector_String );
        auto getter = reinterpret_cast<std::vector<std::string> ( * )( void * )>( metadata.accessor );
        REQUIRE( getter( payload ) == std::vector<std::string>{ value } );
    }

    REQUIRE( telem_gpu::getAccessor( "status" ).valType == logMeta::String );
    REQUIRE( telem_gpu::getAccessor( "unknown" ).accessor == nullptr );
}

/// Exercise real FITS-card lookup for string, unsigned, and double vectors, including empty and absent data.
/** \ingroup flatlogs_unit_test */
TEST_CASE( "GPU vector metadata produces verified FITS cards", "[gpuTelemetry][logMeta]" )
{
    using namespace MagAOX::logger;
    const std::string            uuid  = "GPU-01234567-89ab-cdef-0123-456789abcdef";
    const std::string            uuidB = "GPU-11234567-89ab-cdef-0123-456789abcdef";
    std::vector<telem_gpu::gpuT> gpus{
        { uuid, "0000:01:00.0", "ok", 65, 0.75, 0.25, 256, 8192, 1400, 1410, 7000, 255, 1234567890.25, 0 },
        { uuidB, "0000:02:00.0", "missing", 40, 0.5, 0.25, 512, 8192, 500, 510, 2000, 0, 1234567880.5, 7 } };
    telem_gpu::messageT message( "partial", gpus );
    logMap<>            map;
    appendGPU( map, message, 1 );
    appendGPU( map, message, 20 );

    for( const auto &[member, expected] :
         std::vector<std::pair<std::string, std::string>>{ { "uuid", "," },
                                                           { "pci", "," },
                                                           { "device_status", "ok,missing" },
                                                           { "index", "0,7" },
                                                           { "valid", "255,0" },
                                                           { "load", "0.75,0.5" },
                                                           { "clock_sm", "1410,510" },
                                                           { "sample_time", "1234567890.25,1234567880.5" } } )
    {
        logMeta metadata( { "sysMon", telem_gpu::eventCode, member } );
        auto    card = metadata.card( map, { 5, 0 }, { 10, 0 } );
        INFO( member );
        INFO( metadata.unavailableReason() );
        REQUIRE( card.valueStr() == expected );
    }

    logMeta formatted( { "sysMon", telem_gpu::eventCode, "load", "", "%.2f", "" } );
    REQUIRE( formatted.card( map, { 5, 0 }, { 10, 0 } ).valueStr() == "0.75,0.50" );

    logMap<> absent;
    logMeta  validity( { "sysMon", telem_gpu::eventCode, "valid" } );
    REQUIRE( validity.card( absent, { 5, 0 }, { 10, 0 } ).valueStr() == logMeta::unavailableValue() );
    logMap<>            empty;
    telem_gpu::messageT noDevices( "no_gpus", {} );
    appendGPU( empty, noDevices, 1 );
    appendGPU( empty, noDevices, 20 );

    for( const auto *member : { "uuid", "index", "valid", "load" } )
    {
        logMeta metadata( { "sysMon", telem_gpu::eventCode, member } );
        REQUIRE( metadata.card( empty, { 5, 0 }, { 10, 0 } ).valueStr().empty() );
    }
}

} // namespace loggerTest
} // namespace libXWCTest
