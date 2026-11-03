/** \file logsurgeon_test.cpp
 * \brief Catch2 tests for checking and repairing v1 and v2 log files with logsurgeon.
 *
 * \ingroup files
 */

#include "../../../tests/testXWC.hpp"

#include "../logsurgeon.hpp"

#include "../../../libMagAOX/logger/tests/logTestFiles.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace libXWCTest
{

/** \defgroup logsurgeon_unit_test logsurgeon Unit Tests
 * \brief Unit tests for the logsurgeon utility.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `logsurgeon` unit tests.
/** \ingroup logsurgeon_unit_test
 */
namespace logsurgeonTest
{

using libXWCTest::loggerTest::cerrCapture;
using libXWCTest::loggerTest::fileBytes;
using libXWCTest::loggerTest::tempDir;
using libXWCTest::loggerTest::testEntry;
using libXWCTest::loggerTest::unknownCode;
using libXWCTest::loggerTest::writeLogFile;

/** \cond logsurgeon_test_helpers
 */

/// Run logsurgeon with command-line arguments.
int runLogsurgeon( const std::vector<std::string> &args /**< [in] the arguments, without the program name */ )
{
    std::vector<std::string> sargv( { "logsurgeon" } );
    sargv.insert( sargv.end(), args.begin(), args.end() );

    std::vector<char *> argv( sargv.size() + 1, nullptr );
    for( size_t n = 0; n < sargv.size(); ++n )
    {
        argv[n] = sargv[n].data();
    }

    logsurgeon ls;
    return ls.main( sargv.size(), argv.data() );
}

/// Read a whole file.
std::vector<char> readFile( const std::string &path /**< [in] the file to read */ )
{
    std::ifstream in( path, std::ios::binary );
    return std::vector<char>( ( std::istreambuf_iterator<char>( in ) ), std::istreambuf_iterator<char>() );
}

/// Six text_log entries, one second apart.
std::vector<testEntry> sixEntries( time_t t0 /**< [in] time of the first entry */ )
{
    return { { t0, 0, "one" },
             { t0 + 1, 0, "two" },
             { t0 + 2, 0, "three" },
             { t0 + 3, 0, "four" },
             { t0 + 4, 0, "five" },
             { t0 + 5, 0, "six" } };
}

/** \endcond
 */

/// v2 files are checked, mapped and repaired by their checksums.
/**
 * \ingroup logsurgeon_unit_test
 */
TEST_CASE( "logsurgeon checks and repairs v2 files", "[logsurgeon]" )
{
    // clang-format off
    #ifdef LOGSURGEON_TEST_DOXYGEN_REF
    logsurgeon::execute();
    logsurgeon::mapChain();
    verifyCandidateEntry();
    readCandidateHeader();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir             td( "logsurgeon_v2" );
    std::vector<size_t> offs;

    SECTION( "a clean v2 file passes, including entries of a type this build doesn't know" )
    {
        auto f =
            writeLogFile( td.m_path,
                          "dev1",
                          "binlog2",
                          t0,
                          0,
                          fileBytes( offs,
                                     flatlogs::logVersion2,
                                     { { t0, 0, "one" }, { t0 + 1, 0, "u", unknownCode }, { t0 + 2, 0, "two" } } ) );

        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "-C", "-F", f.fullName() } ) == logsurgeon::noerror );
        REQUIRE( cap.count( "Found 0 bad bytes" ) == 1 );
        REQUIRE( runLogsurgeon( { "-C", "-M", "-F", f.fullName() } ) == logsurgeon::noerror );
    }

    SECTION( "a checksum failure is found, mapped and repaired" )
    {
        std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion2, sixEntries( t0 ) );
        bytes[offs[2] + flatlogs::logHeaderV2::headerSize( bytes.data() + offs[2] ) + 4] ^= 0x01;
        auto f = writeLogFile( td.m_path, "dev1", "binlog2", t0, 0, bytes );

        {
            cerrCapture cap;
            REQUIRE( runLogsurgeon( { "-C", "-F", f.fullName() } ) == logsurgeon::errors_found );
            REQUIRE( cap.count( "Corrupt: " + std::to_string( offs[2] ) + " - " + std::to_string( offs[3] ) ) == 1 );
            REQUIRE( cap.count( "reason=checksum does not match" ) == 1 );
            REQUIRE( readFile( f.fullName() ) == bytes );
        }

        {
            cerrCapture cap;
            REQUIRE( runLogsurgeon( { "-C", "-M", "-F", f.fullName() } ) == logsurgeon::errors_found );
            REQUIRE( cap.count( "Rejected length-chain run" ) == 1 );
        }

        {
            cerrCapture cap;
            REQUIRE( runLogsurgeon( { "-F", f.fullName() } ) == logsurgeon::noerror );
            REQUIRE( cap.count( "Surgery Complete" ) == 1 );
        }

        // The repaired file holds the other five entries byte for byte, checksums intact.
        std::vector<char> expect( bytes.begin(), bytes.begin() + offs[2] );
        expect.insert( expect.end(), bytes.begin() + offs[3], bytes.end() );
        REQUIRE( readFile( f.fullName() ) == expect );
        REQUIRE( readFile( f.fullName() + ".corrupted" ) == bytes );

        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "-C", "-F", f.fullName() } ) == logsurgeon::noerror );
    }

    SECTION( "an entry whose checksum matches but whose message fails its check is kept and reported" )
    {
        std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion2, sixEntries( t0 ) );

        // Break the flatbuffer root offset of entry three, then recompute its checksum.
        size_t   msgAt = offs[2] + flatlogs::logHeaderV2::headerSize( bytes.data() + offs[2] );
        uint32_t root  = 0xFFFFFFF0;
        memcpy( bytes.data() + msgAt, &root, sizeof( root ) );
        flatlogs::logHeaderV2::setCRC( bytes.data() + offs[2] );

        auto f = writeLogFile( td.m_path, "dev1", "binlog2", t0, 0, bytes );

        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "-C", "-F", f.fullName() } ) == logsurgeon::noerror );
        REQUIRE( cap.count( "Entry at byte " + std::to_string( offs[2] ) +
                            " (code=10) passes its checksum but fails the message check; kept." ) == 1 );
        REQUIRE( cap.count( "Found 0 bad bytes" ) == 1 );
    }
}

/// v1 repair refuses to remove entries of log types this build doesn't know, unless told to.
/**
 * \ingroup logsurgeon_unit_test
 */
TEST_CASE( "logsurgeon v1 repair and unknown log types", "[logsurgeon]" )
{
    // clang-format off
    #ifdef LOGSURGEON_TEST_DOXYGEN_REF
    logsurgeon::execute();
    unknownTypeChain();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir             td( "logsurgeon_unknown" );
    std::vector<size_t> offs;

    std::vector<char> bytes = fileBytes( offs,
                                         flatlogs::logVersion1,
                                         { { t0, 0, "one" },
                                           { t0 + 1, 0, "u1", unknownCode },
                                           { t0 + 2, 0, "u2", unknownCode },
                                           { t0 + 3, 0, "two" } } );
    auto              f     = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

    {
        // Check mode reports the stretch as before.
        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "-C", "-F", f.fullName() } ) == logsurgeon::errors_found );
        REQUIRE( cap.count( "Repair refused" ) == 0 );
    }

    {
        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "-F", f.fullName() } ) == logsurgeon::unknown_types );
        REQUIRE( cap.count( "Repair refused" ) == 1 );
        REQUIRE( cap.count( "(codes: 65000)" ) == 1 );
        REQUIRE( readFile( f.fullName() ) == bytes );
        REQUIRE( !std::filesystem::exists( f.fullName() + ".corrupted" ) );
    }

    {
        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "--drop-unknown-types", "-F", f.fullName() } ) == logsurgeon::noerror );
        REQUIRE( cap.count( "Surgery Complete" ) == 1 );
        REQUIRE( readFile( f.fullName() + ".corrupted" ) == bytes );
        REQUIRE( readFile( f.fullName() ).size() == offs[1] + ( bytes.size() - offs[3] ) );
    }
}

/// Entries of unknown types after real damage are also protected from repair.
/**
 * \ingroup logsurgeon_unit_test
 */
TEST_CASE( "logsurgeon v1 repair and unknown log types after damage", "[logsurgeon]" )
{
    // clang-format off
    #ifdef LOGSURGEON_TEST_DOXYGEN_REF
    unknownTypeChain();
    unknownTypeChainAt();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir             td( "logsurgeon_unknown_after_damage" );
    std::vector<size_t> offs;

    // Known entries, a few garbage bytes, four entries of an unknown type, then a known entry. The scan only
    // restarts at known types, so the garbage and the unknown entries make one stretch that starts with garbage.
    std::vector<char> head = fileBytes( offs, flatlogs::logVersion1, { { t0, 0, "one" }, { t0 + 1, 0, "two" } } );
    std::vector<char> garbage( 5, static_cast<char>( 0x7E ) );
    std::vector<char> unknown = fileBytes( offs,
                                           flatlogs::logVersion1,
                                           { { t0 + 2, 0, "u1", unknownCode },
                                             { t0 + 3, 0, "u2", unknownCode },
                                             { t0 + 4, 0, "u3", unknownCode },
                                             { t0 + 5, 0, "u4", unknownCode } } );
    std::vector<char> tail    = fileBytes( offs, flatlogs::logVersion1, { { t0 + 6, 0, "three" } } );

    std::vector<char> bytes = head;
    bytes.insert( bytes.end(), garbage.begin(), garbage.end() );
    bytes.insert( bytes.end(), unknown.begin(), unknown.end() );
    bytes.insert( bytes.end(), tail.begin(), tail.end() );

    auto f = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

    {
        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "-F", f.fullName() } ) == logsurgeon::unknown_types );
        REQUIRE( cap.count( "(codes: 65000)" ) == 1 );
        REQUIRE( readFile( f.fullName() ) == bytes );
        REQUIRE( !std::filesystem::exists( f.fullName() + ".corrupted" ) );
    }

    {
        cerrCapture cap;
        REQUIRE( runLogsurgeon( { "--drop-unknown-types", "-F", f.fullName() } ) == logsurgeon::noerror );
        REQUIRE( readFile( f.fullName() + ".corrupted" ) == bytes );
        REQUIRE( readFile( f.fullName() ).size() == head.size() + tail.size() );
    }
}

/// A v1 candidate whose length is huge is reported, not allocated.
/**
 * \ingroup logsurgeon_unit_test
 */
TEST_CASE( "logsurgeon and a huge v1 length", "[logsurgeon]" )
{
    const time_t t0 = 1732000000;

    tempDir             td( "logsurgeon_huge" );
    std::vector<size_t> offs;

    std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion1, sixEntries( t0 ) );

    // Between entries two and three, a text_log header claiming a 64-bit length near the maximum: the end position
    // wraps negative, which used to pass the "fits in the file" check and reach the allocation.
    std::vector<char> junk( 1 + 2 + 8 + 1 + 8, 0 );
    junk[0]           = flatlogs::logPrio::LOG_INFO;
    uint16_t textCode = MagAOX::logger::eventCodes::TEXT_LOG;
    memcpy( junk.data() + 1, &textCode, sizeof( textCode ) );
    junk[11]         = static_cast<char>( 0xFF );
    uint64_t hugeLen = 0xa0264b00ffffff80ULL;
    memcpy( junk.data() + 12, &hugeLen, sizeof( hugeLen ) );
    bytes.insert( bytes.begin() + offs[2], junk.begin(), junk.end() );

    auto f = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

    cerrCapture cap;
    REQUIRE( runLogsurgeon( { "-C", "-F", f.fullName() } ) == logsurgeon::errors_found );
    REQUIRE( cap.count( "Found " + std::to_string( junk.size() ) + " bad bytes" ) == 1 );
    REQUIRE( runLogsurgeon( { "-C", "-M", "-F", f.fullName() } ) == logsurgeon::errors_found );
}

} // namespace logsurgeonTest
} // namespace libXWCTest
