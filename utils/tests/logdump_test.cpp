/** \file logdump_test.cpp
 * \brief Catch2 tests for the logdump utility.
 *
 * \ingroup logdump_files
 */

#include "../../tests/testXWC.hpp"

#include "../logdump/logdump.hpp"

#include "../../libMagAOX/logger/tests/logTestFiles.hpp"

#include <chrono>
#include <cstdio>
#include <future>
#include <sstream>
#include <string>
#include <thread>

namespace libXWCTest
{

/** \defgroup logdump_unit_test logdump Unit Tests
 * \brief Unit tests for the logdump utility.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `logdump` unit tests.
/** \ingroup logdump_unit_test
 */
namespace logdumpTest
{

using libXWCTest::loggerTest::cerrCapture;
using libXWCTest::loggerTest::fileBytes;
using libXWCTest::loggerTest::tempDir;
using libXWCTest::loggerTest::testEntry;
using libXWCTest::loggerTest::unknownCode;
using libXWCTest::loggerTest::writeLogFile;

/// Test harness exposing logdump's settings and running it on files.
/** \cond logdump_test_harness
 */
class logdumpHarness : public logdump
{
  public:
    using logdump::m_dir;
    using logdump::m_ext;
    using logdump::m_file;
    using logdump::m_fileCheckInterval;
    using logdump::m_follow;
    using logdump::m_jsonMode;
    using logdump::m_keepGoing;
    using logdump::m_pauseTime;
    using logdump::m_prefixes;
    using logdump::m_time;

    /// Run execute(), capturing what it prints.
    int run( std::string &out, /**< [out] what was printed to std::cout */
             std::string &err  /**< [out] what was printed to std::cerr */
    )
    {
        std::ostringstream outs;
        std::ostringstream errs;
        std::streambuf    *oldOut = std::cout.rdbuf( outs.rdbuf() );
        std::streambuf    *oldErr = std::cerr.rdbuf( errs.rdbuf() );

        int rv = execute();

        std::cout.rdbuf( oldOut );
        std::cerr.rdbuf( oldErr );

        out = outs.str();
        err = errs.str();

        return rv;
    }
};

/// Count the occurrences of a string in another.
size_t countOf( const std::string &s,   /**< [in] the string to search */
                const std::string &what /**< [in] the string to count */
)
{
    size_t n = 0;
    for( size_t pos = s.find( what ); pos != std::string::npos; pos = s.find( what, pos + what.size() ) )
    {
        ++n;
    }
    return n;
}

/** \endcond
 */

/// The same entries written as v1 and as v2 are printed identically, and a folder of both is read in time order.
/**
 * \ingroup logdump_unit_test
 */
TEST_CASE( "logdump reads v1 and v2 files", "[logdump]" )
{
    // clang-format off
    #ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
    listLogFiles();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir                td( "logdump_versions" );
    std::vector<size_t>    offs;
    std::vector<testEntry> entries = { { t0, 0, "alpha" }, { t0 + 1, 0, "beta" }, { t0 + 2, 0, "gamma" } };

    auto v1 = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, fileBytes( offs, flatlogs::logVersion1, entries ) );
    auto v2 = writeLogFile(
        td.m_path,
        "dev1",
        "binlog2",
        t0 + 10,
        0,
        fileBytes( offs, flatlogs::logVersion2, { { t0 + 10, 0, "delta" }, { t0 + 11, 0, "epsilon" } } ) );

    SECTION( "a v2 file prints the same as the equivalent v1 file, in text and JSON" )
    {
        tempDir td2( "logdump_same" );
        auto    w2 =
            writeLogFile( td2.m_path, "dev1", "binlog2", t0, 0, fileBytes( offs, flatlogs::logVersion2, entries ) );

        for( bool json : { false, true } )
        {
            logdumpHarness a;
            a.m_file     = v1.fullName();
            a.m_jsonMode = json;
            std::string out1, err1;
            REQUIRE( a.run( out1, err1 ) == 0 );

            logdumpHarness b;
            b.m_file     = w2.fullName();
            b.m_jsonMode = json;
            std::string out2, err2;
            REQUIRE( b.run( out2, err2 ) == 0 );

            REQUIRE( out1 == out2 );
            REQUIRE( countOf( out1, "alpha" ) == 1 );
            REQUIRE( countOf( out1, "gamma" ) == 1 );
        }
    }

    SECTION( "a folder with both extensions is read in time order" )
    {
        logdumpHarness ld;
        ld.m_dir      = td.m_path;
        ld.m_prefixes = { "dev1" };
        ld.m_ext      = ".binlog";

        std::string out, err;
        REQUIRE( ld.run( out, err ) == 0 );

        size_t a = out.find( "alpha" );
        size_t g = out.find( "gamma" );
        size_t d = out.find( "delta" );
        size_t e = out.find( "epsilon" );
        REQUIRE( a != std::string::npos );
        REQUIRE( a < g );
        REQUIRE( g < d );
        REQUIRE( d < e );
    }

    SECTION( "the time-span mode reads each file by its own version" )
    {
        logdumpHarness ld;
        ld.m_dir      = td.m_path;
        ld.m_prefixes = { "dev1" };
        ld.m_ext      = ".binlog";
        ld.m_time     = true;

        std::string out, err;
        REQUIRE( ld.run( out, err ) == 0 );

        // One line per file: name, first time, last time, span, number of entries.
        REQUIRE( out.find( v1.fullName() + " " ) != std::string::npos );
        REQUIRE( out.find( v2.fullName() + " " ) != std::string::npos );
        REQUIRE( out.find( " 2 3\n" ) != std::string::npos );
        REQUIRE( out.find( " 1 2\n" ) != std::string::npos );
    }
}

/// Damaged entries stop logdump by default, and are reported and skipped with --keepGoing.
/**
 * \ingroup logdump_unit_test
 */
TEST_CASE( "logdump and damaged entries", "[logdump]" )
{
    // clang-format off
    #ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::checkEntry();
    logdump::findRestart();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir             td( "logdump_damage" );
    std::vector<size_t> offs;

    std::vector<testEntry> six = { { t0, 0, "one" },
                                   { t0 + 1, 0, "two" },
                                   { t0 + 2, 0, "three" },
                                   { t0 + 3, 0, "four" },
                                   { t0 + 4, 0, "five" },
                                   { t0 + 5, 0, "six" } };

    SECTION( "a v2 checksum failure stops with the reason, or is skipped with --keepGoing" )
    {
        std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion2, six );
        bytes[offs[2] + flatlogs::logHeaderV2::headerSize( bytes.data() + offs[2] ) + 4] ^= 0x01;
        auto f = writeLogFile( td.m_path, "dev1", "binlog2", t0, 0, bytes );

        logdumpHarness ld;
        ld.m_file = f.fullName();
        std::string out, err;
        REQUIRE( ld.run( out, err ) == -1 );
        REQUIRE( countOf( err, "at byte=" + std::to_string( offs[2] ) + " (reason: checksum)" ) == 1 );
        REQUIRE( out.find( "two" ) != std::string::npos );
        REQUIRE( out.find( "four" ) == std::string::npos );

        logdumpHarness kg;
        kg.m_file      = f.fullName();
        kg.m_keepGoing = true;
        REQUIRE( kg.run( out, err ) == -1 );
        REQUIRE( countOf( err,
                          "skipped damaged entry (code=10) at byte=" + std::to_string( offs[2] ) +
                              " (reason: checksum); resumed at byte=" + std::to_string( offs[3] ) ) == 1 );
        REQUIRE( out.find( "three" ) == std::string::npos );
        REQUIRE( out.find( "six" ) != std::string::npos );
    }

    SECTION( "a v1 file with a damaged length mid-file still fails, and --keepGoing recovers" )
    {
        std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion1, six );
        bytes[offs[1] + 11] += 7; // the one-byte message length of entry two
        auto f = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

        logdumpHarness ld;
        ld.m_file = f.fullName();
        std::string out, err;
        REQUIRE( ld.run( out, err ) == -1 );
        REQUIRE( countOf( err, "File possibly corrupt.  Exiting." ) == 1 );

        logdumpHarness kg;
        kg.m_file      = f.fullName();
        kg.m_keepGoing = true;
        REQUIRE( kg.run( out, err ) == -1 );
        REQUIRE( countOf( err, "skipped damaged entry" ) >= 1 );
        REQUIRE( out.find( "one" ) != std::string::npos );
        REQUIRE( out.find( "six" ) != std::string::npos );
    }

    SECTION( "a damaged length claiming more than the file holds is damage when good entries follow" )
    {
        std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion1, six );
        bytes[offs[1] + 11]     = static_cast<char>( 250 );
        auto f                  = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

        logdumpHarness ld;
        ld.m_file = f.fullName();
        std::string out, err;
        REQUIRE( ld.run( out, err ) == -1 );
        REQUIRE( countOf( err, "at byte=" + std::to_string( offs[1] ) + " (reason: structure)" ) == 1 );
    }

    SECTION( "a final entry cut short by the end of the file is a warning, not damage" )
    {
        std::vector<char> bytes = fileBytes( offs, flatlogs::logVersion2, six );
        bytes.resize( bytes.size() - 3 );
        auto f = writeLogFile( td.m_path, "dev1", "binlog2", t0, 0, bytes );

        logdumpHarness ld;
        ld.m_file = f.fullName();
        std::string out, err;
        REQUIRE( ld.run( out, err ) == 0 );
        REQUIRE( countOf( err, "incomplete final entry at byte=" + std::to_string( offs[5] ) ) == 1 );
        REQUIRE( out.find( "five" ) != std::string::npos );
        REQUIRE( out.find( "six" ) == std::string::npos );
    }

    SECTION( "--keepGoing can't be combined with --follow" )
    {
        logdumpHarness ld;
        ld.m_file      = "unused";
        ld.m_keepGoing = true;
        ld.m_follow    = true;
        std::string out, err;
        REQUIRE( ld.run( out, err ) == -1 );
        REQUIRE( countOf( err, "--keepGoing can't be used with --follow" ) == 1 );
    }
}

/// Entries of a type this build doesn't know are printed with the standard prefix in text, and left out of JSON.
/**
 * \ingroup logdump_unit_test
 */
TEST_CASE( "logdump and unknown log types", "[logdump]" )
{
    // clang-format off
    #ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::printLogBuff();
    logdump::printLogJson();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir             td( "logdump_unknown" );
    std::vector<size_t> offs;

    for( flatlogs::logVersionT version : { flatlogs::logVersion1, flatlogs::logVersion2 } )
    {
        auto f = writeLogFile(
            td.m_path,
            "dev1",
            version == flatlogs::logVersion1 ? "binlog" : "binlog2",
            t0 + version * 100,
            0,
            fileBytes(
                offs, version, { { t0, 0, "known" }, { t0 + 1, 0, "u", unknownCode }, { t0 + 2, 0, "after" } } ) );

        logdumpHarness txt;
        txt.m_file = f.fullName();
        std::string out, err;
        REQUIRE( txt.run( out, err ) == 0 );
        REQUIRE( countOf( out, " Unknown log type: 65000" ) == 1 );
        REQUIRE( out.find( "after" ) != std::string::npos );
        REQUIRE( countOf( err, "1 entries of log type 65000, which this build doesn't know, were printed" ) == 1 );

        logdumpHarness json;
        json.m_file     = f.fullName();
        json.m_jsonMode = true;
        REQUIRE( json.run( out, err ) == 0 );
        REQUIRE( countOf( out, "\n" ) == 2 );
        REQUIRE( out.find( "65000" ) == std::string::npos );
        REQUIRE( countOf( err, "were left out of the JSON output" ) == 1 );
    }
}

/// Reading one entry at a time: an entry written in two halves is waited for, not misread.
/**
 * \ingroup logdump_unit_test
 */
TEST_CASE( "logdump reads an entry written in two halves", "[logdump]" )
{
    // clang-format off
    #ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::readEntry();
    #endif
    // clang-format on

    tempDir td( "logdump_halves" );

    for( flatlogs::logVersionT version : { flatlogs::logVersion1, flatlogs::logVersion2 } )
    {
        std::vector<size_t> offs;
        std::vector<char>   bytes = fileBytes( offs, version, { { 1732000000, 0, "halves" } } );

        for( size_t half : { size_t( 5 ), size_t( 14 ), bytes.size() - 1 } )
        {
            std::string path = td.m_path + "/halves_" + std::to_string( version ) + "_" + std::to_string( half );

            FILE *w = fopen( path.c_str(), "wb" );
            fwrite( bytes.data(), 1, half, w );
            fflush( w );

            FILE             *r = fopen( path.c_str(), "rb" );
            std::vector<char> buff;
            size_t            entrySize = 0;
            REQUIRE( logdump::readEntry( entrySize, buff, r, version ) == logdump::readResult::incomplete );
            REQUIRE( ftello( r ) == 0 );

            fwrite( bytes.data() + half, 1, bytes.size() - half, w );
            fflush( w );
            clearerr( r );

            REQUIRE( logdump::readEntry( entrySize, buff, r, version ) == logdump::readResult::entry );
            REQUIRE( entrySize == bytes.size() );
            REQUIRE( flatlogs::parseEntry( buff.data(), entrySize, version ).msg != nullptr );
            REQUIRE( logdump::readEntry( entrySize, buff, r, version ) == logdump::readResult::end );

            fclose( r );
            fclose( w );
        }
    }
}

/// In follow mode an entry still being written is waited for, and a damaged length is caught once good entries follow.
/**
 * \ingroup logdump_unit_test
 */
TEST_CASE( "logdump follow mode and damaged lengths", "[logdump]" )
{
    // clang-format off
    #ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
    logdump::findRestart();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir td( "logdump_follow" );

    std::vector<size_t> offs;
    std::vector<char>   first = fileBytes( offs, flatlogs::logVersion1, { { t0, 0, "first" } } );
    auto                f     = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, first );

    // Runs in a detached thread, so a failure can't hang the test; the harness is deliberately not freed.
    logdumpHarness *ld      = new logdumpHarness;
    ld->m_dir               = td.m_path;
    ld->m_prefixes          = { "dev1" };
    ld->m_ext               = ".binlog";
    ld->m_follow            = true;
    ld->m_pauseTime         = 10;
    ld->m_fileCheckInterval = 2;

    std::promise<int> done;
    std::future<int>  result = done.get_future();
    std::string      *out    = new std::string;
    std::string      *err    = new std::string;
    std::thread( [ld, out, err, &done]() { done.set_value( ld->run( *out, *err ) ); } ).detach();

    auto append = []( const std::string &path, const std::vector<char> &bytes, size_t from, size_t to )
    {
        FILE *w = fopen( path.c_str(), "ab" );
        fwrite( bytes.data() + from, 1, to - from, w );
        fclose( w );
    };

    // An entry written in three parts, slowly: the file grows while the entry is incomplete, but no good entry
    // follows it, so it is waited for, not reported.
    std::vector<char> slow = fileBytes( offs, flatlogs::logVersion1, { { t0 + 1, 0, "slowly written entry" } } );
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    append( f.fullName(), slow, 0, 6 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    append( f.fullName(), slow, 6, 20 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    append( f.fullName(), slow, 20, slow.size() );
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

    REQUIRE( result.wait_for( std::chrono::milliseconds( 0 ) ) == std::future_status::timeout );

    // Then an entry whose length claims more than the file holds, followed by good entries.
    size_t            damagedAt = first.size() + slow.size();
    std::vector<char> more      = fileBytes(
        offs,
        flatlogs::logVersion1,
        { { t0 + 2, 0, "damaged" }, { t0 + 3, 0, "good1" }, { t0 + 4, 0, "good2" }, { t0 + 5, 0, "good3" } } );
    more[offs[0] + 11] = static_cast<char>( 250 );
    append( f.fullName(), more, 0, more.size() );

    REQUIRE( result.wait_for( std::chrono::seconds( 10 ) ) == std::future_status::ready );
    REQUIRE( result.get() == -1 );
    REQUIRE( out->find( "slowly written entry" ) != std::string::npos );
    REQUIRE( countOf( *err, "at byte=" + std::to_string( damagedAt ) + " (reason: structure)" ) == 1 );
}

/// In follow mode, a new file of the v2 extension in the same folder is picked up and read as v2: the cutover moment.
/**
 * \ingroup logdump_unit_test
 */
TEST_CASE( "logdump follow mode moves on to a newer v2 file", "[logdump]" )
{
    // clang-format off
    #ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
    listLogFiles();
    #endif
    // clang-format on

    const time_t t0 = 1732000000;

    tempDir td( "logdump_follow_cutover" );

    std::vector<size_t> offs;
    writeLogFile(
        td.m_path, "dev1", "bintel", t0, 0, fileBytes( offs, flatlogs::logVersion1, { { t0, 0, "last v1 entry" } } ) );

    // Runs in a detached thread, so a failure can't hang the test; the harness is deliberately not freed.
    logdumpHarness *ld      = new logdumpHarness;
    ld->m_dir               = td.m_path;
    ld->m_prefixes          = { "dev1" };
    ld->m_ext               = ".bintel";
    ld->m_follow            = true;
    ld->m_pauseTime         = 10;
    ld->m_fileCheckInterval = 2;

    std::promise<int> done;
    std::future<int>  result = done.get_future();
    std::string      *out    = new std::string;
    std::string      *err    = new std::string;
    std::thread( [ld, out, err, &done]() { done.set_value( ld->run( *out, *err ) ); } ).detach();

    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

    // The cutover: the writer starts a .bintel2 file in the same folder.
    std::vector<char> v2 = fileBytes(
        offs,
        flatlogs::logVersion2,
        { { t0 + 10, 0, "first v2 entry" }, { t0 + 11, 0, "second v2 entry" }, { t0 + 12, 0, "damaged v2 entry" } } );

    // The last entry fails its checksum, which only a file read as v2 can detect; it ends the run with 255.
    v2[offs[2] + flatlogs::logHeaderV2::headerSize( v2.data() + offs[2] ) + 4] ^= 0x01;

    auto f2 = writeLogFile( td.m_path, "dev1", "bintel2", t0 + 10, 0, v2 );

    REQUIRE( result.wait_for( std::chrono::seconds( 10 ) ) == std::future_status::ready );
    REQUIRE( result.get() == -1 );
    REQUIRE( out->find( "first v2 entry" ) != std::string::npos );
    REQUIRE( out->find( "second v2 entry" ) != std::string::npos );
    REQUIRE( countOf( *err,
                      f2.fullName() + " failed verification on code=10 at byte=" + std::to_string( offs[2] ) +
                          " (reason: checksum)" ) == 1 );
}

} // namespace logdumpTest
} // namespace libXWCTest
