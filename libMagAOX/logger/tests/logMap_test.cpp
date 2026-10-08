/** \file logMap_test.hpp
 * \brief Tests for the logMap class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <fstream>

#include "../../file/fileTimes.hpp"

#include "../logMap.hpp"
#include "../logMap.cpp"
#include "../types/text_log.hpp"

#include "logTestFiles.hpp"

namespace libXWCTest
{

namespace loggerTest
{

/** \defgroup logMap_unit_test logMap Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger::logMap tests
/** \ingroup logMap_unit_test
 *
 */
namespace logMapTest
{

// simple time struct to enable log structure creation
struct tmpt
{
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
    int nanosec;
};

// create a bunch of logs on disk to process
void createTestPaths( const std::string &basedir )
{
    std::filesystem::create_directory( basedir );

    std::vector<std::string> devs( { "dev1", "dev2", "dev3" } );

    std::vector<std::vector<tmpt>> ftimes( { /*dev1:*/ { { 2024, 11, 19, 0, 0, 0, 0 },
                                                         { 2024, 11, 19, 0, 0, 30, 0 },
                                                         { 2024, 11, 19, 2, 55, 26, 4000 },
                                                         { 2024, 11, 19, 5, 23, 0, 0 },
                                                         { 2024, 11, 21, 22, 0, 0, 0 },
                                                         { 2024, 11, 21, 23, 59, 59, 999999999 },
                                                         { 2024, 11, 23, 2, 30, 2, 2000 },
                                                         { 2024, 11, 23, 4, 45, 10, 12 } },
                                             /*dev2:*/
                                             { { 2024, 11, 19, 0, 0, 0, 0 },
                                               { 2024, 11, 19, 0, 0, 30, 0 },
                                               { 2024, 11, 19, 2, 55, 26, 4000 },
                                               { 2024, 11, 19, 5, 23, 0, 0 },
                                               { 2024, 11, 21, 22, 0, 0, 0 },
                                               { 2024, 11, 21, 23, 59, 59, 999999999 },
                                               { 2024, 11, 23, 2, 30, 2, 2000 },
                                               { 2024, 11, 23, 4, 45, 10, 12 } },
                                             /*dev3:*/
                                             { { 2024, 11, 19, 0, 0, 0, 0 },
                                               { 2024, 11, 19, 0, 0, 30, 0 },
                                               { 2024, 11, 19, 2, 55, 26, 4000 },
                                               { 2024, 11, 19, 5, 23, 0, 0 },
                                               { 2024, 11, 21, 22, 0, 0, 0 },
                                               { 2024, 11, 21, 23, 59, 59, 999999999 },
                                               { 2024, 11, 23, 2, 30, 2, 2000 },
                                               { 2024, 11, 23, 4, 45, 10, 12 } } } );

    for( size_t d = 0; d < devs.size(); ++d )
    {
        for( size_t f = 0; f < ftimes[d].size(); ++f )
        {
            tm uttime;
            uttime.tm_year = ftimes[d][f].year - 1900;
            uttime.tm_mon  = ftimes[d][f].month - 1;
            uttime.tm_mday = ftimes[d][f].day;
            uttime.tm_hour = ftimes[d][f].hour;
            uttime.tm_min  = ftimes[d][f].minute;
            uttime.tm_sec  = ftimes[d][f].second;

            time_t secs = timegm( &uttime );

            std::string fileName, relPath;

            MagAOX::file::fileTimeRelPath( fileName, relPath, devs[d], "xlog", secs, ftimes[d][f].nanosec );

            std::filesystem::create_directories( basedir + '/' + relPath );

            std::ofstream fout;
            fout.open( basedir + '/' + relPath + '/' + fileName );
            fout.close();
        }
    }
}

/// Building the app-to-file map
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Building the app-to-file map", "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    SECTION( "File matches middle file and one later on same day" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( lm.m_appToFileMap["dev1"].size() == 2 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119025526000004000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119052300000000000.xlog" );
    }

    SECTION( "File matches first file by delta-t and last file on same day" )
    {
        // This is inside second log, but will pick the first log to get the 60 second buffer
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000061000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119042200000000000.xrif" );

        MagAOX::logger::logMap lm;

        lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( lm.m_appToFileMap["dev1"].size() == 4 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000000000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000030000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119025526000004000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119052300000000000.xlog" );
    }

    SECTION( "File matches first file by delta-t and first file on next day" )
    {
        // This is inside second log, but will pick the first log to get the 60 second buffer
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000061000000000.xrif" );

        // This is inside the 2nd to last log, but since next log is < 3600 seconds we have to go to next day
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119052200000000000.xrif" );

        MagAOX::logger::logMap lm;

        lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( lm.m_appToFileMap["dev1"].size() == 5 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000000000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000030000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119025526000004000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119052300000000000.xlog" );
    }

    SECTION( "File matches last file on previous day and first file on current day (times are the same)" )
    {
        // This will pick the first log of 11/19 to get the 60 second buffer
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241121200030000000000.xrif" );

        // Same time is more than an hour before first log of 11/21
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241121200030000000000.xrif" );

        MagAOX::logger::logMap lm;

        lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( lm.m_appToFileMap["dev1"].size() == 2 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119052300000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_21/dev1_20241121220000000000000.xlog" );
    }

    SECTION( "Matches first and last overall files, last one is not > 1 hr" )
    {
        // this is 50 seconds into 2nd file, so will pick the first file which is > 60 secs
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000120000000000.xrif" );

        // this is 10 minutes before end of last log
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_23/cam1_20241123044500000000000.xrif" );

        MagAOX::logger::logMap lm;

        lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( lm.m_appToFileMap["dev1"].size() == 8 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000000000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000030000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119025526000004000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119052300000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_21/dev1_20241121220000000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_21/dev1_20241121235959999999999.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_23/dev1_20241123023002000002000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_23/dev1_20241123044510000000012.xlog" );
    }

    SECTION( "Missing following date directory uses the previous log directory" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241124030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241124040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap["dev1"].size() == 1 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_23/dev1_20241123044510000000012.xlog" );
    }
}

/// Building the app-to-file map with bad arguments
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Building the app-to-file map with bad arguments", "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    SECTION( "bad directory permissions" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241118030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/root/adlknalkejr111", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::eacces );
        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }

    SECTION( "directory does not exist" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241118030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_testX", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::dirnotfound );
        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }

    SECTION( "firstFile is not valid" )
    {
        MagAOX::file::stdFileName firstFile;
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::invalidconfig );
        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }

    SECTION( "lastFile is not valid" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241118030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile;

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::invalidconfig );
        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }

    SECTION( "wrong device name so no files" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241118030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_test", "dev6", ".xlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }

    SECTION( "wrong extension so no files" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241118030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".qlog", firstFile, lastFile );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }
}

/// Building the app-to-file map with errors
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Building the app-to-file map with errors", "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    SECTION( "No prior log" )
    {
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241118030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( lm.m_appToFileMap.size() == 0 );
    }
}

/** \cond logger_test_helpers
 */

/// The text and offset of every usable entry in a walk through the loaded files.
inline std::vector<std::pair<std::string, size_t>>
walkAll( const MagAOX::logger::logInMemory &lim /**< [in] loaded files */ )
{
    std::vector<std::pair<std::string, size_t>> found;

    MagAOX::logger::logMemoryWalk walk( lim );
    MagAOX::logger::logStep       step;
    size_t                        file   = 0;
    size_t                        offset = 0;
    while( walk.next( step, file, offset ) )
    {
        std::string text = "<unsupported>";
        if( step.m_entry.eventCode == MagAOX::logger::eventCodes::TEXT_LOG )
        {
            text = MagAOX::logger::text_log::message( const_cast<char *>( step.m_entry.msg ) );
        }
        found.push_back( { text, offset } );
    }

    return found;
}

/** \endcond
 */

/// Walking through loaded v1 files: the deliberate behaviour changes, and recovery after damage.
/** Two deliberate changes from the earlier lookups are covered: an entry with an unknown event code is stepped over
 * instead of being treated as damage, and recovery after damage never crosses into the next file.
 *
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Walking through loaded v1 log files", "[libMagAOX::logger::logMap]" )
{
    using namespace MagAOX::logger;

    const time_t t0 = 1732000000; // 2024-11-19

    SECTION( "An entry with an unknown event code is stepped over, labelled, and reported once" )
    {
        tempDir td( "logMap_walk_unknown" );

        std::vector<size_t> offs;
        std::vector<char>   bytes = fileBytes(
            offs, flatlogs::logVersion1, { { t0, 0, "a" }, { t0 + 1, 0, "u", unknownCode }, { t0 + 2, 0, "b" } } );
        auto sfn = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

        logMap<> lm;
        lm.m_appToFileMap["dev1"].insert( sfn );

        cerrCapture cap;

        logEntryRecord prior;
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 3, 0 ) ) == 0 );
        REQUIRE( text_log::message( prior.message() ) == "b" );

        logEntryRecord first;
        REQUIRE( lm.getPriorEntry( first, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0, 500 ) ) == 0 );
        REQUIRE( text_log::message( first.message() ) == "a" );

        logEntryRecord next;
        REQUIRE( lm.getNextEntry( next, first, "dev1" ) == 0 );
        REQUIRE( next.m_sourceOffset == prior.m_sourceOffset );
        REQUIRE( text_log::message( next.message() ) == "b" );

        // The unknown entry is still there to walk over, labelled as unsupported.
        auto all = walkAll( lm.m_appToBufferMap["dev1"] );
        REQUIRE( all.size() == 3 );
        REQUIRE( all[1].first == "<unsupported>" );
        REQUIRE( all[1].second == offs[1] );

        REQUIRE( cap.count( "doesn't know" ) == 1 );
        REQUIRE( cap.count( "eventCode=65000 count=1" ) == 1 );
        REQUIRE( cap.count( "Invalid log entry" ) == 0 );
        REQUIRE( lm.recoverableErrors() == 0 );
    }

    SECTION( "Recovery after damage near the end of one file stops at that file's end" )
    {
        tempDir td( "logMap_walk_boundary" );

        std::vector<size_t> offsA, offsB;
        std::vector<char>   bytesA =
            fileBytes( offsA, flatlogs::logVersion1, { { t0, 0, "a1" }, { t0 + 1, 0, "a2" }, { t0 + 2, 0, "a3" } } );
        std::vector<char> bytesB = fileBytes(
            offsB, flatlogs::logVersion1, { { t0 + 10, 0, "b1" }, { t0 + 11, 0, "b2" }, { t0 + 12, 0, "b3" } } );

        bytesA[offsA[2]] = static_cast<char>( 0x7E ); // a3's priority is not a defined one

        // Over A alone there is nowhere to restart; over A followed by B the scan would restart in B.
        // As in a walk, the earliest plausible time is that of the previous usable entry, a2.
        flatlogs::timespecX a2Ts( t0 + 1, 0 );
        REQUIRE( logMapResyncV1( bytesA.data(), bytesA.size(), offsA[2], &a2Ts ) == flatlogs::entryNotFound );
        std::vector<char> joined = bytesA;
        joined.insert( joined.end(), bytesB.begin(), bytesB.end() );
        REQUIRE( logMapResyncV1( joined.data(), joined.size(), offsA[2], &a2Ts ) == bytesA.size() );

        auto sfnA = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytesA );
        auto sfnB = writeLogFile( td.m_path, "dev1", "binlog", t0 + 10, 0, bytesB );

        cerrCapture cap;

        logInMemory lim;
        REQUIRE( lim.loadFile( sfnA ) == 0 );
        REQUIRE( lim.loadFile( sfnB ) == 0 );

        REQUIRE( cap.count( "sourceByte=" + std::to_string( offsA[2] ) + " reason=structure resyncByte=<none>" ) == 1 );
        REQUIRE( lim.m_recoverableErrors == 1 );

        auto all = walkAll( lim );
        REQUIRE( all.size() == 5 );
        REQUIRE( all[0].first == "a1" );
        REQUIRE( all[1].first == "a2" );
        REQUIRE( all[2].first == "b1" );
        REQUIRE( all[2].second == 0 );
        REQUIRE( all[3].first == "b2" );
        REQUIRE( all[4].first == "b3" );
    }

    SECTION( "Entries of an unknown type inside damage are not chosen as the place to restart" )
    {
        tempDir td( "logMap_walk_lookalike" );

        std::vector<size_t> offs;
        std::vector<char>   bytes = fileBytes( offs,
                                               flatlogs::logVersion1,
                                               { { t0, 0, "e1" },
                                                 { t0 + 1, 0, "e2" },
                                                 { t0 + 2, 0, "u1", unknownCode },
                                                 { t0 + 3, 0, "u2", unknownCode },
                                                 { t0 + 4, 0, "u3", unknownCode },
                                                 { t0 + 5, 0, "e3" },
                                                 { t0 + 6, 0, "e4" },
                                                 { t0 + 7, 0, "e5" } } );

        bytes[offs[1]] = static_cast<char>( 0x7E ); // e2's priority is not a defined one

        flatlogs::timespecX e1Ts( t0, 0 );
        REQUIRE( logMapResyncV1( bytes.data(), bytes.size(), offs[1], &e1Ts ) == offs[5] );

        auto sfn = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes );

        cerrCapture cap;

        logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == 0 );
        REQUIRE( cap.count( "sourceByte=" + std::to_string( offs[1] ) +
                            " reason=structure resyncByte=" + std::to_string( offs[5] ) ) == 1 );

        auto all = walkAll( lim );
        REQUIRE( all.size() == 4 );
        REQUIRE( all[0].first == "e1" );
        REQUIRE( all[1].first == "e3" );
        REQUIRE( all[2].first == "e4" );
        REQUIRE( all[3].first == "e5" );
    }

    SECTION( "A v1 entry earlier than the previous one in the same file is damage" )
    {
        tempDir td( "logMap_walk_order" );

        std::vector<size_t> offs;
        std::vector<char>   bytes = fileBytes( offs,
                                               flatlogs::logVersion1,
                                               { { t0 + 5, 0, "e1" },
                                                 { t0, 0, "early" },
                                                 { t0 + 6, 0, "e2" },
                                                 { t0 + 7, 0, "e3" },
                                                 { t0 + 8, 0, "e4" } } );

        auto sfn = writeLogFile( td.m_path, "dev1", "binlog", t0 + 5, 0, bytes );

        cerrCapture cap;

        logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == 0 );
        REQUIRE( cap.count( "sourceByte=" + std::to_string( offs[1] ) +
                            " reason=structure resyncByte=" + std::to_string( offs[2] ) ) == 1 );

        auto all = walkAll( lim );
        REQUIRE( all.size() == 4 );
        REQUIRE( all[1].first == "e2" );
    }

    SECTION( "A file that fails to load is read and reported once, however many lookups need it" )
    {
        tempDir td( "logMap_walk_rejected" );

        std::vector<char> junk( 200, static_cast<char>( 0x7E ) );
        auto              sfn = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, junk );

        logMap<> lm;
        lm.m_appToFileMap["dev1"].insert( sfn );

        cerrCapture cap;

        logEntryRecord prior;
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 3, 0 ) ) == -1 );
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 4, 0 ) ) == -1 );

        REQUIRE( cap.count( "Possibly corrupt logfile: source=" + sfn.fullName() ) == 1 );
        REQUIRE( cap.count( "Invalid log entry" ) == 1 );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_rejectedFiles.count( sfn.fullName() ) == 1 );
    }
}

/// The rule linking extensions to format versions.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Log file extensions and versions", "[libMagAOX::logger::logMap]" )
{
    using namespace MagAOX::logger;

    REQUIRE( logExtensionsFor( ".binlog" ) == std::vector<std::string>( { ".binlog", ".binlog2" } ) );
    REQUIRE( logExtensionsFor( ".bintel" ) == std::vector<std::string>( { ".bintel", ".bintel2" } ) );
    REQUIRE( logExtensionsFor( "binlog" ) == std::vector<std::string>( { "binlog", "binlog2" } ) );
    REQUIRE( logExtensionsFor( ".foo" ) == std::vector<std::string>( { ".foo", ".foo2" } ) );
    REQUIRE( logExtensionsFor( ".binlog2" ) == std::vector<std::string>( { ".binlog2" } ) );
    REQUIRE( logExtensionsFor( ".bintel2" ) == std::vector<std::string>( { ".bintel2" } ) );
    REQUIRE( logExtensionsFor( "" ) == std::vector<std::string>( { "" } ) );

    REQUIRE( logVersionForExtension( ".binlog" ) == flatlogs::logVersion1 );
    REQUIRE( logVersionForExtension( ".bintel" ) == flatlogs::logVersion1 );
    REQUIRE( logVersionForExtension( ".binlog2" ) == flatlogs::logVersion2 );
    REQUIRE( logVersionForExtension( ".bintel2" ) == flatlogs::logVersion2 );
    REQUIRE( logVersionForExtension( ".foo" ) == flatlogs::logVersion1 );
    REQUIRE( logVersionForExtension( ".foo2" ) == flatlogs::logVersion2 );
    REQUIRE( logVersionForExtension( "" ) == flatlogs::logVersion1 );

    REQUIRE( logVersionForFile( "/data/dev1/2024_11_19/dev1_20241119000000000000000.binlog" ) ==
             flatlogs::logVersion1 );
    REQUIRE( logVersionForFile( "/data/dev1/2024_11_19/dev1_20241119000000000000000.bintel2" ) ==
             flatlogs::logVersion2 );
    REQUIRE( logVersionForFile( "dev1_20241119000000000000000.binlog2" ) == flatlogs::logVersion2 );
    REQUIRE( logVersionForFile( "noextension" ) == flatlogs::logVersion1 );
}

/// Finding and reading files of both versions.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Finding and reading v1 and v2 log files", "[libMagAOX::logger::logMap]" )
{
    using namespace MagAOX::logger;

    const time_t t0 = 1732000000; // 2024-11-19 07:06:40

    SECTION( "A v2 file is read by v2 rules, and its entries are verified" )
    {
        tempDir td( "logMap_v2_only" );

        std::vector<size_t> offs;
        std::vector<char>   bytes =
            fileBytes( offs, flatlogs::logVersion2, { { t0, 0, "a" }, { t0 + 1, 0, "b" }, { t0 + 2, 0, "c" } } );
        auto sfn = writeLogFile( td.m_path, "dev1", "binlog2", t0, 0, bytes );

        logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == 0 );
        REQUIRE( lim.m_loadedFiles.size() == 1 );
        REQUIRE( lim.m_loadedFiles[0].m_version == flatlogs::logVersion2 );

        auto all = walkAll( lim );
        REQUIRE( all.size() == 3 );
        REQUIRE( all[0].first == "a" );
        REQUIRE( all[2].first == "c" );
        REQUIRE( all[2].second == offs[2] );

        logMemoryWalk walk( lim );
        logStep       step;
        size_t        file, offset;
        while( walk.next( step, file, offset ) )
        {
            REQUIRE( step.m_integrity == logIntegrity::verified );
        }
    }

    SECTION( "A v1 file followed by a v2 file: each is read by its own rules" )
    {
        tempDir td( "logMap_cutover" );

        std::vector<size_t> offs1, offs2;
        auto                sfn1 =
            writeLogFile( td.m_path,
                          "dev1",
                          "binlog",
                          t0,
                          0,
                          fileBytes( offs1, flatlogs::logVersion1, { { t0, 0, "v1a" }, { t0 + 1, 0, "v1b" } } ) );
        auto sfn2 =
            writeLogFile( td.m_path,
                          "dev1",
                          "binlog2",
                          t0 + 10,
                          0,
                          fileBytes( offs2, flatlogs::logVersion2, { { t0 + 10, 0, "v2a" }, { t0 + 11, 0, "v2b" } } ) );

        cerrCapture cap;

        logInMemory lim;
        REQUIRE( lim.loadFile( sfn1 ) == 0 );
        REQUIRE( lim.loadFile( sfn2 ) == 0 );
        REQUIRE( cap.m_out.str().empty() );

        std::vector<std::string>  texts;
        std::vector<logIntegrity> verdicts;

        logMemoryWalk walk( lim );
        logStep       step;
        size_t        file, offset;
        while( walk.next( step, file, offset ) )
        {
            texts.push_back( text_log::message( const_cast<char *>( step.m_entry.msg ) ) );
            verdicts.push_back( step.m_integrity );
        }

        REQUIRE( texts == std::vector<std::string>( { "v1a", "v1b", "v2a", "v2b" } ) );
        REQUIRE( verdicts[0] == logIntegrity::notChecked );
        REQUIRE( verdicts[1] == logIntegrity::notChecked );
        REQUIRE( verdicts[2] == logIntegrity::verified );
        REQUIRE( verdicts[3] == logIntegrity::verified );
    }

    SECTION( "A range starting before the first v2 file still finds the v2 files" )
    {
        tempDir td( "logMap_find_both" );

        // 2024-11-19: v1 files at 00:00 and 01:00, v2 files at 02:00, 03:00 and 05:00.
        const time_t        day = 1731974400;
        std::vector<size_t> offs;
        writeLogFile(
            td.m_path, "dev1", "binlog", day, 0, fileBytes( offs, flatlogs::logVersion1, { { day, 0, "x" } } ) );
        writeLogFile( td.m_path,
                      "dev1",
                      "binlog",
                      day + 3600,
                      0,
                      fileBytes( offs, flatlogs::logVersion1, { { day + 3600, 0, "x" } } ) );
        for( time_t h : { 2, 3, 5 } )
        {
            writeLogFile( td.m_path,
                          "dev1",
                          "binlog2",
                          day + h * 3600,
                          0,
                          fileBytes( offs, flatlogs::logVersion2, { { day + h * 3600, 0, "x" } } ) );
        }

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119013000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119033000000000000.xrif" );

        logMap<> lm;
        REQUIRE( lm.loadAppToFileMap( td.m_path, "dev1", ".binlog", firstFile, lastFile ) == mx::error_t::noerror );

        REQUIRE( lm.m_appToFileMap["dev1"].size() == 4 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == td.m_path + "/dev1/2024_11_19/dev1_20241119010000000000000.binlog" );
        ++it;
        REQUIRE( it->fullName() == td.m_path + "/dev1/2024_11_19/dev1_20241119020000000000000.binlog2" );
        ++it;
        REQUIRE( it->fullName() == td.m_path + "/dev1/2024_11_19/dev1_20241119030000000000000.binlog2" );
        ++it;
        REQUIRE( it->fullName() == td.m_path + "/dev1/2024_11_19/dev1_20241119050000000000000.binlog2" );
    }

    SECTION( "Finding no files adds nothing, and leaves reporting to the caller" )
    {
        tempDir td( "logMap_no_files" );
        std::filesystem::create_directories( td.m_path + "/dev1/2024_11_19" );

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119013000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119033000000000000.xrif" );

        cerrCapture cap;

        logMap<> lm;
        REQUIRE( lm.loadAppToFileMap( td.m_path, "dev1", ".bintel", firstFile, lastFile ) == mx::error_t::noerror );
        REQUIRE( lm.loadAppToFileMap( td.m_path, "dev1", ".bintel", firstFile, lastFile ) == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap.size() == 0 );

        // The caller may search other directories for the same device, so only it can report this.
        REQUIRE( cap.m_out.str().empty() );
    }

    SECTION( "A v1 file named as v2 is skipped with one report, and read once across lookups" )
    {
        tempDir td( "logMap_misnamed" );

        std::vector<size_t> offs;
        auto sfn = writeLogFile( td.m_path,
                                 "dev1",
                                 "binlog2",
                                 t0,
                                 0,
                                 fileBytes( offs, flatlogs::logVersion1, { { t0, 0, "a" }, { t0 + 1, 0, "b" } } ) );

        logMap<> lm;
        lm.m_appToFileMap["dev1"].insert( sfn );

        cerrCapture cap;

        logEntryRecord prior;
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 3, 0 ) ) == -1 );
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 4, 0 ) ) == -1 );

        REQUIRE( cap.count( "Log file skipped: its extension says v2 but its first entry is not v2: source=" +
                            sfn.fullName() ) == 1 );
        REQUIRE( cap.count( "Invalid log entry" ) == 0 );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_rejectedFiles.count( sfn.fullName() ) == 1 );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_memory.empty() );
        REQUIRE( lm.recoverableErrors() == 1 );
    }

    SECTION( "A v2 checksum failure is reported with the reason, and reading carries on" )
    {
        tempDir td( "logMap_checksum" );

        std::vector<size_t> offs;
        std::vector<char>   bytes = fileBytes(
            offs, flatlogs::logVersion2, { { t0, 0, "aaaa" }, { t0 + 1, 0, "bbbb" }, { t0 + 2, 0, "cccc" } } );

        // Change one byte of the second entry's message.
        size_t msgAt = offs[1] + flatlogs::logHeaderV2::headerSize( bytes.data() + offs[1] );
        bytes[msgAt + 4] ^= 0x01;

        auto sfn = writeLogFile( td.m_path, "dev1", "binlog2", t0, 0, bytes );

        cerrCapture cap;

        logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == 0 );
        REQUIRE( cap.count( "sourceByte=" + std::to_string( offs[1] ) +
                            " reason=checksum resyncByte=" + std::to_string( offs[2] ) ) == 1 );
        REQUIRE( lim.m_recoverableErrors == 1 );

        auto all = walkAll( lim );
        REQUIRE( all.size() == 2 );
        REQUIRE( all[0].first == "aaaa" );
        REQUIRE( all[1].first == "cccc" );
    }
}

/// Lookups that return their own copy of the entry.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "Looking up entries across v1 and v2 files", "[libMagAOX::logger::logMap]" )
{
    using namespace MagAOX::logger;

    const time_t t0 = 1732000000; // 2024-11-19 07:06:40

    SECTION( "Prior and next entries across the cutover carry their verdict and source" )
    {
        tempDir td( "logMap_lookup_cutover" );

        std::vector<size_t> offs1, offs2;
        auto                sfn1 =
            writeLogFile( td.m_path,
                          "dev1",
                          "binlog",
                          t0,
                          0,
                          fileBytes( offs1, flatlogs::logVersion1, { { t0, 0, "v1a" }, { t0 + 1, 0, "v1b" } } ) );
        auto sfn2 = writeLogFile(
            td.m_path,
            "dev1",
            "binlog2",
            t0 + 10,
            0,
            fileBytes( offs2, flatlogs::logVersion2, { { t0 + 10, 0, "u", unknownCode }, { t0 + 11, 0, "v2a" } } ) );

        logMap<> lm;
        lm.m_appToFileMap["dev1"].insert( sfn1 );
        lm.m_appToFileMap["dev1"].insert( sfn2 );

        cerrCapture cap;

        logEntryRecord prior;
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 5, 0 ) ) == 0 );
        REQUIRE( text_log::message( prior.message() ) == "v1b" );
        REQUIRE( prior.m_version == flatlogs::logVersion1 );
        REQUIRE( prior.m_integrity == logIntegrity::notChecked );
        REQUIRE( prior.m_sourceFile == sfn1.fullName() );
        REQUIRE( prior.m_sourceOffset == offs1[1] );
        REQUIRE( prior.m_unsupportedType == false );

        logEntryRecord next;
        REQUIRE( lm.getNextEntry( next, prior, "dev1" ) == 0 );
        REQUIRE( text_log::message( next.message() ) == "v2a" );
        REQUIRE( next.m_version == flatlogs::logVersion2 );
        REQUIRE( next.m_integrity == logIntegrity::verified );
        REQUIRE( next.m_sourceFile == sfn2.fullName() );
        REQUIRE( next.m_sourceOffset == offs2[1] );
        REQUIRE( next.m_ts == flatlogs::timespecX( t0 + 11, 0 ) );

        // Messages are copied to 8-byte boundaries for both versions, whatever the header size.
        REQUIRE( reinterpret_cast<uintptr_t>( prior.message() ) % 8 == 0 );
        REQUIRE( reinterpret_cast<uintptr_t>( next.message() ) % 8 == 0 );
        REQUIRE( prior.m_bytes.size() - prior.m_entrySize < 8 );
        REQUIRE( next.m_bytes.size() - next.m_entrySize < 8 );
        REQUIRE( next.m_entrySize == prior.m_entrySize + 5 ); // same message length; v2 adds 5 bytes

        logEntryRecord none;
        REQUIRE( lm.getNextEntry( none, next, "dev1" ) == 1 );
        REQUIRE( none.empty() );

        // The entry of an unsupported type can be looked up, and is labelled.
        logEntryRecord unsupported;
        REQUIRE( lm.getPriorEntry( unsupported, "dev1", unknownCode, flatlogs::timespecX( t0 + 12, 0 ) ) == 0 );
        REQUIRE( unsupported.m_unsupportedType == true );
        REQUIRE( unsupported.m_integrity == logIntegrity::verified );

        REQUIRE( cap.count( "eventCode=65000 count=1" ) == 1 );
    }

    SECTION( "Loading an earlier file doesn't break results returned before it" )
    {
        tempDir td( "logMap_lookup_prepend" );

        std::vector<size_t> offs;
        auto sfnA = writeLogFile( td.m_path,
                                  "dev1",
                                  "binlog",
                                  t0,
                                  0,
                                  fileBytes( offs, flatlogs::logVersion1, { { t0, 0, "a1" }, { t0 + 1, 0, "a2" } } ) );
        auto sfnB =
            writeLogFile( td.m_path,
                          "dev1",
                          "binlog2",
                          t0 + 100,
                          0,
                          fileBytes( offs,
                                     flatlogs::logVersion2,
                                     { { t0 + 100, 0, "b1" }, { t0 + 101, 0, "b2" }, { t0 + 102, 0, "b3" } } ) );

        // A file before A, so A isn't the first file of the app. loadFiles can't yet load backwards to an app's
        // first file (an existing limitation, not covered here).
        auto sfnZ = writeLogFile( td.m_path,
                                  "dev1",
                                  "binlog",
                                  t0 - 1000,
                                  0,
                                  fileBytes( offs, flatlogs::logVersion1, { { t0 - 1000, 0, "z" } } ) );

        logMap<> lm;
        lm.m_appToFileMap["dev1"].insert( sfnZ );
        lm.m_appToFileMap["dev1"].insert( sfnA );
        lm.m_appToFileMap["dev1"].insert( sfnB );

        // Only B is loaded for this lookup.
        logEntryRecord b1;
        REQUIRE( lm.getPriorEntry( b1, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 100, 500 ) ) == 0 );
        REQUIRE( text_log::message( b1.message() ) == "b1" );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_loadedFiles.size() == 1 );

        // This lookup loads A in front of B, which moves B's bytes in memory.
        logEntryRecord a2;
        REQUIRE( lm.getPriorEntry( a2, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 50, 0 ) ) == 0 );
        REQUIRE( text_log::message( a2.message() ) == "a2" );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_loadedFiles.size() == 2 );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_loadedFiles[1].m_name == sfnB.fullName() );

        // The earlier result still holds its entry, and the next lookup from it still works.
        REQUIRE( text_log::message( b1.message() ) == "b1" );
        logEntryRecord b2;
        REQUIRE( lm.getNextEntry( b2, b1, "dev1" ) == 0 );
        REQUIRE( text_log::message( b2.message() ) == "b2" );
    }

    SECTION( "A result whose file isn't loaded can't be followed" )
    {
        logMap<> lm;

        logEntryRecord current;
        current.m_bytes.assign( 20, 0 );
        current.m_sourceFile = "/no/such/file.binlog";

        logEntryRecord next;
        REQUIRE( lm.getNextEntry( next, current, "dev1" ) == -1 );
        REQUIRE( lm.getNextEntry( next, logEntryRecord(), "dev1" ) == -1 );
    }
}

} // namespace logMapTest
} // namespace loggerTest
} // namespace libXWCTest
