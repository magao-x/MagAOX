/** \file logMap_test.hpp
 * \brief Tests for the logMap class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <fstream>

#include "../../file/fileTimes.hpp"

#include "../logFileRaw.hpp"
#include "../logMap.hpp"
#include "../logMap.cpp"

// Fault injection. Each block below compiles logMap.hpp a second time inside a test
// namespace with one XWCTEST_ fault macro defined. The macro turns one production error
// branch on, such as a thrown exception or a bad return value, that cannot be reached
// from outside. The include guard is undefined first so the header is really re-read.
// The tests then use the namespaced logMap type to reach that branch.

// Forces addFileListToFileMap to throw an xwcException inside its try block.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_AFLTFM_XWCE_ns
#define XWCTEST_LOGMAP_AFLTFM_XWCE
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_AFLTFM_XWCE

// Forces addFileListToFileMap to throw a std::bad_alloc inside its try block.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_AFLTFM_BADALL_ns
#define XWCTEST_LOGMAP_AFLTFM_BADALL
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_AFLTFM_BADALL

// Forces addFileListToFileMap to throw a plain std::exception inside its try block.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_AFLTFM_EXCEPTION_ns
#define XWCTEST_LOGMAP_AFLTFM_EXCEPTION
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_AFLTFM_EXCEPTION

// Forces loadAppToFileMap to throw while parsing a filename during its backward search.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_BADALL1_ns
#define XWCTEST_LOGMAP_LATFM_BADALL1
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_BADALL1

// Forces loadAppToFileMap to throw at the start of a forward search day.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_BADALL2_ns
#define XWCTEST_LOGMAP_LATFM_BADALL2
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_BADALL2

// Forces loadAppToFileMap to throw while building the file list when the previous and
// following logs are on the same day.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_BADALL3_ns
#define XWCTEST_LOGMAP_LATFM_BADALL3
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_BADALL3

// Forces loadAppToFileMap to throw while building the file list for the previous day
// when the previous and following logs are on different days.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_XWCE4_ns
#define XWCTEST_LOGMAP_LATFM_XWCE4
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_XWCE4

// Forces loadAppToFileMap to throw inside the loop over intervening days.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_XWCE5_ns
#define XWCTEST_LOGMAP_LATFM_XWCE5
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_XWCE5

// Forces loadAppToFileMap to throw while building the file list for the day of the
// following log.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_XWCE6_ns
#define XWCTEST_LOGMAP_LATFM_XWCE6
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_XWCE6

// Makes the directory existence check in loadAppToFileMap report a permission error.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_DIREXISTS_ERRC_ns
#define XWCTEST_LOGMAP_LATFM_DIREXISTS_ERRC
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_DIREXISTS_ERRC

// Pushes the following file index past the end of the list in the same day case, so the
// file count sanity check in loadAppToFileMap fires.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_SIZEERR1_ns
#define XWCTEST_LOGMAP_LATFM_SIZEERR1
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_SIZEERR1

// Pushes the following file index past the end of the list for the day of the following
// log, so that file count sanity check in loadAppToFileMap fires.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LATFM_SIZEERR2_ns
#define XWCTEST_LOGMAP_LATFM_SIZEERR2
#include "../logMap.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMAP_LATFM_SIZEERR2

// logInMemory::loadFile is defined out of line in logMap.cpp, unlike the inline template
// methods of logMap<verboseT> above. So reaching its fault path needs a namespaced
// re-inclusion of both the header and the source together. The macro pretends the read
// returned zero bytes, so the short read check in loadFile fires.
#undef logger_logMap_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMAP_LOADFILE_SHORTREAD_ns
#define XWCTEST_LOGINMEMORY_LOADFILE_SHORTREAD
#include "../logMap.hpp"
#include "../logMap.cpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGINMEMORY_LOADFILE_SHORTREAD

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
    // Start clean. Stale files from an earlier run with different fixture times would
    // otherwise show up in the file maps and break the counts below.
    std::filesystem::remove_all( basedir );
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

// Two simple log types with distinct event codes. They are used with logFileRaw to build
// real on-disk log files for testing getPriorLog, getNextLog, and loadFiles. Those
// functions need genuine flatlog formatted binary content to scan.
struct dummyLogA
{
    // This is a real generated event code. The merged logMap only scans entries whose
    // code eventCodeName() recognizes. Unknown codes read as corruption and are resynced
    // away.
    static const flatlogs::eventCodeT eventCode   = MagAOX::logger::eventCodes::TEXT_LOG;
    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef std::string messageT;

    static const char *msg()
    {
        return "A";
    }

    static flatlogs::msgLenT length( const messageT &msg )
    {
        return msg.size();
    }

    static int format( void *msgBuffer, const messageT &msg )
    {
        memcpy( msgBuffer, msg.data(), msg.size() );
        return 0;
    }
};

struct dummyLogB
{
    static const flatlogs::eventCodeT eventCode   = MagAOX::logger::eventCodes::USER_LOG; // This is a real code. See dummyLogA.

    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef std::string messageT;

    static const char *msg()
    {
        return "B";
    }

    static flatlogs::msgLenT length( const messageT &msg )
    {
        return msg.size();
    }

    static int format( void *msgBuffer, const messageT &msg )
    {
        memcpy( msgBuffer, msg.data(), msg.size() );
        return 0;
    }
};

// Out-of-line definitions, needed because getPriorLog takes eventCode by const reference.
const flatlogs::eventCodeT dummyLogA::eventCode;
const flatlogs::eventCodeT dummyLogB::eventCode;

// A log type that always declares a large message length regardless of what is actually
// written. Its header can be used to fabricate a truncated or corrupt trailing entry. The
// buffer that createLog allocates is sized to match the declared length, so writing a
// small message into it is memory safe. Only the declared length in the header is a lie.
struct dummyLogBigDeclared
{
    // This is a real generated event code. See dummyLogA above. The raw literal 50 used
    // here before collided with eventCodes::SOFTWARE_LOG, and unrecognized codes now read
    // as corruption.
    static const flatlogs::eventCodeT eventCode   = MagAOX::logger::eventCodes::STATE_CHANGE;
    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef int messageT;

    static flatlogs::msgLenT length( const messageT & )
    {
        return 5000;
    }

    static int format( void *msgBuffer, const messageT &msg )
    {
        memcpy( msgBuffer, &msg, sizeof( msg ) );
        return 0;
    }
};
const flatlogs::eventCodeT dummyLogBigDeclared::eventCode;

// Writes a single-file log with one dummyLogA entry per supplied timestamp, and returns it
// as a stdFileName ready to hand to logInMemory::loadFile directly.
MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> writeSingleLogFile( const std::string        &dir,
                                                                     const std::string        &dev,
                                                                     const std::vector<time_t> &times )
{
    std::filesystem::remove_all( dir );

    MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> writer;
    writer.logPath( dir );
    writer.logName( dev );
    writer.logExt( "xlog" );
    writer.maxLogSize( 1000000 ); // This is large enough that all entries land in one file.

    flatlogs::bufferPtrT buf;
    for( auto t : times )
    {
        flatlogs::logHeader::createLog<dummyLogA>( buf, flatlogs::timespecX( t, 0 ), dummyLogA::msg(), flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    }
    writer.close();

    std::string fileName, relPath;
    MagAOX::file::fileTimeRelPath( fileName, relPath, dev, "xlog", times.front(), 0 );

    MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( dir + '/' + relPath + '/' + fileName );
    REQUIRE( sfn.valid() );
    return sfn;
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

/// The exception handling inside addFileListToFileMap. The first three sections use
/// fault-injected builds of logMap that throw from inside the function. The last uses the
/// normal build.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "addFileListToFileMap nests or reports exceptions", "[libMagAOX::logger::logMap]" )
{
    std::vector<std::string> flist{ "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119000000000000000.xlog" };

    SECTION( "nests an xwcException" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_AFLTFM_XWCE_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        bool caught = false;
        try
        {
            lm.addFileListToFileMap( "dev1", flist, 0, flist.size() );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "nests a std::bad_alloc" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_AFLTFM_BADALL_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        bool caught = false;
        try
        {
            lm.addFileListToFileMap( "dev1", flist, 0, flist.size() );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "reports a std::exception as an error" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_AFLTFM_EXCEPTION_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        mx::error_t rv = lm.addFileListToFileMap( "dev1", flist, 0, flist.size() );

        REQUIRE( rv == mx::error_t::std_exception );
    }

    SECTION( "skips a valid filename belonging to a different app" )
    {
        // "dev10" is a valid stdFileName with a different appName() than "dev1". So it
        // should be skipped rather than added to the file map for dev1.
        std::vector<std::string> otherAppFlist{ "/tmp/logMap_test/dev1/2024_11_19/dev10_20241119000000000000000.xlog" };

        MagAOX::logger::logMap lm;

        mx::error_t rv = lm.addFileListToFileMap( "dev1", otherAppFlist, 0, otherAppFlist.size() );

        REQUIRE( rv == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap["dev1"].size() == 0 );
    }
}

/// loadAppToFileMap nests and rethrows exceptions raised while parsing filenames or
/// building file lists. Each section uses a fault-injected build that throws at one
/// specific point in the search.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "loadAppToFileMap nests exceptions raised while searching for files", "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    SECTION( "an exception while parsing filenames during the backward search" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_BADALL1_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        bool caught = false;
        try
        {
            lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "an exception while parsing filenames during the forward search" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_BADALL2_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        bool caught = false;
        try
        {
            lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "an exception building the file list when prev and following logs are on the same day" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_BADALL3_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        bool caught = false;
        try
        {
            lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "an exception building the file list when prev and following logs are on different days" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_XWCE4_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        // This spans 2024_11_19 as the previous day to 2024_11_21 as the following day.
        // See "File matches first file by delta-t and first file on next day" above for
        // the non-throwing version.
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000061000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119052200000000000.xrif" );

        bool caught = false;
        try
        {
            lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "an exception building the file list for an intervening day" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_XWCE5_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        // This spans 2024_11_19 as the previous day to 2024_11_23 as the following day.
        // So the intervening day loop runs for 2024_11_20, 2024_11_21, and 2024_11_22.
        // See "Matches first and last overall files" above for the non-throwing version.
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000120000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_23/cam1_20241123044500000000000.xrif" );

        bool caught = false;
        try
        {
            lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "an exception building the file list for the following log's own day" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_XWCE6_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000120000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_23/cam1_20241123044500000000000.xrif" );

        bool caught = false;
        try
        {
            lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }
}

/// The defensive filesystem error and file count sanity checks in loadAppToFileMap. Each
/// section uses a fault-injected build that forces one check to fire.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "loadAppToFileMap reports filesystem and file-count errors", "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    SECTION( "a filesystem error while checking a day during the forward search" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_DIREXISTS_ERRC_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        mx::error_t rv = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( rv == mx::error_t::eacces );
    }

    SECTION( "a miscounted file list when prev and following logs are on the same day" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_SIZEERR1_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        mx::error_t rv = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( rv == mx::error_t::sizeerr );
    }

    SECTION( "a miscounted file list for the following log's own day" )
    {
        MagAOX::logger::XWCTEST_LOGMAP_LATFM_SIZEERR2_ns::logMap<XWC_DEFAULT_VERBOSITY> lm;

        // Unlike the different days cases above, this pair finds the following log
        // normally. The forward search lands on 2024_11_21 rather than going through the
        // not found fallback. That is required to reach the branch this macro targets.
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119000061000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119052200000000000.xrif" );

        mx::error_t rv = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( rv == mx::error_t::sizeerr );
    }
}

/// loadAppToFileMap skips files and subdirectories that do not contribute to the map.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "loadAppToFileMap skips non-standard filenames and empty subdirectories", "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    SECTION( "a non-standard filename in the search directory is ignored" )
    {
        // Both files match the prefix and extension filter of getFileNames(). They start
        // with "dev1" and end with ".xlog". But they are not valid stdFileNames, so they
        // must be skipped rather than corrupting the file count. "dev1_0000.xlog" sorts
        // before all the real timestamped files, so the forward search from low to high
        // index hits it first. "dev1_zzzz.xlog" sorts after all of them, so the backward
        // search from high to low index hits it first.
        std::ofstream junkLow( "/tmp/logMap_test/dev1/2024_11_19/dev1_0000.xlog" );
        junkLow.close();
        std::ofstream junkHigh( "/tmp/logMap_test/dev1/2024_11_19/dev1_zzzz.xlog" );
        junkHigh.close();

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t rv = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( rv == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap["dev1"].size() == 2 );
    }

    SECTION( "an existing but empty intervening subdirectory is skipped" )
    {
        // 2024_11_20 has no dev1 files. Both the backward search from 2024_11_21 and the
        // forward search from 2024_11_19 must step over it.
        std::filesystem::create_directories( "/tmp/logMap_test/dev1/2024_11_20" );

        MagAOX::file::stdFileName firstFile( "cam1/2024_11_21/cam1_20241121210000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2024_11_19/cam1_20241119052200000000000.xrif" );

        MagAOX::logger::logMap lm;

        mx::error_t rv = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

        REQUIRE( rv == mx::error_t::noerror );

        // firstFile at 21:00 on 11_21 is well after the last 11_19 file. So the backward
        // search finds only the last 11_19 file at 05:23:00. prevLogFile_n is 3, so only
        // that one index is included from that day. lastFile at 05:22 on 11_19 means
        // follts lands after all of the 11_19 files. So the forward search finds only the
        // first 11_21 file at 22:00:00. follLogFile_n is 0, so only that one index is
        // included from that day.
        REQUIRE( lm.m_appToFileMap["dev1"].size() == 2 );
        auto it = lm.m_appToFileMap["dev1"].begin();
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_19/dev1_20241119052300000000000.xlog" );
        ++it;
        REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_21/dev1_20241121220000000000000.xlog" );
    }
}

/// loadAppToFileMap falls back to the last available day when no following log is found
/// within the search span.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "loadAppToFileMap falls back to the last available day when no following log is found",
           "[libMagAOX::logger::logMap]" )
{
    createTestPaths( "/tmp/logMap_test" );

    // firstFile is 61 seconds after the last 2024_11_23 entry at 04:45:10. So the
    // backward search finds it immediately and prevLogSubDir is 2024_11_23.
    MagAOX::file::stdFileName firstFile( "cam1/2024_11_23/cam1_20241123044611000000000.xrif" );

    // lastFile is about 3 weeks after the last real data. So the forward search exhausts
    // its span without finding anything. The fallback search then steps backward from
    // 2024_12_15 until it finds an existing directory. That directory is 2024_11_23, the
    // same day as prevLogSubDir.
    MagAOX::file::stdFileName lastFile( "cam1/2024_12_15/cam1_20241215000000000000000.xrif" );

    MagAOX::logger::logMap lm;

    mx::error_t rv = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );

    REQUIRE( rv == mx::error_t::noerror );

    // Only the last 2024_11_23 file should be included. prevLogFile_n is its own index.
    // With no following log found, follLogFile_n falls back to tmp_flist.size(), which is
    // one past the last file in the directory for that day.
    REQUIRE( lm.m_appToFileMap["dev1"].size() == 1 );
    auto it = lm.m_appToFileMap["dev1"].begin();
    REQUIRE( it->fullName() == "/tmp/logMap_test/dev1/2024_11_23/dev1_20241123044510000000012.xlog" );
}

/// getPriorLog scans a loaded in-memory buffer for the last log at or before a timestamp.
/// The buffer is loaded from one real file with five entries.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "getPriorLog scans a loaded buffer for the last log at or before a timestamp",
           "[libMagAOX::logger::logMap]" )
{
    std::filesystem::remove_all( "/tmp/logMap_test3" );

    MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> writer;
    writer.logPath( "/tmp/logMap_test3" );
    writer.logName( "dev1" );
    writer.logExt( "xlog" );
    writer.maxLogSize( 1000000 ); // This is large enough that all 5 entries land in one file.

    // Write five entries ten seconds apart, with event codes A, B, A, A, and B.
    const time_t         base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day
    flatlogs::bufferPtrT buf;
    flatlogs::logHeader::createLog<dummyLogA>(
        buf, flatlogs::timespecX( base, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogB>(
        buf, flatlogs::timespecX( base + 10, 0 ), "B", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogA>(
        buf, flatlogs::timespecX( base + 20, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogA>(
        buf, flatlogs::timespecX( base + 30, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogB>(
        buf, flatlogs::timespecX( base + 40, 0 ), "B", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    writer.close();

    std::string fileName, relPath;
    MagAOX::file::fileTimeRelPath( fileName, relPath, "dev1", "xlog", base, 0 );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;

    MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( "/tmp/logMap_test3/" + relPath + '/' + fileName );
    REQUIRE( sfn.valid() );
    lm.m_appToFileMap["dev1"].insert( sfn );

    SECTION( "no entry at all for the app returns an error" )
    {
        char *logBefore = nullptr; // The output is unused on error, but the call needs an lvalue.

        int rv = lm.getPriorLog( logBefore, "dev2", dummyLogA::eventCode, flatlogs::timespecX( base + 25, 0 ) );

        REQUIRE( rv == -1 );
    }

    SECTION( "finds the last log at or before the timestamp, with no hint" )
    {
        char *logBefore = nullptr;

        int rv = lm.getPriorLog(
            logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 25, 0 ) );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::eventCode( logBefore ) == dummyLogA::eventCode );
        REQUIRE( flatlogs::logHeader::timespec( logBefore ).time_s == base + 20 );
    }

    SECTION( "a hint at or before the timestamp is used directly" )
    {
        // First load the buffer and get real pointers into it by walking from the start.
        // This is exactly what getPriorLog and getNextLog do internally. The merged
        // getPriorLog ignores the hint argument. These sections now document that the
        // result does not depend on the hint. The time base+5 is used because the merged
        // loadFiles only loads when a file starts strictly before the requested time.
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode,
                         flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        char *p0 = lm.m_appToBufferMap["dev1"].m_memory.data(); // Entry 0 is A at base+0.
        char *p1 = p0 + flatlogs::logHeader::totalSize( p0 );   // Entry 1 is B at base+10.

        char *logBefore2 = nullptr;
        int   rv = lm.getPriorLog( logBefore2, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 25, 0 ), p1 );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::timespec( logBefore2 ).time_s == base + 20 );
    }

    SECTION( "a hint after the timestamp falls back to the start of the buffer" )
    {
        // The time base+5 is used because the merged loadFiles only loads when a file
        // starts strictly before the requested time.
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode,
                         flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        char *p0 = lm.m_appToBufferMap["dev1"].m_memory.data(); // Entry 0 is A at base+0.
        char *p1 = p0 + flatlogs::logHeader::totalSize( p0 );   // Entry 1 is B at base+10.
        char *p2 = p1 + flatlogs::logHeader::totalSize( p1 );   // Entry 2 is A at base+20.
        char *p3 = p2 + flatlogs::logHeader::totalSize( p2 );   // Entry 3 is A at base+30.

        char *logBefore2 = nullptr;
        int   rv = lm.getPriorLog( logBefore2, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 25, 0 ), p3 );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::timespec( logBefore2 ).time_s == base + 20 );
    }

    SECTION( "an event code that never appears is reported as not found" )
    {
        char *logBefore = nullptr;

        int rv = lm.getPriorLog( logBefore, "dev1", 99, flatlogs::timespecX( base + 25, 0 ) );

        REQUIRE( rv == -1 );
    }

    SECTION( "reaches the end of the buffer scanning past the last matching-code entry" )
    {
        // Request the event code of the last entry, which is B at base+40, with a
        // timestamp beyond it. The merged getPriorLog scans the whole buffer and returns
        // the last match. The old implementation returned 1 here, meaning that more data
        // needed to be loaded.
        char *logBefore = nullptr;

        int rv = lm.getPriorLog( logBefore, "dev1", dummyLogB::eventCode, flatlogs::timespecX( base + 100, 0 ) );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::eventCode( logBefore ) == dummyLogB::eventCode );
        REQUIRE( flatlogs::logHeader::timespec( logBefore ).time_s == base + 40 );
    }

    SECTION( "reaches the end of the buffer while skipping a mismatched last entry" )
    {
        // Request A with a timestamp beyond everything. The scan passes the mismatched
        // last entry, which is B, and returns the last A. The old implementation returned
        // 1 here, meaning that more data needed to be loaded.
        char *logBefore = nullptr;

        int rv = lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 100, 0 ) );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::timespec( logBefore ).time_s == base + 30 );
    }

    SECTION( "detects a corrupt length field that would overshoot the buffer (requesting the corrupted entry's own code)" )
    {
        char *logBefore = nullptr;

        // Populate the buffer first with a normal, in-range call. The time base+5 is used
        // because the merged loadFiles only loads when a file starts strictly before the
        // requested time.
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode,
                         flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        // Walk to the last entry, which is B at base+40, and inflate its declared message
        // length so that totalSize() overshoots the real end of m_memory.
        std::vector<char> &mem = lm.m_appToBufferMap["dev1"].m_memory;
        char              *p   = mem.data();
        for( int i = 0; i < 4; ++i )
        {
            p += flatlogs::logHeader::totalSize( p );
        }
        p[flatlogs::logHeader::headerSize( p ) - 1] += 50;

        // Request B, the code of the corrupted entry, with a timestamp beyond it. The
        // merged scan detects the bad extent. It cannot resync because nothing valid
        // follows. It returns the last good B. The old implementation returned -1 here.
        int rv = lm.getPriorLog( logBefore, "dev1", dummyLogB::eventCode, flatlogs::timespecX( base + 100, 0 ) );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::timespec( logBefore ).time_s == base + 10 );
    }

    SECTION( "detects a corrupt length field that would overshoot the buffer (requesting a different code)" )
    {
        char *logBefore = nullptr;

        // The time base+5 is used because the merged loadFiles only loads when a file
        // starts strictly before the requested time.
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode,
                         flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        std::vector<char> &mem = lm.m_appToBufferMap["dev1"].m_memory;
        char              *p   = mem.data();
        for( int i = 0; i < 4; ++i )
        {
            p += flatlogs::logHeader::totalSize( p );
        }
        p[flatlogs::logHeader::headerSize( p ) - 1] += 50;

        // Request A with a timestamp beyond everything. The merged scan stops at the
        // corrupted entry and returns the last good A. The old implementation returned -1
        // here.
        int rv = lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 100, 0 ) );

        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::timespec( logBefore ).time_s == base + 30 );
    }
}

/// The empty memory guard in getPriorLog. It runs after a loadFiles() call that returns 0
/// without ever populating m_memory. loadFiles() does not propagate the per-file return
/// value of loadFile(). So this can only be reached by a real file selected by
/// loadFiles() that exists on disk as a valid stdFileName but has content that loadFile()
/// itself rejects. An empty file does that.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "getPriorLog reports an error when loadFiles succeeds but leaves memory empty",
           "[libMagAOX::logger::logMap]" )
{
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    std::filesystem::remove_all( "/tmp/logMap_test_emptyload" );
    std::string fileName, relPath;
    MagAOX::file::fileTimeRelPath( fileName, relPath, "devEmptyLoad", "xlog", base, 0 );
    std::filesystem::create_directories( "/tmp/logMap_test_emptyload/" + relPath );
    std::ofstream( "/tmp/logMap_test_emptyload/" + relPath + '/' + fileName ).close(); // loadFile() rejects an empty file.

    MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( "/tmp/logMap_test_emptyload/" + relPath + '/' + fileName );
    REQUIRE( sfn.valid() );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap["devEmptyLoad"].insert( sfn );

    char *logBefore = nullptr;
    int   rv = lm.getPriorLog(
        logBefore, "devEmptyLoad", dummyLogA::eventCode, flatlogs::timespecX( base + 100, 0 ) );

    REQUIRE( rv == -1 );
    REQUIRE( lm.m_appToBufferMap["devEmptyLoad"].m_memory.size() == 0 );
}

/// getNextLog steps forward to the next log with a matching event code. Pointers into the
/// loaded buffer are obtained by walking entry sizes from the start.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "getNextLog steps forward to the next log with the same event code", "[libMagAOX::logger::logMap]" )
{
    std::filesystem::remove_all( "/tmp/logMap_test4" );

    MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> writer;
    writer.logPath( "/tmp/logMap_test4" );
    writer.logName( "dev1" );
    writer.logExt( "xlog" );
    writer.maxLogSize( 1000000 );

    const time_t         base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day
    flatlogs::bufferPtrT buf;
    flatlogs::logHeader::createLog<dummyLogA>(
        buf, flatlogs::timespecX( base, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogB>(
        buf, flatlogs::timespecX( base + 10, 0 ), "B", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogA>(
        buf, flatlogs::timespecX( base + 20, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogA>(
        buf, flatlogs::timespecX( base + 30, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    flatlogs::logHeader::createLog<dummyLogB>(
        buf, flatlogs::timespecX( base + 40, 0 ), "B", flatlogs::logPrio::LOG_NOTICE );
    writer.writeLog( buf );
    writer.close();

    std::string fileName, relPath;
    MagAOX::file::fileTimeRelPath( fileName, relPath, "dev1", "xlog", base, 0 );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;

    MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( "/tmp/logMap_test4/" + relPath + '/' + fileName );
    lm.m_appToFileMap["dev1"].insert( sfn );

    // Load the buffer through getPriorLog as above, then walk to real pointers for each
    // entry. The time base+5 is used because the merged loadFiles only loads when a file
    // starts strictly before the requested time.
    char *logBefore = nullptr;
    REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode,
                         flatlogs::timespecX( base + 5, 0 ) ) == 0 );

    char *p0 = lm.m_appToBufferMap["dev1"].m_memory.data(); // Entry 0 is A at base+0.
    char *p1 = p0 + flatlogs::logHeader::totalSize( p0 );   // Entry 1 is B at base+10.
    char *p2 = p1 + flatlogs::logHeader::totalSize( p1 );   // Entry 2 is A at base+20.
    char *p3 = p2 + flatlogs::logHeader::totalSize( p2 );   // Entry 3 is A at base+30.
    char *p4 = p3 + flatlogs::logHeader::totalSize( p3 );   // Entry 4 is B at base+40.

    SECTION( "finds the very next entry when it already matches" )
    {
        char *logAfter = nullptr;
        int   rv       = lm.getNextLog( logAfter, p2, "dev1" ); // p2 is A, and the next entry p3 is also A.

        REQUIRE( rv == 0 );
        REQUIRE( logAfter == p3 );
    }

    SECTION( "skips entries with a different event code to find the next match" )
    {
        char *logAfter = nullptr;
        int   rv       = lm.getNextLog( logAfter, p1, "dev1" ); // p1 is B. The next B is p4, skipping p2 and p3, which are both A.

        REQUIRE( rv == 0 );
        REQUIRE( logAfter == p4 );
    }

    SECTION( "reaches the end of the buffer immediately after the last entry" )
    {
        char *logAfter = nullptr;
        int   rv       = lm.getNextLog( logAfter, p4, "dev1" ); // p4 is the last entry.

        REQUIRE( rv == 1 );
    }

    SECTION( "reaches the end of the buffer while skipping mismatched entries" )
    {
        char *logAfter = nullptr;
        int   rv       = lm.getNextLog( logAfter, p3, "dev1" ); // p3 is A. Only p4 follows, which is B, and then the buffer ends.

        REQUIRE( rv == 1 );
    }
}

/// loadFiles selects which on-disk files to bring into memory. Four files are written two
/// hours apart with one entry each, so each call can pick a subset.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "loadFiles selects on-disk files to bring into memory", "[libMagAOX::logger::logMap]" )
{
    std::filesystem::remove_all( "/tmp/logMap_test5" );

    MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> writer;
    writer.logPath( "/tmp/logMap_test5" );
    writer.logName( "dev1" );
    writer.logExt( "xlog" );
    writer.maxLogSize( 1 ); // This forces every entry into its own file.

    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day
    const time_t times[4]{ base, base + 7200, base + 14400, base + 21600 };

    for( auto t : times )
    {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<dummyLogA>( buf, flatlogs::timespecX( t, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
        writer.writeLog( buf );
    }
    writer.close();

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;

    for( auto t : times )
    {
        std::string fileName, relPath;
        MagAOX::file::fileTimeRelPath( fileName, relPath, "dev1", "xlog", t, 0 );

        MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( "/tmp/logMap_test5/" + relPath + '/' + fileName );
        REQUIRE( sfn.valid() );
        lm.m_appToFileMap["dev1"].insert( sfn );
    }

    SECTION( "no files for the app is an error" )
    {
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> emptyLm;

        int rv = emptyLm.loadFiles( "dev1", flatlogs::timespecX( base, 0 ) );

        REQUIRE( rv == -1 );
    }

    SECTION( "the first load picks the file before the target time and the one after it" )
    {
        int rv = lm.loadFiles( "dev1", flatlogs::timespecX( times[1] + 1800, 0 ) );

        REQUIRE( rv == 0 );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_memory.size() > 0 );
        REQUIRE( lm.m_appToBufferMap["dev1"].m_startTime.time_s == times[1] );
    }

    SECTION( "a second call within the already-loaded range returns immediately" )
    {
        REQUIRE( lm.loadFiles( "dev1", flatlogs::timespecX( times[1] + 1800, 0 ) ) == 0 );

        int rv = lm.loadFiles( "dev1", flatlogs::timespecX( times[1] + 1800, 0 ) );

        REQUIRE( rv == 0 );
    }

    SECTION( "loading an earlier time extends the buffer backward" )
    {
        REQUIRE( lm.loadFiles( "dev1", flatlogs::timespecX( times[2] + 1800, 0 ) ) == 0 );

        int rv = lm.loadFiles( "dev1", flatlogs::timespecX( times[0], 0 ) );

        REQUIRE( rv == 0 );
    }

    SECTION( "loading a later time extends the buffer forward" )
    {
        REQUIRE( lm.loadFiles( "dev1", flatlogs::timespecX( times[0] + 1800, 0 ) ) == 0 );

        int rv = lm.loadFiles( "dev1", flatlogs::timespecX( times[3], 0 ) );

        REQUIRE( rv == 0 );
    }

    SECTION( "extending backward past every known file stops at the oldest one" )
    {
        // Fake an already-loaded buffer whose m_startTime is after every file in the map.
        // The backward search walks forward looking for the first file at or after
        // m_startTime. So it runs off the end of the map instead of finding one.
        lm.m_appToBufferMap["dev1"].m_memory.assign( 1, 0 );
        lm.m_appToBufferMap["dev1"].m_startTime = flatlogs::timespecX( times[3] + 100, 0 );
        lm.m_appToBufferMap["dev1"].m_endTime   = flatlogs::timespecX( times[3] + 100, 0 );

        int rv = lm.loadFiles( "dev1", flatlogs::timespecX( times[0], 0 ) );

        REQUIRE( rv == 0 );
    }

    SECTION( "extending forward past every known file stops at the newest one" )
    {
        // Fake an already-loaded buffer whose m_endTime is before every file in the map.
        // The first search walks backward looking for a file at or before m_endTime, so
        // it runs off the beginning of the map. The second search walks forward looking
        // for a file at or after the requested time, so it runs off the end of the map.
        lm.m_appToBufferMap["dev1"].m_memory.assign( 1, 0 );
        lm.m_appToBufferMap["dev1"].m_startTime = flatlogs::timespecX( times[0] - 100, 0 );
        lm.m_appToBufferMap["dev1"].m_endTime   = flatlogs::timespecX( times[0] - 100, 0 );

        int rv = lm.loadFiles( "dev1", flatlogs::timespecX( times[3] + 100, 0 ) );

        REQUIRE( rv == 0 );
    }
}

/// The defensive checks inside logInMemory::loadFile, called directly with no logMap
/// involved.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "logInMemory::loadFile detects filesystem and framing errors", "[libMagAOX::logger::logMap]" )
{
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    SECTION( "a short read (fewer bytes than the file's reported size) is reported as an error" )
    {
        // A genuine short read from a regular file, where the stat size differs from the
        // bytes actually read, is not reproducible without a concurrent truncation race.
        // So this uses the namespaced build of logInMemory that forces nrd to 0 after the
        // real read() call.
        auto sfn = writeSingleLogFile( "/tmp/logMap_test_loadfile_shortread", "dev1", { base } );

        MagAOX::logger::XWCTEST_LOGMAP_LOADFILE_SHORTREAD_ns::logInMemory lim;
        int                                                               rv = lim.loadFile( sfn );

        REQUIRE( rv == -1 );
    }

    SECTION( "a truncated trailing entry after otherwise-valid entries is reported as corrupt" )
    {
        auto sfn = writeSingleLogFile( "/tmp/logMap_test_loadfile_corrupt", "dev1", { base, base + 10 } );

        // Append a header claiming a 5000 byte message, but write only the header itself.
        // The file ends long before the message it claims to contain. So walking the
        // buffer by declared entry sizes overshoots the actual file size.
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<dummyLogBigDeclared>(
            buf, flatlogs::timespecX( base + 20, 0 ), 0, flatlogs::logPrio::LOG_NOTICE );
        size_t hsz = flatlogs::logHeader::headerSize( buf.get() );

        std::ofstream fout( sfn.fullName(), std::ios::binary | std::ios::app );
        fout.write( buf.get(), hsz );
        fout.close();

        // The merged loadFile tolerates a truncated trailing entry. It keeps the valid
        // prefix and reports success. The old implementation rejected the whole file with
        // -1 here.
        MagAOX::logger::logInMemory lim;
        int                         rv = lim.loadFile( sfn );

        REQUIRE( rv == 0 );
        REQUIRE( lim.m_endTime.time_s == base + 10 ); // The truncated base+20 entry is dropped.
    }

    SECTION( "a file overlapping the start of an already-loaded buffer is rejected" )
    {
        auto sfnA = writeSingleLogFile( "/tmp/logMap_test_loadfile_overlapA", "dev1", { base, base + 10, base + 20 } );
        auto sfnB = writeSingleLogFile( "/tmp/logMap_test_loadfile_overlapB", "dev2", { base - 10, base + 5 } );

        MagAOX::logger::logInMemory lim;
        REQUIRE( lim.loadFile( sfnA ) == 0 );
        REQUIRE( lim.m_startTime == flatlogs::timespecX( base, 0 ) );

        // sfnB starts at base-10, which is before the start of lim at base. It ends at
        // base+5, which is at or after that start.
        int rv = lim.loadFile( sfnB );

        REQUIRE( rv == -1 );
    }

    SECTION( "a file whose range falls inside an already-loaded buffer hits the fallback branch" )
    {
        auto sfnA = writeSingleLogFile( "/tmp/logMap_test_loadfile_middleA", "dev1", { base, base + 20 } );
        auto sfnB = writeSingleLogFile( "/tmp/logMap_test_loadfile_middleB", "dev2", { base + 5, base + 15 } );

        MagAOX::logger::logInMemory lim;
        REQUIRE( lim.loadFile( sfnA ) == 0 );
        REQUIRE( lim.m_startTime == flatlogs::timespecX( base, 0 ) );
        REQUIRE( lim.m_endTime == flatlogs::timespecX( base + 20, 0 ) );

        // The start of sfnB at base+5 is neither before the start of lim nor after the end
        // of lim.
        int rv = lim.loadFile( sfnB );

        REQUIRE( rv == -1 );
    }
}


/// The free helper functions that the scan and resync machinery is built from, called
/// directly.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "logMap free helpers: debug formatting, extent validation, resync", "[libMagAOX::logger::logMap]" )
{
    SECTION( "logMapDebugTime formats a timestamp (normally only used by DEBUG_CRUMB)" )
    {
        std::string out = MagAOX::logger::logMapDebugTime( flatlogs::timespecX( 1732176000, 5 ) ); // arbitrary time and 5 ns
        REQUIRE( out.find( "1732176000.5" ) != std::string::npos );
    }

    SECTION( "logMapEntryExtentValid rejects null/too-short/lying-header buffers" )
    {
        size_t totalSize = 99;
        REQUIRE( !MagAOX::logger::logMapEntryExtentValid( totalSize, nullptr, nullptr ) );
        REQUIRE( totalSize == 0 );

        // A real entry, but viewed through a window shorter than minHeadSize.
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<dummyLogA>(
            buf, flatlogs::timespecX( 1732176000, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( !MagAOX::logger::logMapEntryExtentValid( totalSize, buf.get(), buf.get() + 1 ) );

        // An entry whose extended header itself is longer than the window. The msgLen of
        // the big-declared type needs the extended header form.
        flatlogs::bufferPtrT big;
        flatlogs::logHeader::createLog<dummyLogBigDeclared>(
            big, flatlogs::timespecX( 1732176000, 0 ), 0, flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( flatlogs::logHeader::headerSize( big.get() ) >
                 static_cast<size_t>( flatlogs::logHeader::minHeadSize ) );
        REQUIRE( !MagAOX::logger::logMapEntryExtentValid(
            totalSize, big.get(), big.get() + flatlogs::logHeader::minHeadSize ) );
    }

    SECTION( "logMapResync rejects null buffers and finds a valid chain past garbage" )
    {
        REQUIRE( MagAOX::logger::logMapResync( nullptr, nullptr ) == nullptr );

        // The buffer is 8 garbage bytes followed by 3 valid consecutive entries. A resync
        // from the garbage must land exactly on the first valid entry, which starts a
        // 3-link chain.
        std::vector<char>    mem( 8, '\xff' );
        flatlogs::bufferPtrT buf;
        for( int i = 0; i < 3; ++i )
        {
            flatlogs::logHeader::createLog<dummyLogA>(
                buf, flatlogs::timespecX( 1732176000 + 10 * i, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
            mem.insert( mem.end(), buf.get(), buf.get() + flatlogs::logHeader::totalSize( buf.get() ) );
        }
        char *found = MagAOX::logger::logMapResync( mem.data(), mem.data() + mem.size() );
        REQUIRE( found == mem.data() + 8 );

        // Garbage followed by exactly two valid entries that reach the buffer end. That is
        // fewer than a full 3-link chain, but a chain that ends exactly at bufferEnd also
        // counts.
        std::vector<char> mem2( 8, '\xff' );
        for( int i = 0; i < 2; ++i )
        {
            flatlogs::logHeader::createLog<dummyLogA>(
                buf, flatlogs::timespecX( 1732176000 + 10 * i, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
            mem2.insert( mem2.end(), buf.get(), buf.get() + flatlogs::logHeader::totalSize( buf.get() ) );
        }
        REQUIRE( MagAOX::logger::logMapResync( mem2.data(), mem2.data() + mem2.size() ) == mem2.data() + 8 );

        // An all garbage buffer has no chain to find.
        std::vector<char> junk( 64, '\xff' );
        REQUIRE( MagAOX::logger::logMapResync( junk.data(), junk.data() + junk.size() ) == nullptr );
    }

    SECTION( "recordRecoverableError / recoverableErrors accumulate across apps" )
    {
        MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
        REQUIRE( lm.recoverableErrors() == 0 );
        lm.recordRecoverableError( "devA" );
        lm.recordRecoverableError( "devA" );
        lm.recordRecoverableError( "devB" );
        REQUIRE( lm.recoverableErrors() == 3 );
    }
}

/// loadFile tolerates corruption in the middle of a file, and the source attribution
/// accessors map entries back to the file they came from.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "logInMemory corruption recovery and source attribution", "[libMagAOX::logger::logMap]" )
{
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    SECTION( "an empty file is rejected" )
    {
        std::filesystem::remove_all( "/tmp/logMap_test_empty" );
        std::string fileName, relPath;
        MagAOX::file::fileTimeRelPath( fileName, relPath, "dev1", "xlog", base, 0 );
        std::filesystem::create_directories( "/tmp/logMap_test_empty/" + relPath );
        std::ofstream( "/tmp/logMap_test_empty/" + relPath + '/' + fileName ).close();

        MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( "/tmp/logMap_test_empty/" + relPath + '/' + fileName );
        REQUIRE( sfn.valid() );

        MagAOX::logger::logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == -1 );
    }

    SECTION( "garbage in the middle of a file is skipped by a real resync" )
    {
        auto sfn = writeSingleLogFile( "/tmp/logMap_test_midcorrupt", "dev1", { base, base + 10 } );

        // Append garbage, then three more valid entries, so the resync scan has a
        // 3-link chain to land on.
        std::ofstream fout( sfn.fullName(), std::ios::binary | std::ios::app );
        std::vector<char> junk( 16, '\xff' );
        fout.write( junk.data(), junk.size() );
        flatlogs::bufferPtrT buf;
        for( int i = 2; i < 5; ++i )
        {
            flatlogs::logHeader::createLog<dummyLogA>(
                buf, flatlogs::timespecX( base + 10 * i, 0 ), "A", flatlogs::logPrio::LOG_NOTICE );
            fout.write( buf.get(), flatlogs::logHeader::totalSize( buf.get() ) );
        }
        fout.close();

        MagAOX::logger::logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == 0 );
        REQUIRE( lim.m_recoverableErrors > 0 );
        REQUIRE( lim.m_endTime.time_s == base + 40 ); // The entries after the garbage were kept.
    }

    SECTION( "a file of pure garbage is rejected" )
    {
        std::filesystem::remove_all( "/tmp/logMap_test_garbage" );
        std::string fileName, relPath;
        MagAOX::file::fileTimeRelPath( fileName, relPath, "dev1", "xlog", base, 0 );
        std::filesystem::create_directories( "/tmp/logMap_test_garbage/" + relPath );
        {
            std::ofstream fout( "/tmp/logMap_test_garbage/" + relPath + '/' + fileName, std::ios::binary );
            std::vector<char> junk( 64, '\xff' );
            fout.write( junk.data(), junk.size() );
        }

        MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> sfn( "/tmp/logMap_test_garbage/" + relPath + '/' + fileName );
        REQUIRE( sfn.valid() );

        MagAOX::logger::logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == -1 );
    }

    SECTION( "sourceFile and sourceOffset attribute entries to their on-disk file" )
    {
        auto sfn = writeSingleLogFile( "/tmp/logMap_test_source", "dev1", { base, base + 10 } );

        MagAOX::logger::logInMemory lim;
        REQUIRE( lim.loadFile( sfn ) == 0 );

        char *p0 = lim.m_memory.data();
        char *p1 = p0 + flatlogs::logHeader::totalSize( p0 );

        REQUIRE( lim.sourceFile( p0 ) == sfn.fullName() );
        REQUIRE( lim.sourceFile( p1 ) == sfn.fullName() );
        REQUIRE( lim.sourceOffset( p0 ) == 0 );
        REQUIRE( lim.sourceOffset( p1 ) == static_cast<size_t>( p1 - p0 ) );

        // Check the error forms for a null pointer, an empty buffer, and an out-of-range
        // pointer.
        REQUIRE( lim.sourceFile( nullptr ) == "<unknown>" );
        REQUIRE( lim.sourceOffset( nullptr ) == std::numeric_limits<size_t>::max() );

        char *past = lim.m_memory.data() + lim.m_memory.size();
        REQUIRE( lim.sourceFile( past ) == "<outside-loaded-buffer>" );
        REQUIRE( lim.sourceOffset( past ) == std::numeric_limits<size_t>::max() );

        MagAOX::logger::logInMemory empty;
        REQUIRE( empty.sourceFile( p0 ) == "<unknown>" );
        REQUIRE( empty.sourceOffset( p0 ) == std::numeric_limits<size_t>::max() );

        // A buffer with bytes but no recorded loaded-file ranges. This can only be built
        // by hand, because loadFile always records contiguous ranges. It exercises the
        // returns for a pointer that is in range but unknown.
        MagAOX::logger::logInMemory bare;
        bare.m_memory.resize( 32, 0 );
        REQUIRE( bare.sourceFile( bare.m_memory.data() ) == "<unknown-loaded-file>" );
        REQUIRE( bare.sourceOffset( bare.m_memory.data() ) == std::numeric_limits<size_t>::max() );
    }
}

/// The guard and recovery paths of getNextLog and getPriorLog over real buffers, some of
/// which are corrupted in place.
/**
 * \ingroup logMap_unit_test
 */
TEST_CASE( "getNextLog and getPriorLog guard and resync paths", "[libMagAOX::logger::logMap]" )
{
    const time_t base = 1732176000; // 2024-11-21 08:00:00 UTC, an arbitrary fixed time on the fixture day

    auto sfn = writeSingleLogFile( "/tmp/logMap_test_guards", "dev1", { base, base + 10, base + 20, base + 30, base + 40 } );

    MagAOX::logger::logMap<XWC_DEFAULT_VERBOSITY> lm;
    lm.m_appToFileMap["dev1"].insert( sfn );


    SECTION( "a lastFile whose day-directory vanished falls back to the previous boundary" )
    {
        createTestPaths( "/tmp/logMap_test" );

        // lastFile claims a day far past every real directory. The forward scan finds
        // nothing within m_searchDaySpan days. The fallback walk back from that day also
        // finds nothing within the span. So follLogSubDir falls back to prevLogSubDir and
        // the search proceeds from there.
        MagAOX::file::stdFileName firstFile( "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName lastFile( "cam1/2025_06_01/cam1_20250601000000000000000.xrif" );

        MagAOX::logger::logMap lm;
        lm.m_searchDaySpan = 5;

        mx::error_t errc = lm.loadAppToFileMap( "/tmp/logMap_test", "dev1", ".xlog", firstFile, lastFile );
        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( lm.m_appToFileMap["dev1"].size() > 0 );
    }

    SECTION( "getPriorLog with no file strictly before the requested time fails to load" )
    {
        // The timestamp of the single file equals base, so there is no file strictly
        // before ts=base. The initial loadFiles finds no prior file and getPriorLog
        // reports it.
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base, 0 ) ) == -1 );
    }

    SECTION( "getNextLog rejects a null current entry and an unloaded buffer" )
    {
        char *logAfter = nullptr;
        REQUIRE( lm.getNextLog( logAfter, nullptr, "dev1" ) == -1 );
    }

    SECTION( "getNextLog rejects a pointer outside the loaded buffer" )
    {
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        char *past     = lm.m_appToBufferMap["dev1"].m_memory.data() + lm.m_appToBufferMap["dev1"].m_memory.size();
        char *logAfter = nullptr;
        REQUIRE( lm.getNextLog( logAfter, past, "dev1" ) == -1 );
    }

    SECTION( "getNextLog rejects a current entry whose header overshoots the buffer" )
    {
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        std::vector<char> &mem = lm.m_appToBufferMap["dev1"].m_memory;
        char              *p   = mem.data();
        for( int i = 0; i < 4; ++i )
        {
            p += flatlogs::logHeader::totalSize( p );
        }
        p[flatlogs::logHeader::headerSize( p ) - 1] += 50; // Inflate the declared length of the last entry.

        char *logAfter = nullptr;
        REQUIRE( lm.getNextLog( logAfter, p, "dev1" ) == -1 );
    }

    SECTION( "getNextLog resyncs past a corrupted middle entry" )
    {
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        std::vector<char> &mem = lm.m_appToBufferMap["dev1"].m_memory;
        char              *p0  = mem.data();
        char              *p1  = p0 + flatlogs::logHeader::totalSize( p0 );

        // Corrupt the priority byte of entry 1, which is the first field of the header.
        // The value 30 is not a valid on-disk priority. Valid values are 0 to 8 and 64.
        // This makes the entry insane without changing any size fields. So entries 2 to 4
        // still chain and the resync can land on entry 2.
        char *p2 = p1 + flatlogs::logHeader::totalSize( p1 );
        p1[0]    = 30;

        char *logAfter = nullptr;
        int   rv       = lm.getNextLog( logAfter, p0, "dev1" );
        REQUIRE( rv == 0 );
        REQUIRE( logAfter == p2 ); // The scan resynced past the corrupt entry to the next A.
    }

    SECTION( "getPriorLog resyncs past a corrupted middle entry during its scan" )
    {
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        std::vector<char> &mem = lm.m_appToBufferMap["dev1"].m_memory;
        char              *p0  = mem.data();
        char              *p1  = p0 + flatlogs::logHeader::totalSize( p0 );
        p1[0]                  = 30; // An invalid priority makes entry 1 insane while its sizes stay intact.

        char *logBefore2 = nullptr;
        int   rv = lm.getPriorLog( logBefore2, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 100, 0 ) );
        REQUIRE( rv == 0 );
        REQUIRE( flatlogs::logHeader::timespec( logBefore2 ).time_s == base + 40 );
    }

    SECTION( "getNextLog gives up when a corrupt tail cannot be resynced" )
    {
        char *logBefore = nullptr;
        REQUIRE( lm.getPriorLog( logBefore, "dev1", dummyLogA::eventCode, flatlogs::timespecX( base + 5, 0 ) ) == 0 );

        std::vector<char> &mem = lm.m_appToBufferMap["dev1"].m_memory;
        char              *p   = mem.data();
        for( int i = 0; i < 3; ++i )
        {
            p += flatlogs::logHeader::totalSize( p );
        }
        char *p4 = p + flatlogs::logHeader::totalSize( p );
        p4[0]    = 30; // Corrupt the last entry. Nothing valid follows, so the resync must fail.

        char *logAfter = nullptr;
        REQUIRE( lm.getNextLog( logAfter, p, "dev1" ) == 1 ); // The scan ends without a match.
    }
}

} // namespace logMapTest
} // namespace loggerTest
} // namespace libXWCTest
