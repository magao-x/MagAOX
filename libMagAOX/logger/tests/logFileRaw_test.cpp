/** \file logFileRaw_test.hpp
 * \brief Tests for the logFileRaw class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logFileRaw.hpp"

#include <sstream>

namespace libXWCTest
{

/** \defgroup logger_unit_test libXWC::logger Unit Tests
 * \ingroup unit_test
 */

/// Namespace for XWC::logger tests
/** \ingroup logger_unit_test
 *
 */
namespace loggerTest
{

/** \defgroup logFileRaw_unit_test logFileRaw Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger::logFileRaw tests
/** \ingroup logFileRaw_unit_test
 *
 */
namespace logFileRawTest
{

class logFileRawTest : public MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY>
{
  public:
    std::string testPath;

    logFileRawTest()
    {
        m_logPath = "/tmp";

        testPath = m_logPath + '/' + m_logName;
    }

    explicit logFileRawTest( const std::string &lp )
    {
        m_logPath = lp;

        testPath = m_logPath + '/' + m_logName;
    }

    mx::error_t test_createFile( flatlogs::timespecX &ts )
    {
        return createFile( ts );
    }

    uint64_t test_currFileStartSec()
    {
        return m_currFileStartSec;
    }

    bool test_isOpen()
    {
        return ( m_fout != nullptr );
    }
};

/// Construction of logFileRaw
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Construction of logFileRaw", "[libMagAOX::logger::logFileRaw]" )
{
    SECTION( "basic construction and member access" )
    {
        MagAOX::logger::logFileRaw lfr;

        REQUIRE( lfr.logPath() == "." );
        REQUIRE( lfr.logName() == "xlog" );
        REQUIRE( lfr.logExt() == MAGAOX_default_logExt );
        REQUIRE( lfr.maxLogSize() == MAGAOX_default_max_logSize );
        REQUIRE( lfr.maxLogTime() == MAGAOX_default_maxLogTime );

        lfr.logPath( "/newp/test/x" );
        REQUIRE( lfr.logPath() == "/newp/test/x" );

        lfr.logName( "newdev" );
        REQUIRE( lfr.logName() == "newdev" );

        lfr.logExt( "bintel" );
        REQUIRE( lfr.logExt() == "bintel" );

        lfr.maxLogSize( 10 );
        REQUIRE( lfr.maxLogSize() == 10 );

        lfr.maxLogTime( 15 );
        REQUIRE( lfr.maxLogTime() == 15 );

        lfr.maxLogTime( 0 );
        REQUIRE( lfr.maxLogTime() == 0 );
    }
}

/// Creating a log file
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Creating a log file", "[libMagAOX::logger::logFileRaw]" )
{
    // clang-format off
    #ifdef XWCTEST_DOXYGEN_REF_PROTECTED
        logFileRaw          lfr;
        flatlogs::timespecX ts1( 1732170780, 1 );
        lfr.createFile( ts1 );
        lfr.logName();
        lfr.logExt();
        lfr.m_logPath;
        lfr.m_logName;
    #endif
    // clang-format on

    SECTION( "Two valid timestamps" )
    {
        logFileRawTest lfr;

        // safety check to make sure we don't delete all of /tmp
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( lfr.testPath );

        flatlogs::timespecX ts1( 1732170780, 1 );

        mx::error_t rv = lfr.test_createFile( ts1 );

        REQUIRE( rv == mx::error_t::noerror );
        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000001." +
                                          lfr.logExt() ) );

        flatlogs::timespecX ts2( 1763706780, 50 );

        rv = lfr.test_createFile( ts2 );

        REQUIRE( rv == mx::error_t::noerror );
        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2025_11_21/" ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2025_11_21/" + lfr.logName() + "_20251121063300000000050." +
                                          lfr.logExt() ) );
    }

    SECTION( "logPath without permissions" )
    {
        // check that this path doesn't already exist
        if( std::filesystem::exists( "/lfrtest" ) )
        {
            std::cerr << "\nTESTING-ERROR: path /lsfrtest exists so permission test will be invalid.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        logFileRawTest lfr( "/lfrtest/" ); // not just root as a just in case

        flatlogs::timespecX ts1( 1732170780, 1 );

        mx::error_t errc = lfr.test_createFile( ts1 );


        REQUIRE( errc != mx::error_t::noerror );
    }

    SECTION( "2nd timestamp is the same as the first, file already exists" )
    {
        logFileRawTest lfr;

        // safety check to make sure we don't delete all of /tmp
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( lfr.testPath );

        flatlogs::timespecX ts1( 1732170780, 1 );

        mx::error_t rv = lfr.test_createFile( ts1 );

        REQUIRE( rv == mx::error_t::noerror );
        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000001." +
                                          lfr.logExt() ) );

        flatlogs::timespecX ts2( 1732170780, 1 );

        rv = lfr.test_createFile( ts2 );

        REQUIRE( rv == mx::error_t::eexist );
    }
}

struct dummyLog
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = 1;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

    typedef std::string messageT;

    /// The message string
    static const char *msg()
    {
        return "LOOP CLOSED";
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

/// Writing to a log file
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Writing to a log file", "[libMagAOX::logger::logFileRaw]" )
{
    // clang-format off
    #ifdef XWCTEST_DOXYGEN_REF_PROTECTED
        logFileRaw          lfr;
        flatlogs::timespecX ts1( 1732170780, 1 );
        lfr.createFile( ts1 );
        flatlogs::bufferPtrT logbuff;
        lfr.writeLog( logbuff );
        lfr.logName();
        lfr.logExt();
        lfr.close()
        lfr.m_logPath;
        lfr.m_logName;
    #endif
    // clang-format on

    SECTION( "Write to existing log" )
    {
        logFileRawTest lfr;

        // safety check to make sure we don't delete all of /tmp
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( lfr.testPath );

        flatlogs::timespecX ts1( 1732170780, 1 );

        mx::error_t rv = lfr.test_createFile( ts1 );

        REQUIRE( rv == mx::error_t::noerror );
        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" ) );

        std::string fullPath = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000001.";
        fullPath += lfr.logExt();

        REQUIRE( std::filesystem::exists( fullPath ) );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts2( 1732170780, 2 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts2, msg, flatlogs::logPrio::LOG_NOTICE );

        rv = lfr.writeLog( logbuff );
        REQUIRE( rv == mx::error_t::noerror );

        REQUIRE( lfr.close() == mx::error_t::noerror );

        std::uintmax_t fsz = std::filesystem::file_size( fullPath );

        REQUIRE( fsz == 1 * ( 256 + 14 ) );
    }

    SECTION( "Write to log that doesn't exist yet" )
    {
        logFileRawTest lfr;

        // safety check to make sure we don't delete all of /tmp
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( lfr.testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts2( 1732170780, 2 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts2, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );
        REQUIRE( rv == mx::error_t::noerror );

        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" ) );

        std::string fullPath = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000002.";
        fullPath += lfr.logExt();

        REQUIRE( std::filesystem::exists( fullPath ) );

        REQUIRE( lfr.close() == mx::error_t::noerror );

        std::uintmax_t fsz = std::filesystem::file_size( fullPath );

        REQUIRE( fsz == 1 * ( 256 + 14 ) );
    }

    SECTION( "Write to log twice, does not exceed size" )
    {
        logFileRawTest lfr;

        // safety check to make sure we don't delete all of /tmp
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( lfr.testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts2( 1732170780, 2 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts2, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );
        REQUIRE( rv == mx::error_t::noerror );

        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" ) );

        std::string fullPath = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000002.";
        fullPath += lfr.logExt();

        REQUIRE( std::filesystem::exists( fullPath ) );

        flatlogs::timespecX  ts3( 1732170780, 50 );
        flatlogs::bufferPtrT logbuff3;
        flatlogs::logHeader::createLog<dummyLog>( logbuff3, ts3, msg, flatlogs::logPrio::LOG_NOTICE );

        rv = lfr.writeLog( logbuff3 );
        REQUIRE( rv == mx::error_t::noerror );

        // New file not created
        std::string fullPath2 = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000050.";
        fullPath2 += lfr.logExt();

        REQUIRE( !std::filesystem::exists( fullPath2 ) );

        lfr.close();

        std::uintmax_t fsz = std::filesystem::file_size( fullPath );

        REQUIRE( fsz == 2 * ( 256 + 14 ) ); // has two logs in it
    }

    SECTION( "Write to log twice, does exceed size" )
    {
        logFileRawTest lfr;
        lfr.maxLogSize( 256 );

        // safety check to make sure we don't delete all of /tmp
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return;
        }

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( lfr.testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 2 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );
        REQUIRE( rv == mx::error_t::noerror );

        REQUIRE( std::filesystem::exists( lfr.testPath ) );
        REQUIRE( std::filesystem::exists( lfr.testPath + "/2024_11_21/" ) );

        std::string fullPath = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000002.";
        fullPath += lfr.logExt();

        REQUIRE( std::filesystem::exists( fullPath ) );

        flatlogs::timespecX  ts2( 1732170780, 50 );
        flatlogs::bufferPtrT logbuff2;
        flatlogs::logHeader::createLog<dummyLog>( logbuff2, ts2, msg, flatlogs::logPrio::LOG_NOTICE );

        rv = lfr.writeLog( logbuff2 );
        REQUIRE( rv == mx::error_t::noerror );

        // New file created
        std::string fullPath2 = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000050.";
        fullPath2 += lfr.logExt();

        REQUIRE( std::filesystem::exists( fullPath2 ) );

        // Test this before closing, as this will probably only pass if the previous file was closed
        std::uintmax_t fsz = std::filesystem::file_size( fullPath );

        REQUIRE( fsz == 1 * ( 256 + 14 ) );

        lfr.close();

        fsz = std::filesystem::file_size( fullPath2 );

        REQUIRE( fsz == 1 * ( 256 + 14 ) );
    }
}


/// Time-based rotation of a log file
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Time based rotation of a log file", "[libMagAOX::logger::logFileRaw]" )
{
    // All timestamps below are chosen relative to a 10 minute (600 second) interval:
    //   1732170780 = 2024-11-21 06:33:00 UTC, bin 2886951
    //   1732171199 = 2024-11-21 06:39:59 UTC, bin 2886951 (same)
    //   1732171200 = 2024-11-21 06:40:00 UTC, bin 2886952 (next)
    //   1732171260 = 2024-11-21 06:41:00 UTC, bin 2886952 (same)

    std::string msg( 256, 't' );

    /// Write a log with the given timestamp, requiring success
    auto writeAt = []( logFileRawTest &lfr, flatlogs::timespecX ts, const std::string &m )
    {
        flatlogs::bufferPtrT logbuff;
        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, m, flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );
    };

    /// Path of the file that an entry at this timestamp would create
    auto pathFor = []( logFileRawTest &lfr, const std::string &stamp )
    { return lfr.testPath + "/2024_11_21/" + lfr.logName() + "_" + stamp + "." + lfr.logExt(); };

    /// Guard against deleting all of /tmp, and clear any previous run
    auto resetPath = []( logFileRawTest &lfr )
    {
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return false;
        }
        std::filesystem::remove_all( lfr.testPath );
        return true;
    };

    SECTION( "two entries in the same interval share a file" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );
        writeAt( lfr, flatlogs::timespecX( 1732171199, 3 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000002" ) ) );
        REQUIRE( !std::filesystem::exists( pathFor( lfr, "20241121063959000000003" ) ) );

        lfr.close();
        REQUIRE( std::filesystem::file_size( pathFor( lfr, "20241121063300000000002" ) ) == 2 * ( 256 + 14 ) );
    }

    SECTION( "an entry in the next interval starts a new file" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );
        writeAt( lfr, flatlogs::timespecX( 1732171200, 4 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000002" ) ) );
        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121064000000000004" ) ) );

        REQUIRE( std::filesystem::file_size( pathFor( lfr, "20241121063300000000002" ) ) == 1 * ( 256 + 14 ) );

        // The new file, not the old one, is now the current file
        REQUIRE( lfr.test_currFileStartSec() == 1732171200 );

        lfr.close();
        REQUIRE( std::filesystem::file_size( pathFor( lfr, "20241121064000000000004" ) ) == 1 * ( 256 + 14 ) );
    }

    SECTION( "a size triggered split does not shift the interval boundary" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        lfr.maxLogSize( 256 ); // force every write after the first to split
        if( !resetPath( lfr ) )
        {
            return;
        }

        // Two entries in the same interval, split by size rather than by time
        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );
        writeAt( lfr, flatlogs::timespecX( 1732171199, 3 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000002" ) ) );
        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063959000000003" ) ) );

        // The mid-interval file inherits the later start time, but is still in interval 2886951 ...
        REQUIRE( lfr.test_currFileStartSec() == 1732171199 );

        // ... so an entry in the next interval must still start a new file at the boundary.
        writeAt( lfr, flatlogs::timespecX( 1732171200, 4 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121064000000000004" ) ) );
        REQUIRE( lfr.test_currFileStartSec() == 1732171200 );

        lfr.close();
    }

    SECTION( "maxLogTime of 0 disables time based rotation" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 0 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        // These are in different 10 minute intervals, and even different days, but time rotation is off
        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );
        writeAt( lfr, flatlogs::timespecX( 1732171200, 4 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000002" ) ) );
        REQUIRE( !std::filesystem::exists( pathFor( lfr, "20241121064000000000004" ) ) );

        lfr.close();
        REQUIRE( std::filesystem::file_size( pathFor( lfr, "20241121063300000000002" ) ) == 2 * ( 256 + 14 ) );
    }

    SECTION( "a change to maxLogTime takes effect on the next entry" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 60 ); // both timestamps below are in the same 60 minute interval
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );

        // Shortening the interval puts the open file's start time in a different interval than the
        // next entry, so the next entry must rotate.
        lfr.maxLogTime( 10 );

        writeAt( lfr, flatlogs::timespecX( 1732171200, 4 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121064000000000004" ) ) );

        lfr.close();
    }
}

/// Rotation of a log file on request
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Requested rotation of a log file", "[libMagAOX::logger::logFileRaw]" )
{
    std::string msg( 256, 't' );

    auto writeAt = []( logFileRawTest &lfr, flatlogs::timespecX ts, const std::string &m )
    {
        flatlogs::bufferPtrT logbuff;
        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, m, flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );
    };

    auto pathFor = []( logFileRawTest &lfr, const std::string &stamp )
    { return lfr.testPath + "/2024_11_21/" + lfr.logName() + "_" + stamp + "." + lfr.logExt(); };

    auto resetPath = []( logFileRawTest &lfr )
    {
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return false;
        }
        std::filesystem::remove_all( lfr.testPath );
        return true;
    };

    SECTION( "a request forces a new file on the next entry" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 0 ); // isolate the request from time based rotation
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );

        lfr.requestRotation();

        writeAt( lfr, flatlogs::timespecX( 1732171199, 3 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000002" ) ) );
        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063959000000003" ) ) );

        // The request must be consumed, so a further entry does not rotate again
        writeAt( lfr, flatlogs::timespecX( 1732171200, 4 ), msg );

        REQUIRE( !std::filesystem::exists( pathFor( lfr, "20241121064000000000004" ) ) );

        lfr.close();
        REQUIRE( std::filesystem::file_size( pathFor( lfr, "20241121063959000000003" ) ) == 2 * ( 256 + 14 ) );
    }

    SECTION( "a request with no subsequent entry creates no file" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 0 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );

        lfr.requestRotation();

        lfr.close();

        // Only the one file from the single write exists
        int nfiles = 0;
        for( const auto &e : std::filesystem::directory_iterator( lfr.testPath + "/2024_11_21/" ) )
        {
            (void)e;
            ++nfiles;
        }

        REQUIRE( nfiles == 1 );
        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000002" ) ) );
    }

    SECTION( "a request is consumed even when another condition also triggers" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 2 ), msg );

        lfr.requestRotation();

        // This entry is in the next interval, so time rotation also fires.  The request must still be
        // cleared, otherwise it would cause a spurious rotation on the entry after this one.
        writeAt( lfr, flatlogs::timespecX( 1732171200, 4 ), msg );

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121064000000000004" ) ) );

        // Same interval as the previous entry, and no pending request, so no new file
        writeAt( lfr, flatlogs::timespecX( 1732171260, 5 ), msg );

        REQUIRE( !std::filesystem::exists( pathFor( lfr, "20241121064100000000005" ) ) );

        lfr.close();
        REQUIRE( std::filesystem::file_size( pathFor( lfr, "20241121064000000000004" ) ) == 2 * ( 256 + 14 ) );
    }
}


/// Closing a log file when its interval elapses with no entries
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Interval close of a log file", "[libMagAOX::logger::logFileRaw]" )
{
    // closeIfIntervalElapsed compares the open file's start time against the current clock, so these
    // sections drive it by choosing entry timestamps relative to now rather than by sleeping.

    std::string msg( 256, 't' );

    auto writeAt = []( logFileRawTest &lfr, flatlogs::timespecX ts, const std::string &m )
    {
        flatlogs::bufferPtrT logbuff;
        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, m, flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );
    };

    auto resetPath = []( logFileRawTest &lfr )
    {
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return false;
        }
        std::filesystem::remove_all( lfr.testPath );
        return true;
    };

    SECTION( "an elapsed interval closes the file without creating a new one" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 1 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        // An entry stamped well in the past, so its interval has certainly elapsed by now
        flatlogs::timespecX old( 1732170780, 2 ); // 2024-11-21
        writeAt( lfr, old, msg );

        REQUIRE( lfr.test_isOpen() );

        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );

        // File is closed, and nothing new was created for the intervening intervals
        REQUIRE( !lfr.test_isOpen() );

        int nfiles = 0;
        for( const auto &e : std::filesystem::recursive_directory_iterator( lfr.testPath ) )
        {
            if( std::filesystem::is_regular_file( e ) )
            {
                ++nfiles;
            }
        }

        REQUIRE( nfiles == 1 );
    }

    SECTION( "a current interval leaves the file open" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 1440 ); // a day, so an entry stamped now is certainly in the current interval
        if( !resetPath( lfr ) )
        {
            return;
        }

        flatlogs::timespecX now;
        now.gettime();
        writeAt( lfr, now, msg );

        REQUIRE( lfr.test_isOpen() );

        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );

        REQUIRE( lfr.test_isOpen() );

        lfr.close();
    }

    SECTION( "maxLogTime of 0 never closes on interval" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 0 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        flatlogs::timespecX old( 1732170780, 2 );
        writeAt( lfr, old, msg );

        REQUIRE( lfr.test_isOpen() );

        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );

        REQUIRE( lfr.test_isOpen() );

        lfr.close();
    }

    SECTION( "no file open is not an error" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 1 );

        REQUIRE( !lfr.test_isOpen() );
        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );
        REQUIRE( !lfr.test_isOpen() );
    }

    SECTION( "the next entry after an interval close opens a file stamped with its own time" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 1 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        flatlogs::timespecX old( 1732170780, 2 ); // 2024-11-21 06:33:00
        writeAt( lfr, old, msg );

        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );
        REQUIRE( !lfr.test_isOpen() );

        // A later entry, still in the past but in a different interval
        flatlogs::timespecX later( 1732171200, 4 ); // 2024-11-21 06:40:00
        writeAt( lfr, later, msg );

        REQUIRE( lfr.test_isOpen() );
        REQUIRE( lfr.test_currFileStartSec() == 1732171200 );

        std::string p1 = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121063300000000002." + lfr.logExt();
        std::string p2 = lfr.testPath + "/2024_11_21/" + lfr.logName() + "_20241121064000000000004." + lfr.logExt();

        REQUIRE( std::filesystem::exists( p1 ) );
        REQUIRE( std::filesystem::exists( p2 ) );

        lfr.close();
    }
}


/// Rotation with late and future timestamps
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Rotation with late and future timestamps", "[libMagAOX::logger::logFileRaw]" )
{
    // 10 minute (600 second) intervals:
    //   1732170780 = 2024-11-21 06:33:00 UTC, interval K
    //   1732171199 = 2024-11-21 06:39:59 UTC, interval K
    //   1732171200 = 2024-11-21 06:40:00 UTC, interval K+1
    //   1732171260 = 2024-11-21 06:41:00 UTC, interval K+1
    //   4070908800 = 2099-01-01 00:00:00 UTC, later than the current clock

    std::string msg( 256, 't' );

    auto writeAt = []( logFileRawTest &lfr, flatlogs::timespecX ts, const std::string &m )
    {
        flatlogs::bufferPtrT logbuff;
        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, m, flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );
    };

    auto countFiles = []( const std::string &path )
    {
        int n = 0;
        for( const auto &e : std::filesystem::recursive_directory_iterator( path ) )
        {
            if( std::filesystem::is_regular_file( e ) )
            {
                ++n;
            }
        }
        return n;
    };

    auto resetPath = []( logFileRawTest &lfr )
    {
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return false;
        }
        std::filesystem::remove_all( lfr.testPath );
        return true;
    };

    SECTION( "out of order entries at a boundary produce one file per interval" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 1 ), msg ); // K
        writeAt( lfr, flatlogs::timespecX( 1732171200, 2 ), msg ); // K+1
        writeAt( lfr, flatlogs::timespecX( 1732171199, 3 ), msg ); // K, arriving late
        writeAt( lfr, flatlogs::timespecX( 1732171260, 4 ), msg ); // K+1

        lfr.close();

        REQUIRE( countFiles( lfr.testPath ) == 2 );
    }

    SECTION( "an older entry after a newer file does not start a new file" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732171200, 2 ), msg ); // K+1
        writeAt( lfr, flatlogs::timespecX( 1732170780, 1 ), msg ); // K, older

        REQUIRE( lfr.test_currFileStartSec() == 1732171200 );

        lfr.close();

        REQUIRE( countFiles( lfr.testPath ) == 1 );
        REQUIRE( std::filesystem::file_size( lfr.testPath + "/2024_11_21/" + lfr.logName() +
                                             "_20241121064000000000002." + lfr.logExt() ) == 2 * ( 256 + 14 ) );
    }

    SECTION( "a future timestamp does not start a new file or a future-dated directory" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 1 ), msg );
        writeAt( lfr, flatlogs::timespecX( 4070908800, 2 ), msg ); // 2099
        writeAt( lfr, flatlogs::timespecX( 1732170790, 3 ), msg );

        REQUIRE( lfr.test_currFileStartSec() == 1732170780 );

        lfr.close();

        REQUIRE( countFiles( lfr.testPath ) == 1 );
        REQUIRE( !std::filesystem::exists( lfr.testPath + "/2099_01_01" ) );
    }

    SECTION( "with no file open, a future timestamp names the file no later than now" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 10 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 4070908800, 2 ), msg ); // 2099

        flatlogs::timespecX now;
        now.gettime();

        REQUIRE( lfr.test_currFileStartSec() <= now.time_s );

        lfr.close();

        REQUIRE( countFiles( lfr.testPath ) == 1 );
        REQUIRE( !std::filesystem::exists( lfr.testPath + "/2099_01_01" ) );
    }
}

/// Unique file names when entries share a timestamp
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Unique file names when entries share a timestamp", "[libMagAOX::logger::logFileRaw]" )
{
    std::string msg( 256, 't' );

    auto writeAt = []( logFileRawTest &lfr, flatlogs::timespecX ts, const std::string &m )
    {
        flatlogs::bufferPtrT logbuff;
        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, m, flatlogs::logPrio::LOG_NOTICE );
        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );
    };

    auto pathFor = []( logFileRawTest &lfr, const std::string &stamp )
    { return lfr.testPath + "/2024_11_21/" + lfr.logName() + "_" + stamp + "." + lfr.logExt(); };

    auto resetPath = []( logFileRawTest &lfr )
    {
        if( lfr.testPath == "/tmp" )
        {
            std::cerr << "\nTESTING-ERROR: testPath is just /tmp, so logName is null.  Can't go on\n";
            std::cerr << __FILE__ << ' ' << __LINE__ << "\n\n";
            REQUIRE( false );
            return false;
        }
        std::filesystem::remove_all( lfr.testPath );
        return true;
    };

    auto countFiles = []( const std::string &path )
    {
        int n = 0;
        for( const auto &e : std::filesystem::recursive_directory_iterator( path ) )
        {
            if( std::filesystem::is_regular_file( e ) )
            {
                ++n;
            }
        }
        return n;
    };

    SECTION( "an entry repeated after an idle close is written to a file named from the current time" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 1 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 0 ), msg );

        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );
        REQUIRE( !lfr.test_isOpen() );

        flatlogs::timespecX before;
        before.gettime();

        // Same timestamp again: writeAt requires this to succeed rather than return eexist
        writeAt( lfr, flatlogs::timespecX( 1732170780, 0 ), msg );

        REQUIRE( lfr.test_currFileStartSec() >= before.time_s );

        lfr.close();

        REQUIRE( std::filesystem::exists( pathFor( lfr, "20241121063300000000000" ) ) );
        REQUIRE( countFiles( lfr.testPath ) == 2 );
    }

    SECTION( "the fallback file stays open, so later entries with the same timestamp do not create more files" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( MAGAOX_max_maxLogTime ); // the longest interval, so the test can not straddle a boundary
        if( !resetPath( lfr ) )
        {
            return;
        }

        flatlogs::timespecX old( 1732170780, 0 ); // 2024, in an earlier interval than the current time

        writeAt( lfr, old, msg );
        REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );
        REQUIRE( !lfr.test_isOpen() );

        writeAt( lfr, old, msg ); // collides, so the file is named from the current time

        for( int n = 0; n < 5; ++n )
        {
            REQUIRE( lfr.closeIfIntervalElapsed() == mx::error_t::noerror );
            REQUIRE( lfr.test_isOpen() );
            writeAt( lfr, old, msg );
        }

        lfr.close();

        REQUIRE( countFiles( lfr.testPath ) == 2 );
    }

    SECTION( "repeated rotations onto one timestamp all succeed" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 0 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 5 ), msg );

        for( int n = 0; n < 20; ++n )
        {
            lfr.requestRotation();
            writeAt( lfr, flatlogs::timespecX( 1732170780, 5 ), msg );
        }

        lfr.close();

        REQUIRE( countFiles( lfr.testPath ) == 21 );
    }

    SECTION( "a collision reports the taken name once" )
    {
        logFileRawTest lfr;
        lfr.maxLogTime( 0 );
        if( !resetPath( lfr ) )
        {
            return;
        }

        writeAt( lfr, flatlogs::timespecX( 1732170780, 5 ), msg );
        lfr.requestRotation();

        flatlogs::bufferPtrT logbuff;
        flatlogs::logHeader::createLog<dummyLog>(
            logbuff, flatlogs::timespecX( 1732170780, 5 ), msg, flatlogs::logPrio::LOG_NOTICE );

        std::stringstream captured;
        std::streambuf   *orig = std::cerr.rdbuf( captured.rdbuf() );

        mx::error_t rv = lfr.writeLog( logbuff );

        std::cerr.rdbuf( orig );

        REQUIRE( rv == mx::error_t::noerror );

        std::string out     = captured.str();
        size_t      reports = 0;
        for( size_t pos = out.find( "(eexist)" ); pos != std::string::npos; pos = out.find( "(eexist)", pos + 1 ) )
        {
            ++reports;
        }

        REQUIRE( reports == 1 );

        lfr.close();
    }
}

/// Validation of maxLogTime
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "Validation of maxLogTime", "[libMagAOX::logger::logFileRaw]" )
{
    SECTION( "the setter accepts 0 and the maximum, and rejects larger values keeping the old value" )
    {
        logFileRawTest lfr;

        REQUIRE( lfr.maxLogTime( 0 ) == mx::error_t::noerror );
        REQUIRE( lfr.maxLogTime() == 0 );

        REQUIRE( lfr.maxLogTime( MAGAOX_max_maxLogTime ) == mx::error_t::noerror );
        REQUIRE( lfr.maxLogTime() == MAGAOX_max_maxLogTime );

        REQUIRE( lfr.maxLogTime( 15 ) == mx::error_t::noerror );
        REQUIRE( lfr.maxLogTime( MAGAOX_max_maxLogTime + 1 ) == mx::error_t::erange );
        REQUIRE( lfr.maxLogTime() == 15 );
    }

    SECTION( "the parser accepts numbers within range, rounding to whole minutes" )
    {
        std::vector<std::pair<std::string, unsigned>> good = { { "0", 0 },
                                                               { "15", 15 },
                                                               { "15.0", 15 },
                                                               { " 15.0 ", 15 },
                                                               { "\t42\t", 42 },
                                                               { "1e1", 10 },
                                                               { "14.9999999", 15 },
                                                               { "15.4", 15 },
                                                               { "15.6", 16 },
                                                               { "525600", 525600 },
                                                               { "0.0", 0 },
                                                               { "-0", 0 },
                                                               { "0.5", 1 },
                                                               { "0.6", 1 } };

        for( const auto &g : good )
        {
            unsigned v = 7;
            INFO( "input \"" << g.first << "\"" );
            REQUIRE( logFileRawTest::parseMaxLogTime( v, g.first ) == mx::error_t::noerror );
            REQUIRE( v == g.second );
        }
    }

    SECTION( "the parser rejects strings which are not numbers, leaving the value unchanged" )
    {
        std::vector<std::string> invalid = { "abc", "", "   ", "15abc", "1 5", "nan", "inf", "infinity", "1e", "..", "+-5", "5-" };

        for( const auto &str : invalid )
        {
            unsigned v = 7;
            INFO( "input \"" << str << "\"" );
            REQUIRE( logFileRawTest::parseMaxLogTime( v, str ) == mx::error_t::invalidarg );
            REQUIRE( v == 7 );
        }
    }

    SECTION( "the parser rejects numbers out of range, leaving the value unchanged" )
    {
        std::vector<std::string> outOfRange = { "-5", "-0.6", "525601", "99999999999", "1e30", "1e999", "0.4", "0.0001", "1e-10" };

        for( const auto &str : outOfRange )
        {
            unsigned v = 7;
            INFO( "input \"" << str << "\"" );
            REQUIRE( logFileRawTest::parseMaxLogTime( v, str ) == mx::error_t::erange );
            REQUIRE( v == 7 );
        }
    }
}

} // namespace logFileRawTest
} // namespace loggerTest
} // namespace libXWCTest
