/** \file logFileRaw_test.hpp
 * \brief Tests for the logFileRaw class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logFileRaw.hpp"

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

        lfr.logPath( "/newp/test/x" );
        REQUIRE( lfr.logPath() == "/newp/test/x" );

        lfr.logName( "newdev" );
        REQUIRE( lfr.logName() == "newdev" );

        lfr.logExt( "bintel" );
        REQUIRE( lfr.logExt() == "bintel" );

        lfr.maxLogSize( 10 );
        REQUIRE( lfr.maxLogSize() == 10 );
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

} // namespace logFileRawTest
} // namespace loggerTest
} // namespace libXWCTest
