/** \file logFileRaw_test.hpp
 * \brief Tests for the logFileRaw class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logFileRaw.hpp"

// Fault injection. Each block below compiles logFileRaw.hpp a second time inside a test
// namespace with one XWCTEST_ fault macro defined. The macro turns one production error
// branch on that cannot be reached through the public interface, such as a failed fwrite
// on a healthy file. The include guard is undefined first so the header is really
// re-read. The tests at the end of this file use the namespaced logFileRaw types.

// Forces logPath() to throw a std::bad_alloc inside its try block.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_LOGPATH_BAD_ALLOC_ns
#define XWCTEST_LOGFILERAW_LOGPATH_BAD_ALLOC
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_LOGPATH_BAD_ALLOC

// Forces logPath() to throw a plain std::exception inside its try block.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_LOGPATH_EXCEPTION_ns
#define XWCTEST_LOGFILERAW_LOGPATH_EXCEPTION
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_LOGPATH_EXCEPTION

// Forces logName() to throw a std::bad_alloc inside its try block.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_LOGNAME_BAD_ALLOC_ns
#define XWCTEST_LOGFILERAW_LOGNAME_BAD_ALLOC
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_LOGNAME_BAD_ALLOC

// Forces logName() to throw a plain std::exception inside its try block.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_LOGNAME_EXCEPTION_ns
#define XWCTEST_LOGFILERAW_LOGNAME_EXCEPTION
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_LOGNAME_EXCEPTION

// Forces logExt() to throw a std::bad_alloc inside its try block.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_LOGEXT_BAD_ALLOC_ns
#define XWCTEST_LOGFILERAW_LOGEXT_BAD_ALLOC
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_LOGEXT_BAD_ALLOC

// Forces logExt() to throw a plain std::exception inside its try block.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_LOGEXT_EXCEPTION_ns
#define XWCTEST_LOGFILERAW_LOGEXT_EXCEPTION
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_LOGEXT_EXCEPTION

// Makes writeLog() see fwrite report zero bytes written.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL_ns
#define XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL

// Makes flush() see fflush fail with an I/O error.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_FLUSH_FFLUSH_FAIL_ns
#define XWCTEST_LOGFILERAW_FLUSH_FFLUSH_FAIL
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_FLUSH_FFLUSH_FAIL

// Makes close() see fclose fail with an I/O error.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_CLOSE_FCLOSE_FAIL_ns
#define XWCTEST_LOGFILERAW_CLOSE_FCLOSE_FAIL
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_CLOSE_FCLOSE_FAIL

// Forces createFile() to throw right after fileTimeRelPath, so its catch block wraps
// and rethrows the exception.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_CREATEFILE_EXCEPTION_ns
#define XWCTEST_LOGFILERAW_CREATEFILE_EXCEPTION
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_CREATEFILE_EXCEPTION

// Makes the directory existence check in createFile() report a permission error.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_CREATEFILE_EXISTS_ERRC_ns
#define XWCTEST_LOGFILERAW_CREATEFILE_EXISTS_ERRC
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_CREATEFILE_EXISTS_ERRC

// Closes the file that fopen just opened inside createFile() and pretends fopen failed.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_CREATEFILE_FOPEN_FAIL_ns
#define XWCTEST_LOGFILERAW_CREATEFILE_FOPEN_FAIL
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_CREATEFILE_FOPEN_FAIL

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

        lfr.maxLogSize( 10 ); // arbitrary small value for the setter round trip
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
        flatlogs::timespecX ts1( 1732170780, 1 ); // 2024-11-21 06:33:00 UTC, an arbitrary fixed time used throughout this file
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

        flatlogs::timespecX ts2( 1763706780, 50 ); // one year after ts1, so a new dated directory is created

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
        std::string          msg( 256, 't' ); // filler message. The file size checks below are built on 256

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts2, msg, flatlogs::logPrio::LOG_NOTICE );

        rv = lfr.writeLog( logbuff );
        REQUIRE( rv == mx::error_t::noerror );

        REQUIRE( lfr.close() == mx::error_t::noerror );

        std::uintmax_t fsz = std::filesystem::file_size( fullPath );

        REQUIRE( fsz == 1 * ( 256 + 14 ) ); // 14 is the flatlogs header size for a 256 byte message
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

        REQUIRE( fsz == 1 * ( 256 + 14 ) ); // 14 is the flatlogs header size for a 256 byte message
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

        REQUIRE( fsz == 2 * ( 256 + 14 ) ); // has two logs in it. 14 is the flatlogs header size for a 256 byte message
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

        REQUIRE( fsz == 1 * ( 256 + 14 ) ); // 14 is the flatlogs header size for a 256 byte message

        lfr.close();

        fsz = std::filesystem::file_size( fullPath2 );

        REQUIRE( fsz == 1 * ( 256 + 14 ) ); // 14 is the flatlogs header size for a 256 byte message
    }
}

/// The exception paths in the string-valued setters logPath, logName, and logExt. Each
/// section uses a fault-injected build that throws inside the setter, and checks that a
/// bad_alloc is nested in an xwcException while a plain std::exception becomes an error
/// code.
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "logFileRaw setter exception paths", "[libMagAOX::logger::logFileRaw]" )
{
    SECTION( "logPath nests a bad_alloc in an xwcException" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_LOGPATH_BAD_ALLOC_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;

        bool caught = false;

        try
        {
            lfr.logPath( "/tmp" );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }

    SECTION( "logPath reports a std::exception as an error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_LOGPATH_EXCEPTION_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;

        mx::error_t rv = lfr.logPath( "/tmp" );

        REQUIRE( rv == mx::error_t::std_exception );
    }

    SECTION( "logName nests a bad_alloc in an xwcException" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_LOGNAME_BAD_ALLOC_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;

        bool caught = false;

        try
        {
            lfr.logName( "newdev" );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }

    SECTION( "logName reports a std::exception as an error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_LOGNAME_EXCEPTION_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;

        mx::error_t rv = lfr.logName( "newdev" );

        REQUIRE( rv == mx::error_t::std_exception );
    }

    SECTION( "logExt nests a bad_alloc in an xwcException" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_LOGEXT_BAD_ALLOC_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;

        bool caught = false;

        try
        {
            lfr.logExt( "bintel" );
        }
        catch( const MagAOX::xwcException &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }

    SECTION( "logExt reports a std::exception as an error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_LOGEXT_EXCEPTION_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;

        mx::error_t rv = lfr.logExt( "bintel" );

        REQUIRE( rv == mx::error_t::std_exception );
    }
}

/// The I/O error paths in writeLog, flush, close, and createFile. Fault-injected builds
/// force fwrite, fflush, fclose, and fopen failures that cannot be triggered on a healthy
/// file under /tmp. The last sections use the normal build for the paths that can be
/// reached directly.
/**
 * \ingroup logFileRaw_unit_test
 */
TEST_CASE( "logFileRaw internal I/O error paths", "[libMagAOX::logger::logFileRaw]" )
{
    SECTION( "writeLog reports an fwrite error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );

        REQUIRE( rv != mx::error_t::noerror );
    }

    SECTION( "flush reports an fflush error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_FLUSH_FFLUSH_FAIL_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );

        mx::error_t rv = lfr.flush();

        REQUIRE( rv != mx::error_t::noerror );
    }

    SECTION( "close reports an fclose error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_CLOSE_FCLOSE_FAIL_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );

        mx::error_t rv = lfr.close();

        REQUIRE( rv != mx::error_t::noerror );
    }

    SECTION( "createFile rethrows an unexpected exception from fileTimeRelPath" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_CREATEFILE_EXCEPTION_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        bool caught = false;

        try
        {
            lfr.writeLog( logbuff );
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY> &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }

    SECTION( "createFile reports an error from checking existence" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_CREATEFILE_EXISTS_ERRC_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );

        REQUIRE( rv == mx::error_t::eacces );
    }

    SECTION( "createFile reports an fopen error" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_CREATEFILE_FOPEN_FAIL_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );

        REQUIRE( rv == mx::error_t::eacces );
    }

    SECTION( "flush succeeds on an open file" )
    {
        MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 1732170780, 1 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );

        REQUIRE( lfr.flush() == mx::error_t::noerror );

        lfr.close();
    }

    SECTION( "writeLog reports an invalid all-zero timestamp from createFile" )
    {
        MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts( 0, 0 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts, msg, flatlogs::logPrio::LOG_NOTICE );

        mx::error_t rv = lfr.writeLog( logbuff );

        REQUIRE( rv == mx::error_t::invalidarg );
    }

    SECTION( "createFile continues after a close error while switching files" )
    {
        MagAOX::logger::XWCTEST_LOGFILERAW_CLOSE_FCLOSE_FAIL_ns::logFileRaw<XWC_DEFAULT_VERBOSITY> lfr;
        lfr.logPath( "/tmp" );
        lfr.maxLogSize( 256 );

        std::string testPath = lfr.logPath() + '/' + lfr.logName();
        std::filesystem::remove_all( testPath );

        flatlogs::bufferPtrT logbuff;
        flatlogs::timespecX  ts1( 1732170780, 2 );
        std::string          msg( 256, 't' );

        flatlogs::logHeader::createLog<dummyLog>( logbuff, ts1, msg, flatlogs::logPrio::LOG_NOTICE );

        // Opens the first file. m_fout starts null, so the internal close() call is a no-op.
        REQUIRE( lfr.writeLog( logbuff ) == mx::error_t::noerror );

        flatlogs::timespecX  ts2( 1732170780, 50 );
        flatlogs::bufferPtrT logbuff2;
        flatlogs::logHeader::createLog<dummyLog>( logbuff2, ts2, msg, flatlogs::logPrio::LOG_NOTICE );

        // This write exceeds maxLogSize, so createFile() runs again. This time m_fout is
        // open, so its internal close() call actually fires. That call fails because this
        // is the FCLOSE_FAIL build. createFile() should report the failure and continue on
        // to open the new file rather than aborting.
        mx::error_t rv = lfr.writeLog( logbuff2 );

        REQUIRE( rv == mx::error_t::noerror );
    }
}

} // namespace logFileRawTest
} // namespace loggerTest
} // namespace libXWCTest
