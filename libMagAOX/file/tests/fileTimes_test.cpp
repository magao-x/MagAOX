/** \file fileTimes_test.hpp
 * \brief Tests for file timestamps
 * \ingroup file_files
 */


#include <iostream>

#include "../../../tests/testXWC.hpp"

#include "../fileTimes.hpp"

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_TIMESTAMP_THROW_BAD_ALLOC_ns
#define XWCTEST_TIMESTAMP_THROW_BAD_ALLOC
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TIMESTAMP_THROW_BAD_ALLOC

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_TIMESTAMP_THROW_FORMAT_ERROR_ns
#define XWCTEST_TIMESTAMP_THROW_FORMAT_ERROR
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TIMESTAMP_THROW_FORMAT_ERROR

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_TIMESTAMP_THROW_EXCEPTION_ns
#define XWCTEST_TIMESTAMP_THROW_EXCEPTION
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TIMESTAMP_THROW_EXCEPTION

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_TIMESTAMP_GMTIME_OTHER_ns
#define XWCTEST_TIMESTAMP_GMTIME_OTHER
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TIMESTAMP_GMTIME_OTHER

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_FILETIMERELPATH_THROW_BAD_ALLOC_ns
#define XWCTEST_FILETIMERELPATH_THROW_BAD_ALLOC
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_FILETIMERELPATH_THROW_BAD_ALLOC

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_FILETIMERELPATH_THROW_FORMAT_ERROR_ns
#define XWCTEST_FILETIMERELPATH_THROW_FORMAT_ERROR
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_FILETIMERELPATH_THROW_FORMAT_ERROR

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_FILETIMERELPATH_THROW_EXCEPTION_ns
#define XWCTEST_FILETIMERELPATH_THROW_EXCEPTION
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_FILETIMERELPATH_THROW_EXCEPTION

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_FILETIMERELPATHSTRING_THROW_BAD_ALLOC_ns
#define XWCTEST_FILETIMERELPATHSTRING_THROW_BAD_ALLOC
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_FILETIMERELPATHSTRING_THROW_BAD_ALLOC

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_FILETIMERELPATHSTRING_THROW_EXCEPTION_ns
#define XWCTEST_FILETIMERELPATHSTRING_THROW_EXCEPTION
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_FILETIMERELPATHSTRING_THROW_EXCEPTION

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_PARSETIMESTAMP_THROW_BAD_ALLOC_ns
#define XWCTEST_PARSETIMESTAMP_THROW_BAD_ALLOC
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_PARSETIMESTAMP_THROW_BAD_ALLOC

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_PARSETIMESTAMP_THROW_OUT_OF_RANGE_ns
#define XWCTEST_PARSETIMESTAMP_THROW_OUT_OF_RANGE
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_PARSETIMESTAMP_THROW_OUT_OF_RANGE

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_PARSETIMESTAMP_THROW_EXCEPTION_ns
#define XWCTEST_PARSETIMESTAMP_THROW_EXCEPTION
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_PARSETIMESTAMP_THROW_EXCEPTION

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_PARSEFILEPATH_THROW_BAD_ALLOC_ns
#define XWCTEST_PARSEFILEPATH_THROW_BAD_ALLOC
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_PARSEFILEPATH_THROW_BAD_ALLOC

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_PARSEFILEPATH_THROW_OUT_OF_RANGE_ns
#define XWCTEST_PARSEFILEPATH_THROW_OUT_OF_RANGE
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_PARSEFILEPATH_THROW_OUT_OF_RANGE

#undef file_fileTimes_hpp
#define XWCTEST_NAMESPACE XWCTEST_PARSEFILEPATH_THROW_EXCEPTION_ns
#define XWCTEST_PARSEFILEPATH_THROW_EXCEPTION
#include "../fileTimes.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_PARSEFILEPATH_THROW_EXCEPTION

namespace libXWCTest
{

/** \defgroup file_unit_test libXWC::file Unit Tests
 * \ingroup unit_test
*/

/// Namespace for XWC::file tests
/** \ingroup file_unit_test
 *
*/
namespace fileTest
{

/** \defgroup fileTimes_unit_test fileTimes Unit Tests
 * \ingroup file_unit_test
*/

/// Namespace for XWC::file::fileTimes tests
/** \ingroup fileTimes_unit_test
 *
*/
namespace fileTimesTest
{


/// Getting timestamp string and broken-down time for a given time
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp string and broken-down time for a given time", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc = MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( tstamp == "20241121063300000000000" );
        REQUIRE( uttime.tm_year == 124 );
        REQUIRE( uttime.tm_mon == 10 );
        REQUIRE( uttime.tm_mday == 21 );
        REQUIRE( uttime.tm_hour == 6 );
        REQUIRE( uttime.tm_min == 33 );
        REQUIRE( uttime.tm_sec == 0 );
    }

    SECTION( "A time with non-0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170781;
        unsigned long ts_nsec = 0;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc = MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( tstamp == "20241121063301000000000" );
        REQUIRE( uttime.tm_year == 124 );
        REQUIRE( uttime.tm_mon == 10 );
        REQUIRE( uttime.tm_mday == 21 );
        REQUIRE( uttime.tm_hour == 6 );
        REQUIRE( uttime.tm_min == 33 );
        REQUIRE( uttime.tm_sec == 1 );
    }

    SECTION( "A time with non-0 sec and 9-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 434878292;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc = MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( tstamp == "20241121063305434878292" );
        REQUIRE( uttime.tm_year == 124 );
        REQUIRE( uttime.tm_mon == 10 );
        REQUIRE( uttime.tm_mday == 21 );
        REQUIRE( uttime.tm_hour == 6 );
        REQUIRE( uttime.tm_min == 33 );
        REQUIRE( uttime.tm_sec == 5 );
    }

    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 292;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc = MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( tstamp == "20241121063305000000292" );
        REQUIRE( uttime.tm_year == 124 );
        REQUIRE( uttime.tm_mon == 10 );
        REQUIRE( uttime.tm_mday == 21 );
        REQUIRE( uttime.tm_hour == 6 );
        REQUIRE( uttime.tm_min == 33 );
        REQUIRE( uttime.tm_sec == 5 );
    }
}

/// Getting timestamp and broken-down time with errors
/**
 * This is in a separate file due to need to define a buffer size too small to generate errors
 *
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp and broken-down time with errors", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A year that's too big (gmtime_r error)" )
    {
        time_t        ts_sec  = 1.355388599402496e+17; // huge year
        unsigned long ts_nsec = 0;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc = MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::eoverflow );
    }

    SECTION( "gmtime_r error with errno == 0" )
    {
        time_t        ts_sec  = 1.355388599402496e+17; // huge year
        unsigned long ts_nsec = 0;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc =
            MagAOX::file::XWCTEST_TIMESTAMP_GMTIME_OTHER_ns::timestamp( tstamp, uttime, ts_sec, ts_nsec );

        XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::error );
    }
}

/// Getting timestamp only with errors
/**
 * This is in a separate file due to need to define a buffer size too small to generate errors
 *
 * \anchor tests_libMagAOX_file_fileTimes_timestamp_only_errors
 */
TEST_CASE( "Getting timestamp only with errors", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A year that's too big (gmtime_r error)" )
    {
        time_t        ts_sec  = 1.355388599402496e+17; // huge year
        unsigned long ts_nsec = 0;

        std::string tstamp;

        mx::error_t errc = MagAOX::file::timestamp( tstamp, ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::eoverflow );
    }
}


/// Getting timestamp string and broken-down time causes bad_alloc
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp string and broken-down time causes bad_alloc", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {

        bool caught = false;

        try
        {
            time_t        ts_sec  = 1732170785;
            unsigned long ts_nsec = 292;

            std::string tstamp;
            tm          uttime;

            MagAOX::file::XWCTEST_TIMESTAMP_THROW_BAD_ALLOC_ns::timestamp( tstamp, uttime, ts_sec, ts_nsec );

            XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec ) );
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY>  &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }
}

/// Getting timestamp only causes bad_alloc
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp only causes bad_alloc", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 292;

        std::string tstamp;

        bool caught = false;

        try
        {
            MagAOX::file::XWCTEST_TIMESTAMP_THROW_BAD_ALLOC_ns::timestamp( tstamp, ts_sec, ts_nsec );
            XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, ts_sec, ts_nsec ) );
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY> &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }
}

/// Getting timestamp string and broken-down time causes filesystem_error
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp string and broken-down time causes filesystem_error", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 292;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc =
            MagAOX::file::XWCTEST_TIMESTAMP_THROW_FORMAT_ERROR_ns::timestamp( tstamp, uttime, ts_sec, ts_nsec );
        XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::std_format_error );
    }
}

/// Getting timestamp only causes filesystem_error
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp only causes filesystem_error", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 292;

        std::string tstamp;

        mx::error_t errc = MagAOX::file::XWCTEST_TIMESTAMP_THROW_FORMAT_ERROR_ns::timestamp( tstamp, ts_sec, ts_nsec );
        XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::std_format_error );
    }
}

/// Getting timestamp string and broken-down time causes exception
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp string and broken-down time causes exception", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 292;

        std::string tstamp;
        tm          uttime;

        mx::error_t errc =
            MagAOX::file::XWCTEST_TIMESTAMP_THROW_EXCEPTION_ns::timestamp( tstamp, uttime, ts_sec, ts_nsec );
        XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, uttime, ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::std_exception );
    }
}

/// Getting timestamp only causes exception
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting timestamp only causes exception", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with non-0 sec and 3-digit nsec" )
    {
        time_t        ts_sec  = 1732170785;
        unsigned long ts_nsec = 292;

        std::string tstamp;

        mx::error_t errc = MagAOX::file::XWCTEST_TIMESTAMP_THROW_EXCEPTION_ns::timestamp( tstamp, ts_sec, ts_nsec );
        XWCTEST_DOXYGEN_REF( MagAOX::file::timestamp( tstamp, ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::std_exception );
    }
}

/// Getting filename and relative path for a given time
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( fileName == "tdevice_20241121063300000000000.txt" );
        REQUIRE( relPath == "tdevice/2024_11_21" );
    }
}

/// Getting filename and relative path for a given time causes overflow
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes overflow", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A year that's too big (gmtime_r error)" )
    {
        time_t        ts_sec  = 1.355388599402496e+17; // huge year
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

        REQUIRE( errc == mx::error_t::eoverflow );
    }
}

/// Getting filename and relative path for a given time causes bad_alloc in timestamp
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes bad_alloc in timestamp",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {

        bool caught = false;
        try
        {
            time_t        ts_sec  = 1732170780;
            unsigned long ts_nsec = 0;
            std::string   fileName, relPath;

            MagAOX::file::XWCTEST_TIMESTAMP_THROW_BAD_ALLOC_ns::fileTimeRelPath(
                fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );
            XWCTEST_DOXYGEN_REF(
                MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ) );
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY> &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }
}

/// Getting filename and relative path for a given time causes format_error in timestamp
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes format_error in timestamp",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::XWCTEST_TIMESTAMP_THROW_FORMAT_ERROR_ns::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

        XWCTEST_DOXYGEN_REF( MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::std_format_error );
    }
}

/// Getting filename and relative path for a given time causes exception in timestamp
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes exception in timestamp",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::XWCTEST_TIMESTAMP_THROW_EXCEPTION_ns::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

        XWCTEST_DOXYGEN_REF( MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ) );

        REQUIRE( errc == mx::error_t::std_exception );
    }
}

/// Getting filename and relative path for a given time causes bad_alloc in top relpath
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes bad_alloc in top relpath",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        bool caught = false;
        try
        {
            MagAOX::file::XWCTEST_FILETIMERELPATH_THROW_BAD_ALLOC_ns::fileTimeRelPath(
                fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

            XWCTEST_DOXYGEN_REF(
                MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ) );
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY> &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }
}

/// Getting filename and relative path for a given time causes bad_alloc in relpath string
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes bad_alloc in relpath string",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        bool caught = false;
        try
        {
            MagAOX::file::XWCTEST_FILETIMERELPATHSTRING_THROW_BAD_ALLOC_ns::fileTimeRelPath(
                fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

            XWCTEST_DOXYGEN_REF(
                MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ) );
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY>  &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }
}

/// Getting filename and relative path for a given time causes format_error in top relpath
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes format_error in top relpath",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::XWCTEST_FILETIMERELPATH_THROW_FORMAT_ERROR_ns::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

        XWCTEST_DOXYGEN_REF(MagAOX::file::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ));

        REQUIRE( errc == mx::error_t::std_format_error );

        // for doxygen (not a test):
        MagAOX::file::fileTimeRelPath( fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );
    }
}

/// Getting filename and relative path for a given time causes exception in top relpath
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes exception in top relpath",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::XWCTEST_FILETIMERELPATH_THROW_EXCEPTION_ns::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

        XWCTEST_DOXYGEN_REF(MagAOX::file::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ));

        REQUIRE( errc == mx::error_t::std_exception );
    }
}

/// Getting filename and relative path for a given time causes exception in relpath string
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Getting filename and relative path for a given time causes exception in relpath string",
           "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A time with 0 sec and 0 nsec" )
    {
        time_t        ts_sec  = 1732170780;
        unsigned long ts_nsec = 0;

        std::string fileName, relPath;

        mx::error_t errc = MagAOX::file::XWCTEST_FILETIMERELPATHSTRING_THROW_EXCEPTION_ns::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec );

            XWCTEST_DOXYGEN_REF(MagAOX::file::fileTimeRelPath(
            fileName, relPath, "tdevice", "txt", ts_sec, ts_nsec ));

        REQUIRE( errc == mx::error_t::std_exception );
    }
}

/// Parsing filenames, paths and timestamps, with no errors
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Parsing filenames, paths and timestamps, with no errors", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "A valid MagAO-X filename" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "device_20241121063300000000000.txt" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "device" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filepath" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "device" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filename without extension" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "device_20241121063300000000000" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "device" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filepath without extension" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "device" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filename without device, no _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "20241121063300000000000.txt" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filepath without device, no _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/20241121063300000000000.txt" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filename without device or extension, no _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "20241121063300000000000" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filepath without device or extension, no _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/20241121063300000000000" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filename without device, with _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "_20241121063300000000000.txt" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filepath without device, with _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/_20241121063300000000000.txt" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filename without device or extension, with _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "_20241121063300000000000" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }

    SECTION( "A valid MagAO-X filepath without device or extension, with _" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc =
            MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/_20241121063300000000000" );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( devName == "" );
        REQUIRE( YYYY == "2024" );
        REQUIRE( MM == "11" );
        REQUIRE( DD == "21" );
        REQUIRE( hh == "06" );
        REQUIRE( mm == "33" );
        REQUIRE( ss == "00" );
        REQUIRE( nn == "000000000" );
    }
}

/// Parsing filenames and paths with errors
/**
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Parsing filenames and paths with errors", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "An invalid MagAO-X filepath with too short timestamp" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_2024112106330000000000.txt" );

        REQUIRE( errc == mx::error_t::invalidarg );
    }

    SECTION( "An invalid MagAO-X filepath with too long timestamp" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_202411210633000000000001.txt" );
        REQUIRE( errc == mx::error_t::invalidarg );
    }

    SECTION( "An valid MagAO-X filepath but bad_alloc is thrown in parseTimeStamp" )
    {

        bool caught = false;
        try
        {
            std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

            MagAOX::file::XWCTEST_PARSETIMESTAMP_THROW_BAD_ALLOC_ns::parseFilePath(
                devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

            XWCTEST_DOXYGEN_REF(MagAOX::file::parseFilePath(
                devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" ));
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY> & e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }

    SECTION( "An valid MagAO-X filepath but out_of_range is thrown in parseTimeStamp" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::XWCTEST_PARSETIMESTAMP_THROW_OUT_OF_RANGE_ns::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

            XWCTEST_DOXYGEN_REF(MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" ));

        REQUIRE( errc == mx::error_t::std_out_of_range );
    }

    SECTION( "An valid MagAO-X filepath but exception is thrown in parseTimeStamp" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::XWCTEST_PARSETIMESTAMP_THROW_EXCEPTION_ns::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

            XWCTEST_DOXYGEN_REF(MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" ));

        REQUIRE( errc == mx::error_t::std_exception );
    }

    SECTION( "An valid MagAO-X filepath but bad_alloc is thrown in parseFilePath" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        bool caught = false;
        try
        {
            MagAOX::file::XWCTEST_PARSEFILEPATH_THROW_BAD_ALLOC_ns::parseFilePath(
                devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

            XWCTEST_DOXYGEN_REF(MagAOX::file::parseFilePath(
                devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" ));
        }
        catch( const mx::exception<XWC_DEFAULT_VERBOSITY>  &e )
        {
            caught = true;
        }
        REQUIRE( caught == true );
    }

    SECTION( "An valid MagAO-X filepath but out_of_range is thrown in parseFilePAth" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::XWCTEST_PARSEFILEPATH_THROW_OUT_OF_RANGE_ns::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

        XWCTEST_DOXYGEN_REF(MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" ));

        REQUIRE( errc == mx::error_t::std_out_of_range );
    }

    SECTION( "An valid MagAO-X filepath but exception is thrown in parseFilePAth" )
    {
        std::string devName, YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::XWCTEST_PARSEFILEPATH_THROW_EXCEPTION_ns::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" );

            XWCTEST_DOXYGEN_REF(MagAOX::file::parseFilePath(
            devName, YYYY, MM, DD, hh, mm, ss, nn, "/path/to/device_20241121063300000000000.txt" ));

        REQUIRE( errc == mx::error_t::std_exception );
    }
}

/// Parsing timestamps with errors
/**
 * Tests only size errors.  Exceptions tested with parseFilePath tests.
 *
 * \ingroup fileTimes_unit_test
 */
TEST_CASE( "Parsing timestamps with errors", "[libMagAOX::file::fileTimes]" )
{
    SECTION( "An invalid MagAO-X timestamp, too short" )
    {
        std::string YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseTimestamp( YYYY, MM, DD, hh, mm, ss, nn, "202411210633000000000" );

        REQUIRE( errc == mx::error_t::invalidarg );
    }

    SECTION( "An invalid MagAO-X timestamp, too long" )
    {
        std::string YYYY, MM, DD, hh, mm, ss, nn;

        mx::error_t errc = MagAOX::file::parseTimestamp( YYYY, MM, DD, hh, mm, ss, nn, "202411210633000000000001" );
        REQUIRE( errc == mx::error_t::invalidarg );
    }
}

} // namespace fileTimesTest
} // namespace fileTest
} // namespace libXWCTest
