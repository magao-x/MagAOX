/** \file logFileRaw_test.hpp
 * \brief Tests for the logFileRaw class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logFileRaw.cpp"

namespace logFileRaw_test
{

class logFileRawTest : public MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY>
{

  typedef MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY> logFileRawT;

  public:
    std::string testPath;

    logFileRawTest()
    {
        logFileRawT::m_logPath = "/tmp";

        testPath = logFileRawT::m_logPath + '/' + logFileRawT::m_logName;
    }

    explicit logFileRawTest( const std::string &lp )
    {
        logFileRawT::m_logPath = lp;

        testPath = logFileRawT::m_logPath + '/' + logFileRawT::m_logName;
    }

    mx::error_t test_createFile( flatlogs::timespecX &ts )
    {
        return logFileRawT::createFile( ts );
    }
};


SCENARIO( "Creating a log file with error from fileTimeRelPath", "[libMagAOX::logger::logFileRaw]" )
{
    GIVEN( "A valid timestamp" )
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

        flatlogs::timespecX ts1( 0, 0 ); //0,0 is an error in fileTimeRelPath

        mx::error_t rv = lfr.test_createFile( ts1 );

        REQUIRE( rv != mx::error_t::noerror );

    }

}

} // namespace logFileRaw_test
