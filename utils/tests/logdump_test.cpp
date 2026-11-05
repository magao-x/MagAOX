/** \file logdump_test.cpp
 * \brief Offline regressions for complete and incrementally appended binary log records.
 * \ingroup logdump_unit_test
 */
#include "../../tests/testXWC.hpp"

#include <deque>
#include <fstream>
#include <functional>
#include <sstream>

#include "../../libMagAOX/libMagAOX.hpp"

namespace libXWCTest
{
namespace logdumpTest
{
/// \cond LOGDUMP_TEST_HARNESS
/// Append the next producer fragment immediately after a reader observes an incomplete record.
static std::function<void()> afterShortRead;

/// Preserve real stdio reads while exposing a deterministic producer/consumer boundary.
size_t readFile( void  *buffer /**< [out] Read destination. */,
                 size_t size /**< [in] Element size. */,
                 size_t count /**< [in] Requested element count. */,
                 FILE  *file /**< [in] Open local input file. */ )
{
    size_t received = std::fread( buffer, size, count, file );

    if( received < count && afterShortRead )
    {
        afterShortRead();
    }

    return received;
}
/// \endcond
} // namespace logdumpTest
} // namespace libXWCTest

#define fread libXWCTest::logdumpTest::readFile
#include "../logdump/logdump.hpp"
#undef fread

namespace libXWCTest
{
namespace logdumpTest
{
/// \cond LOGDUMP_TEST_HARNESS
/// Expose offline reader configuration without starting an application or instrument service.
class Reader : public logdump
{
  public:
    using logdump::m_codes;
    using logdump::m_dir;
    using logdump::m_ext;
    using logdump::m_file;
    using logdump::m_fileCheckInterval;
    using logdump::m_follow;
    using logdump::m_level;
    using logdump::m_nfiles;
    using logdump::m_pauseTime;
    using logdump::m_prefixes;
    using logdump::m_time;
};

/// Own temporary local files, output capture and the short-read append barrier.
struct Fixture
{
    std::string             m_directory; ///< Unique directory owned by this fixture.
    std::string             m_path;      ///< Current local input file.
    std::ostringstream      m_output;    ///< Captured complete-record output.
    std::ostringstream      m_errors;    ///< Captured input diagnostics.
    std::streambuf         *m_stdout;    ///< Original output buffer restored at destruction.
    std::streambuf         *m_stderr;    ///< Original diagnostic buffer restored at destruction.
    std::deque<std::string> m_fragments; ///< Producer fragments appended at successive short reads.
    Reader                  m_reader;    ///< Reader configured only for the fixture file.

    /// Create an isolated file and capture the reader's streams.
    Fixture();

    /// Restore streams and remove only the directory owned by this fixture.
    ~Fixture();

    /// Replace the input with a complete or intentionally unfinished capture.
    void write( const std::string &bytes /**< [in] Initial file contents. */ );

    /// Supply producer fragments only after the consumer has actually observed a short read.
    void follow();
};

Fixture::Fixture()
{
    char        directory[] = "/tmp/logdump-test-XXXXXX";
    const char *created     = mkdtemp( directory );
    REQUIRE( created != nullptr );
    m_directory                  = created;
    m_path                       = m_directory + "/input.bintel";
    m_reader.m_file              = m_path;
    m_reader.m_pauseTime         = 0;
    m_reader.m_fileCheckInterval = 1;
    m_reader.m_level             = logPrio::LOG_TELEM;
    m_stdout                     = std::cout.rdbuf( m_output.rdbuf() );
    m_stderr                     = std::cerr.rdbuf( m_errors.rdbuf() );
}

Fixture::~Fixture()
{
    afterShortRead = {};
    std::cout.rdbuf( m_stdout );
    std::cerr.rdbuf( m_stderr );
    std::filesystem::remove_all( m_directory );
}

void Fixture::write( const std::string &bytes )
{
    std::ofstream file( m_path, std::ios::binary );
    file.write( bytes.data(), bytes.size() );
    REQUIRE( file.good() );
}

void Fixture::follow()
{
    m_reader.m_follow = true;
    afterShortRead    = [&]()
    {
        REQUIRE_FALSE( m_fragments.empty() );
        std::ofstream file( m_path, std::ios::binary | std::ios::app );
        const auto   &fragment = m_fragments.front();
        file.write( fragment.data(), fragment.size() );
        REQUIRE( file.good() );
        file.close();
        m_fragments.pop_front();
    };
}

/// Serialize a registered FlatBuffer log without retaining a binary fixture.
std::string record( size_t             size /**< [in] Text message length. */,
                    flatlogs::logPrioT priority = logPrio::LOG_INFO /**< [in] Record priority. */ )
{
    flatlogs::bufferPtrT buffer;
    text_log::messageT   message( std::string( size, 'x' ) );
    flatlogs::logHeader::createLog<text_log>( buffer, { 1793395297, 123456789 }, message, priority );
    return { buffer.get(), flatlogs::logHeader::totalSize( buffer ) };
}

/// Preserve complete record framing while invalidating its FlatBuffer root.
std::string malformed()
{
    auto bytes = record( 20 );
    std::fill( bytes.begin() + flatlogs::logHeader::headerSize( bytes.data() ), bytes.end(), 0 );
    return bytes;
}
/// \endcond

/** \brief Finite readers omit an incomplete base header, extended header or payload without corruption diagnostics. */
TEST_CASE( "logdump ignores unfinished finite tails", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
#endif
    // clang-format on
    size_t  split    = GENERATE( size_t( 1 ), size_t( 11 ), size_t( 12 ), size_t( 13 ), size_t( 14 ), size_t( 8192 ) );
    auto    complete = record( 20 );
    auto    unfinished = record( GENERATE( size_t( 11000 ), size_t( 70000 ) ) );
    Fixture fixture;
    fixture.write( complete + unfinished.substr( 0, split ) );

    REQUIRE( fixture.m_reader.execute() == 0 );
    const auto output = fixture.m_output.str();
    REQUIRE( std::count( output.begin(), output.end(), '\n' ) == 1 );
    REQUIRE( fixture.m_errors.str().find( "failed verification" ) == std::string::npos );
}

/** \brief Follow mode retains fragments across split base headers, both extended lengths and an 8192-byte append. */
TEST_CASE( "logdump waits for complete appended records", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
#endif
    // clang-format on
    size_t  length   = GENERATE( size_t( 11000 ), size_t( 70000 ) );
    auto    complete = record( length );
    size_t  header   = flatlogs::logHeader::headerSize( complete.data() );
    Fixture fixture;
    fixture.write( complete.substr( 0, 5 ) );
    fixture.m_fragments = { complete.substr( 5, 7 ),
                            complete.substr( 12, 1 ),
                            complete.substr( 13, header - 13 ),
                            complete.substr( header, 8192 - header ),
                            complete.substr( 8192 ),
                            malformed() };
    fixture.follow();

    // The complete malformed sentinel stops follow mode after the valid record has been emitted.
    REQUIRE( fixture.m_reader.execute() == -1 );
    REQUIRE( fixture.m_fragments.empty() );
    const auto output = fixture.m_output.str();
    REQUIRE( std::count( output.begin(), output.end(), '\n' ) == 1 );
    REQUIRE( output.find( std::string( length, 'x' ) ) != std::string::npos );
    REQUIRE( fixture.m_errors.str().find( "at byte=" + std::to_string( complete.size() ) + "." ) != std::string::npos );
}

/** \brief Rotation drains a newly completed old record before following a new file or date directory. */
TEST_CASE( "logdump drains completed tails during rotation", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
#endif
    // clang-format on
    bool    nextDay  = GENERATE( false, true );
    auto    complete = record( 11000 );
    Fixture fixture;
    auto    directory     = fixture.m_directory + "/testapp/2026_10_30";
    auto    nextDirectory = fixture.m_directory + ( nextDay ? "/testapp/2026_10_31" : "/testapp/2026_10_30" );
    std::filesystem::create_directories( directory );
    fixture.m_path = directory + "/testapp_20261030211221630583847.bintel";
    auto nextPath  = nextDirectory +
                    ( nextDay ? "/testapp_20261031211221630583847.bintel" : "/testapp_20261030211222630583847.bintel" );
    fixture.write( complete.substr( 0, 8192 ) );
    fixture.m_reader.m_file.clear();
    fixture.m_reader.m_dir      = fixture.m_directory;
    fixture.m_reader.m_ext      = ".bintel";
    fixture.m_reader.m_prefixes = { "testapp" };
    fixture.m_reader.m_follow   = true;
    bool   appended             = false;
    size_t shortReads           = 0;
    afterShortRead              = [&]()
    {
        REQUIRE( ++shortReads <= 2 );

        if( appended )
        {
            return;
        }

        std::ofstream tail( fixture.m_path, std::ios::binary | std::ios::app );
        tail.write( complete.data() + 8192, complete.size() - 8192 );
        REQUIRE( tail.good() );
        tail.close();
        std::filesystem::create_directories( nextDirectory );
        std::ofstream next( nextPath, std::ios::binary );
        auto          sentinel = malformed();
        next.write( sentinel.data(), sentinel.size() );
        REQUIRE( next.good() );
        appended = true;
    };

    REQUIRE( fixture.m_reader.execute() == -1 );
    REQUIRE( appended );
    const auto output = fixture.m_output.str();
    REQUIRE( std::count( output.begin(), output.end(), '\n' ) == 1 );
    REQUIRE( output.find( std::string( 11000, 'x' ) ) != std::string::npos );
    REQUIRE( fixture.m_errors.str().find( "Log " + nextPath + " failed verification" ) != std::string::npos );
    REQUIRE( fixture.m_errors.str().find( "at byte=0." ) != std::string::npos );
}

/** \brief Complete malformed records fail verification at physical offsets that include filtered records. */
TEST_CASE( "logdump reports physical offsets after filters", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
#endif
    // clang-format on
    auto                 excludedCode = record( 11000 );
    flatlogs::bufferPtrT alias( excludedCode.data(),
                                []( char * )
                                {
                                } );
    flatlogs::logHeader::eventCode( alias, eventCodes::USER_LOG );
    auto    excludedPriority = record( 70000, logPrio::LOG_DEBUG2 );
    Fixture fixture;
    fixture.m_reader.m_codes = { eventCodes::TEXT_LOG };
    fixture.m_reader.m_level = logPrio::LOG_INFO;
    fixture.write( excludedCode + excludedPriority + malformed() );

    REQUIRE( fixture.m_reader.execute() == -1 );
    REQUIRE( fixture.m_output.str().empty() );
    auto offset = excludedCode.size() + excludedPriority.size();
    REQUIRE( fixture.m_errors.str().find( "at byte=" + std::to_string( offset ) + "." ) != std::string::npos );
}

/** \brief Filtered entries wait for unfinished payloads before the next selected record is parsed. */
TEST_CASE( "logdump follows partial filtered records", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
#endif
    // clang-format on
    bool                 byCode   = GENERATE( false, true );
    auto                 filtered = record( 11000, byCode ? logPrio::LOG_INFO : logPrio::LOG_DEBUG2 );
    flatlogs::bufferPtrT alias( filtered.data(),
                                []( char * )
                                {
                                } );
    flatlogs::logHeader::eventCode( alias, byCode ? eventCodes::USER_LOG : eventCodes::TEXT_LOG );
    auto    selected = record( 20 );
    Fixture fixture;
    fixture.m_reader.m_codes = { eventCodes::TEXT_LOG };
    fixture.m_reader.m_level = logPrio::LOG_INFO;
    fixture.write( filtered.substr( 0, 8192 ) );
    fixture.m_fragments = { filtered.substr( 8192 ), selected + malformed() };
    fixture.follow();

    REQUIRE( fixture.m_reader.execute() == -1 );
    REQUIRE( fixture.m_fragments.empty() );
    const auto output = fixture.m_output.str();
    REQUIRE( std::count( output.begin(), output.end(), '\n' ) == 1 );
    auto offset = filtered.size() + selected.size();
    REQUIRE( fixture.m_errors.str().find( "at byte=" + std::to_string( offset ) + "." ) != std::string::npos );
}

/** \brief File discovery still applies the requested number of newest files. */
TEST_CASE( "logdump selects the requested newest files", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::execute();
#endif
    // clang-format on
    size_t  count = GENERATE( size_t( 1 ), size_t( 2 ), size_t( 0 ) );
    Fixture fixture;
    auto    directory = fixture.m_directory + "/testapp/2026_10_30";
    std::filesystem::create_directories( directory );

    for( size_t index = 0; index < 3; ++index )
    {
        fixture.m_path = directory + "/testapp_2026103021122" + std::to_string( index ) + "630583847.bintel";
        fixture.write( record( 20 ) );
    }

    fixture.m_reader.m_file.clear();
    fixture.m_reader.m_dir      = fixture.m_directory;
    fixture.m_reader.m_ext      = ".bintel";
    fixture.m_reader.m_prefixes = { "testapp" };
    fixture.m_reader.m_nfiles   = count;

    REQUIRE( fixture.m_reader.execute() == 0 );
    const auto output = fixture.m_output.str();
    REQUIRE( static_cast<size_t>( std::count( output.begin(), output.end(), '\n' ) ) == ( count ? count : 3 ) );
}

/** \brief Time-span mode counts only complete records and initializes a single-record span consistently. */
TEST_CASE( "logdump time spans exclude incomplete tails", "[logdump]" )
{
    // clang-format off
#ifdef LOGDUMP_TEST_DOXYGEN_REF
    logdump::gettimes( logs );
#endif
    // clang-format on
    size_t  split       = GENERATE( size_t( 0 ), size_t( 1 ), size_t( 13 ), size_t( 8192 ) );
    bool    hasComplete = GENERATE( false, true );
    Fixture fixture;
    fixture.m_reader.m_time = true;
    fixture.write( ( hasComplete ? record( 20 ) : "" ) + record( 11000 ).substr( 0, split ) );

    REQUIRE( fixture.m_reader.execute() == 0 );

    if( hasComplete )
    {
        REQUIRE( fixture.m_output.str().find( "Z 0 1\n" ) != std::string::npos );
    }
    else
    {
        REQUIRE( fixture.m_output.str().empty() );
    }

    REQUIRE( fixture.m_errors.str().find( "failed verification" ) == std::string::npos );
}
} // namespace logdumpTest
} // namespace libXWCTest
