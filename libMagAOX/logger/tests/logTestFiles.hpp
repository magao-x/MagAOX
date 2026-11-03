/** \file logTestFiles.hpp
 * \brief Helpers for logger tests that write real log files of either format version.
 * \ingroup logger_files
 */

#ifndef logger_tests_logTestFiles_hpp
#define logger_tests_logTestFiles_hpp

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <flatlogs/flatlogs.hpp>

#include "../../file/fileTimes.hpp"
#include "../../file/stdFileName.hpp"
#include "../types/text_log.hpp"

namespace libXWCTest
{
namespace loggerTest
{

/** \cond logger_test_helpers
 */

/// Event code that this build doesn't define, for entries of an unsupported log type.
constexpr flatlogs::eventCodeT unknownCode = 65000;

/// One entry to write into a test log file.
struct testEntry
{
    time_t               m_sec;                                          ///< Entry time, seconds.
    long                 m_nsec;                                         ///< Entry time, nanoseconds.
    std::string          m_text;                                         ///< The text_log message.
    flatlogs::eventCodeT m_code{ MagAOX::logger::eventCodes::TEXT_LOG }; ///< Event code written into the entry.
};

/// Build the bytes of one text_log entry of a given version, with its event code replaced if asked.
template <flatlogs::logVersionT version>
std::vector<char> entryBytes( const testEntry &e )
{
    flatlogs::bufferPtrT buf;
    flatlogs::logHeaderT<version>::template createLog<MagAOX::logger::text_log>(
        buf, flatlogs::timespecX( e.m_sec, e.m_nsec ), e.m_text, flatlogs::logPrio::LOG_DEFAULT );

    size_t total = flatlogs::logHeaderT<version>::totalSize( buf.get() );
    if( e.m_code != MagAOX::logger::eventCodes::TEXT_LOG )
    {
        // The event code follows the priority byte, and the version byte for v2.
        size_t codeAt = ( version >= flatlogs::logVersion2 ) ? 2 : 1;
        memcpy( buf.get() + codeAt, &e.m_code, sizeof( e.m_code ) );
        flatlogs::logHeaderT<version>::setCRC( buf.get() );
    }

    return std::vector<char>( buf.get(), buf.get() + total );
}

/// Build the bytes of a log file, recording where each entry starts.
inline std::vector<char> fileBytes( std::vector<size_t>          &offsets, ///< [out] offset of each entry
                                    flatlogs::logVersionT         version, ///< [in] format version of the entries
                                    const std::vector<testEntry> &entries  ///< [in] the entries
)
{
    std::vector<char> bytes;
    offsets.clear();
    for( const testEntry &e : entries )
    {
        offsets.push_back( bytes.size() );
        std::vector<char> eb = ( version == flatlogs::logVersion2 ) ? entryBytes<flatlogs::logVersion2>( e )
                                                                    : entryBytes<flatlogs::logVersion1>( e );
        bytes.insert( bytes.end(), eb.begin(), eb.end() );
    }
    return bytes;
}

/// Write bytes to a standard log file path under a base directory, named for the first entry's time.
/**
 * \returns the standard file name of the written file.
 */
inline MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY>
writeLogFile( const std::string       &basedir, ///< [in] base directory
              const std::string       &app,     ///< [in] app name
              const std::string       &ext,     ///< [in] extension, no '.'
              time_t                   sec,     ///< [in] file time, s
              long                     nsec,    ///< [in] file time, ns
              const std::vector<char> &bytes    ///< [in] file contents
)
{
    std::string fileName, relPath;
    MagAOX::file::fileTimeRelPath( fileName, relPath, app, ext, sec, nsec );
    std::filesystem::create_directories( basedir + '/' + relPath );

    std::string   full = basedir + '/' + relPath + '/' + fileName;
    std::ofstream fout( full, std::ios::binary );
    fout.write( bytes.data(), bytes.size() );
    fout.close();

    return MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY>( full );
}

/// Capture std::cerr for the lifetime of this object.
struct cerrCapture
{
    std::ostringstream m_out;            ///< What was written to std::cerr.
    std::streambuf    *m_old{ nullptr }; ///< The original buffer, restored on destruction.

    cerrCapture() : m_old( std::cerr.rdbuf( m_out.rdbuf() ) )
    {
    }

    ~cerrCapture()
    {
        std::cerr.rdbuf( m_old );
    }

    /// Count the occurrences of a string in the captured output.
    size_t count( const std::string &what ) const
    {
        std::string s = m_out.str();
        size_t      n = 0;
        for( size_t pos = s.find( what ); pos != std::string::npos; pos = s.find( what, pos + what.size() ) )
        {
            ++n;
        }
        return n;
    }
};

/// A temporary directory removed when the object is destroyed.
struct tempDir
{
    std::string m_path; ///< The directory.

    explicit tempDir( const std::string &name )
        : m_path( ( std::filesystem::temp_directory_path() / ( name + "_" + std::to_string( getpid() ) ) ).string() )
    {
        std::filesystem::remove_all( m_path );
        std::filesystem::create_directories( m_path );
    }

    ~tempDir()
    {
        std::filesystem::remove_all( m_path );
    }
};

/** \endcond
 */

} // namespace loggerTest
} // namespace libXWCTest

#endif // logger_tests_logTestFiles_hpp
