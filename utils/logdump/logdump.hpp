/** \file logdump.hpp
 * \brief A simple utility to dump MagAO-X binary logs to stdout.
 *
 * \ingroup logdump_files
 */

#ifndef logdump_hpp
#define logdump_hpp

#include <algorithm>
#include <iostream>
#include <cstring>
#include <filesystem>
#include <limits>

#include <mx/mxError.hpp>
#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
using namespace MagAOX::logger;

using namespace flatlogs;

/** \defgroup logdump logdump: MagAO-X Log Reader
 * \brief Read a MagAO-X binary log file.
 *
 * <a href="../handbook/utils/logdump.html">Utility Documentation</a>
 *
 * \ingroup utils
 *
 */

/** \defgroup logdump_files logdump Files
 * \ingroup logdump
 */

/// An application to dump MagAo-X binary logs to the terminal.
/** \todo document this
 * \todo add config for colors, both on/off and options to change.
 *
 * \ingroup logdump
 */
class logdump : public mx::app::application
{
    typedef mx::verbose::vvv verboseT; ///< File-discovery error reporting policy.

  protected:
    std::string m_dir;  ///< Root directory searched for application logs.
    std::string m_ext;  ///< Selected log filename extension.
    std::string m_file; ///< Explicit input file, or empty for application discovery.

    bool m_time{ false };     ///< Report complete-record time spans instead of messages.
    bool m_jsonMode{ false }; ///< Emit one JSON document per complete selected record.

    unsigned long m_pauseTime{ 250 };        ///< Milliseconds between attempts to read appended data.
    int           m_fileCheckInterval{ 20 }; ///< Read pauses between rotation checks.

    std::vector<std::string> m_prefixes; ///< Application prefixes used during file discovery.

    size_t m_nfiles{ 0 }; ///< Number of newest files to read; zero selects all discovered files.

    bool m_follow{ false }; ///< Wait for appended records and newly discovered log files.

    logPrioT m_level{ logPrio::LOG_DEFAULT }; ///< Maximum selected on-disk priority value.

    std::vector<eventCodeT> m_codes; ///< Selected event codes; empty selects every code.

    /// Print one verified record with the selected priority formatting.
    void printLogBuff( const logPrioT   &lvl /**< [in] Record priority. */,
                       const eventCodeT &ec /**< [in] Record event code. */,
                       const msgLenT    &len /**< [in] Serialized message length. */,
                       bufferPtrT       &logBuff /**< [in] Complete verified record. */ );

    /// Print one verified record as JSON.
    void printLogJson( const msgLenT &len /**< [in] Serialized message length. */,
                       bufferPtrT    &logBuff /**< [in] Complete verified record. */ );

  public:
    /// Register command-line configuration options.
    virtual void setupConfig();

    /// Apply command-line options and log-directory defaults.
    virtual void loadConfig();

    /// Read complete selected records, waiting for unfinished appends in follow mode.
    virtual int execute();

    /// Report the first and last complete record timestamps for each selected file.
    virtual int gettimes( std::vector<std::string> &logs /**< [in] Chronologically ordered input files. */ );
};

void logdump::setupConfig()
{
    config.add( "pauseTime",
                "p",
                "pauseTime",
                argType::Required,
                "",
                "pauseTime",
                false,
                "int",
                "When following, time in milliseconds to pause before checking for new entries." );
    config.add( "fileCheckInterval",
                "",
                "fileCheckInterval",
                argType::Required,
                "",
                "fileCheckInterval",
                false,
                "int",
                "When following, number of pause intervals between checks for new files." );

    config.add( "dir",
                "d",
                "dir",
                argType::Required,
                "",
                "dir",
                false,
                "string",
                "Directory to search for logs. MagAO-X default is normally used." );
    config.add( "ext",
                "e",
                "ext",
                argType::Required,
                "",
                "ext",
                false,
                "string",
                "The file extension of log files.  MagAO-X default is normally used." );
    config.add( "nfiles",
                "n",
                "nfiles",
                argType::Required,
                "",
                "nfiles",
                false,
                "int",
                "Number of log files to dump.  If 0, then all matching files dumped.  Default: 0, 1 if following." );
    config.add( "follow",
                "f",
                "follow",
                argType::True,
                "",
                "follow",
                false,
                "bool",
                "Follow the log, printing new entries as they appear." );
    config.add( "level",
                "L",
                "level",
                argType::Required,
                "",
                "level",
                false,
                "int/string",
                "Minimum log level to dump, either an integer or a string. -1/TELEMETRY [the default], 0/DEFAULT, "
                "1/D1/DBG1/DEBUG2, 2/D2/DBG2/DEBUG1,3/INFO,4/WARNING,5/ERROR,6/CRITICAL,7/FATAL.  Note that only the "
                "mininum unique string is required." );
    config.add( "code",
                "C",
                "code",
                argType::Required,
                "",
                "code",
                false,
                "int",
                "The event code, or vector of codes, to dump.  If not specified, all codes are dumped.  See "
                "logCodes.hpp for a complete list of codes." );
    config.add( "file",
                "F",
                "file",
                argType::Required,
                "",
                "file",
                false,
                "string",
                "A single file to process.  If no / are found in name it will look in the specified directory (or "
                "MagAO-X default)." );
    config.add( "time",
                "T",
                "time",
                argType::True,
                "",
                "time",
                false,
                "bool",
                "time span mode: prints the ISO 8601 UTC timestamps of the first and last entry, the elapsed time in "
                "seconds, and the number of records in the file as a space-delimited string" );
    config.add( "json",
                "J",
                "json",
                argType::True,
                "",
                "json",
                false,
                "bool",
                "JSON mode: emits one JSON document per line for each record in the log" );
}

void logdump::loadConfig()
{
    config( m_pauseTime, "pauseTime" );
    config( m_fileCheckInterval, "fileCheckInterval" );

    // Get default log dir
    std::string tmpstr = mx::sys::getEnv( MAGAOX_env_path );
    if( tmpstr == "" )
    {
        tmpstr = MAGAOX_path;
    }
    m_dir = tmpstr + "/" + MAGAOX_logRelPath;
    ;

    // Now check for config option for dir
    config( m_dir, "dir" );

    m_ext = ".";
    m_ext += MAGAOX_default_logExt;
    config( m_ext, "ext" );
    ///\todo need to check for lack of "." and error or fix

    config( m_file, "file" );

    if( m_file == "" && config.nonOptions.size() < 1 )
    {
        std::cerr << "logdump: need application name. Try logdump -h for help.\n";
    }

    if( m_file == "" && config.nonOptions.size() > 1 )
    {
        std::cerr << "logdump: only one application at a time supported. Try logdump -h for help.\n";
    }

    m_prefixes.resize( config.nonOptions.size() );
    for( size_t i = 0; i < config.nonOptions.size(); ++i )
    {
        m_prefixes[i] = config.nonOptions[i];
    }

    if( config.isSet( "time" ) )
    {
        m_time = true;
    }
    if( config.isSet( "json" ) )
    {
        m_jsonMode = true;
    }

    config( m_follow, "follow" );

    if( m_follow )
    {
        m_nfiles = 1; // default to 1 if follow is set.
    }
    config( m_nfiles, "nfiles" );

    tmpstr = "";
    config( tmpstr, "level" );
    if( tmpstr != "" )
    {
        m_level = logLevelFromString( tmpstr );
    }

    config( m_codes, "code" );

    std::cerr << m_codes.size() << "\n";
}

int logdump::execute()
{

    if( m_file == "" && m_prefixes.size() != 1 )
    {
        return -1; // error message will have been printed in loadConfig.
    }

    std::vector<std::string> subdirs, logs;

    if( m_file != "" )
    {
        if( m_file.find( '/' ) == std::string::npos )
        {

            std::string devName, YYYY, MM, DD, hh, mm, ss, nn;
            mx_error_check_rv( MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, m_file ), -1 );

            m_file = m_dir + '/' + devName + '/' + YYYY + '_' + MM + '_' + DD + '/' + m_file;
        }

        logs.push_back( m_file );
    }
    else
    {
        mx_error_check_rv( mx::ioutils::getFileNames( subdirs, m_dir + "/" + m_prefixes[0], "", "", "" ), -1 );

        if( m_follow )
        {
            mx_error_check_rv( mx::ioutils::getFileNames( logs, subdirs.back(), m_prefixes[0], "", m_ext ), -1 );
        }
        else
        {
            for( size_t d = 0; d < subdirs.size(); ++d )
            {
                // get subdir vectors one at a time and append to build the whole list
                std::vector<std::string> tlogs;
                mx_error_check_rv( mx::ioutils::getFileNames( tlogs, subdirs[d], m_prefixes[0], "", m_ext ), -1 );
                logs.insert( logs.end(), tlogs.begin(), tlogs.end() );
            }
        }
    }

    ///\todo if follow is set, then should nfiles default to 1 unless explicitly set?
    if( m_nfiles == 0 )
    {
        m_nfiles = logs.size();
    }

    if( m_nfiles > logs.size() )
    {
        m_nfiles = logs.size();
    }

    if( m_time )
    {
        return gettimes( logs );
    }

    bool firstRun = true; // for only showing latest entries on first run when following.

    for( size_t i = logs.size() - m_nfiles; i < logs.size(); ++i )
    {
        std::string fname = logs[i];
        FILE       *fin;

        bufferPtrT head( (char *)::operator new( logHeader::maxHeadSize ) );

        bufferPtrT logBuff;

        fin = fopen( fname.c_str(), "rb" );

        if( fin == nullptr )
        {
            std::cerr << "Could not open log " << fname << '\n';
            return -1;
        }

        off_t finSize = mx::ioutils::fileSize( fileno( fin ) );

        std::cerr << fname << "\n";

        off_t totNrd    = 0;
        bool  readError = false;

        // Keep each partial header or payload until its remaining bytes arrive.
        auto readBytes = [&]( char *destination, size_t wanted )
        {
            size_t received = 0;
            int    check    = 0;

            while( received < wanted )
            {
                size_t nrd = fread( destination + received, sizeof( char ), wanted - received, fin );
                received += nrd;
                totNrd += nrd;

                if( received == wanted )
                {
                    return true;
                }

                if( ferror( fin ) )
                {
                    std::cerr << "Error reading log " << fname << '\n';
                    readError = true;
                    return false;
                }

                if( !m_follow || i != logs.size() - 1 )
                {
                    return false;
                }

                firstRun = false;
                std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::milli>( m_pauseTime ) );
                clearerr( fin );
                ++check;

                if( check >= m_fileCheckInterval && m_file.empty() )
                {
                    std::vector<std::string> discovered;

                    if( mx::ioutils::getFileNames( subdirs, m_dir + "/" + m_prefixes[0], "", "", "" ) !=
                            mx::error_t::noerror ||
                        subdirs.empty() ||
                        mx::ioutils::getFileNames( discovered, subdirs.back(), m_prefixes[0], "", m_ext ) !=
                            mx::error_t::noerror )
                    {
                        readError = true;
                        return false;
                    }

                    for( const auto &path : discovered )
                    {
                        if( std::find( logs.begin(), logs.end(), path ) == logs.end() )
                        {
                            logs.push_back( path );
                        }
                    }

                    check = 0;
                }
            }

            return true;
        };

        size_t buffSz = 0;

        while( true )
        {
            const off_t entryOffset = totNrd;

            if( !readBytes( head.get(), logHeader::minHeadSize ) )
            {
                break;
            }

            size_t extra = 0;

            if( logHeader::msgLen0( head ) == logHeader::MAX_LEN0 - 1 )
            {
                extra = sizeof( msgLen1T );
            }
            else if( logHeader::msgLen0( head ) == logHeader::MAX_LEN0 )
            {
                extra = sizeof( msgLen2T );
            }

            if( !readBytes( head.get() + logHeader::minHeadSize, extra ) )
            {
                break;
            }

            logPrioT   lvl = logHeader::logLevel( head );
            eventCodeT ec  = logHeader::eventCode( head );
            msgLenT    len = logHeader::msgLen( head );

            size_t hSz = logHeader::headerSize( head );

            if( len > std::numeric_limits<size_t>::max() - hSz )
            {
                fclose( fin );
                std::cerr << "Log " << fname << " has an invalid length at byte=" << entryOffset << '\n';
                return -1;
            }

            if( hSz + len > buffSz )
            {
                buffSz  = hSz + len;
                logBuff = bufferPtrT( (char *)::operator new( buffSz ) );
            }

            memcpy( logBuff.get(), head.get(), hSz );

            if( !readBytes( logBuff.get() + hSz, len ) )
            {
                break;
            }

            // Consume filtered records completely so a partial append cannot move past the writer.
            if( lvl > m_level ||
                ( !m_codes.empty() && std::find( m_codes.begin(), m_codes.end(), ec ) == m_codes.end() ) )
            {
                continue;
            }

            if( m_follow && firstRun && finSize > 512 && totNrd < finSize - 512 )
            {
                // firstRun = false;
                continue;
            }

            if( !logVerify( ec, logBuff, len ) )
            {
                std::cerr << "Log " << fname << " failed verification on code=" << ec << " at byte=" << entryOffset
                          << ". File possibly corrupt.  Exiting." << std::endl;
                fclose( fin );
                return -1;
            }

            if( m_jsonMode )
            {
                printLogJson( len, logBuff );
            }
            else
            {
                printLogBuff( lvl, ec, len, logBuff );
            }
        }

        fclose( fin );

        if( readError )
        {
            return -1;
        }
    }

    return 0;
}

inline void logdump::printLogBuff( const logPrioT &lvl, const eventCodeT &ec, const msgLenT &len, bufferPtrT &logBuff )
{
    static_cast<void>( len ); // be unused

    if( ec == eventCodes::GIT_STATE )
    {
        if( git_state::repoName( logHeader::messageBuffer( logBuff ) ) == "MagAOX" )
        {
            for( int i = 0; i < 80; ++i )
            {
                std::cout << '-';
            }
            std::cout << "\n\t\t\t\t SOFTWARE RESTART\n";
            for( int i = 0; i < 80; ++i )
            {
                std::cout << '-';
            }
            std::cout << '\n';
        }
    }

    if( lvl < logPrio::LOG_INFO )
    {
        if( lvl == logPrio::LOG_EMERGENCY )
        {
            std::cout << "\033[104m\033[91m\033[5m\033[1m";
        }

        if( lvl == logPrio::LOG_ALERT )
        {
            std::cout << "\033[101m\033[5m";
        }

        if( lvl == logPrio::LOG_CRITICAL )
        {
            std::cout << "\033[41m\033[1m";
        }

        if( lvl == logPrio::LOG_ERROR )
        {
            std::cout << "\033[91m\033[1m";
        }

        if( lvl == logPrio::LOG_WARNING )
        {
            std::cout << "\033[93m\033[1m";
        }

        if( lvl == logPrio::LOG_NOTICE )
        {
            std::cout << "\033[1m";
        }
    }

    logStdFormat( std::cout, logBuff );

    std::cout << "\033[0m";
    std::cout << std::endl;
}

inline void logdump::printLogJson( const msgLenT &len, bufferPtrT &logBuff )
{
    static_cast<void>( len ); // be unused
    logJsonFormat( std::cout, logBuff );
    std::cout << std::endl;
}

int logdump::gettimes( std::vector<std::string> &logs )
{
    for( size_t i = logs.size() - m_nfiles; i < logs.size(); ++i )
    {
        const std::string &fname = logs[i];
        FILE              *fin   = fopen( fname.c_str(), "rb" );

        if( fin == nullptr )
        {
            std::cerr << "Could not open log " << fname << '\n';
            return -1;
        }

        bufferPtrT  head( (char *)::operator new( logHeader::maxHeadSize ) );
        const off_t finSize = mx::ioutils::fileSize( fileno( fin ) );
        timespecX   first, last;
        uint32_t    nRecords  = 0;
        bool        readError = false;

        while( fread( head.get(), sizeof( char ), logHeader::minHeadSize, fin ) == logHeader::minHeadSize )
        {
            size_t extra = 0;

            if( logHeader::msgLen0( head ) == logHeader::MAX_LEN0 - 1 )
            {
                extra = sizeof( msgLen1T );
            }
            else if( logHeader::msgLen0( head ) == logHeader::MAX_LEN0 )
            {
                extra = sizeof( msgLen2T );
            }

            if( fread( head.get() + logHeader::minHeadSize, sizeof( char ), extra, fin ) != extra )
            {
                break;
            }

            msgLenT len      = logHeader::msgLen( head );
            off_t   position = ftello( fin );

            if( position < 0 )
            {
                readError = true;
                break;
            }

            if( position > finSize || len > static_cast<uint64_t>( finSize - position ) )
            {
                break;
            }

            if( fseeko( fin, len, SEEK_CUR ) != 0 )
            {
                readError = true;
                break;
            }

            last = logHeader::timespec( head );

            if( nRecords == 0 )
            {
                first = last;
            }

            ++nRecords;
        }

        readError = readError || ferror( fin );
        fclose( fin );

        if( readError )
        {
            std::cerr << "Error reading log " << fname << '\n';
            return -1;
        }

        if( nRecords == 0 )
        {
            continue;
        }

        double duration = static_cast<double>( last.time_s ) - first.time_s +
                          ( static_cast<double>( last.time_ns ) - first.time_ns ) / 1e9;

        std::cout << fname << " " << first.ISO8601DateTimeStrX() << "Z " << last.ISO8601DateTimeStrX() << "Z "
                  << duration << " " << nRecords << std::endl;
    }

    return 0;
}

#endif // logdump_hpp
