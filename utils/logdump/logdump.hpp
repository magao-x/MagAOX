/** \file logdump.hpp
 * \brief A simple utility to dump MagAO-X binary logs to stdout.
 *
 * \ingroup logdump_files
 */

#ifndef logdump_hpp
#define logdump_hpp

#include <iostream>
#include <cstring>
#include <filesystem>

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
    typedef mx::verbose::vvv verboseT;

  protected:
    std::string m_dir;
    std::string m_ext;
    std::string m_file;

    bool m_time{ false };
    bool m_jsonMode{ false };

    unsigned long m_pauseTime{ 250 }; /// When following, pause time to check for new data. msec. Default is 250 msec.
    int           m_fileCheckInterval{
        20 }; /// When following, number of loops to wait before checking for a new file.  Default is 4.

    std::vector<std::string> m_prefixes;

    size_t m_nfiles{ 0 }; /// Number of files to dump.  Default is 0, unless following then the default is 1.

    bool m_follow{ false };

    bool m_keepGoing{ false }; ///< If true, carry on past damaged entries and report each one.

    logPrioT m_level{ logPrio::LOG_DEFAULT };

    std::vector<eventCodeT> m_codes;

    /// Print one entry as standard text.
    void printLogBuff( const parsedEntry &entry /**< [in] the decoded entry to print */ );

    /// Print one entry as JSON.
    void printLogJson( const parsedEntry &entry /**< [in] the decoded entry to print */ );

    /// Check a decoded entry.
    /** A v2 entry must pass its checksum. Any entry must fit.
     * A v1 entry of a type this build doesn't know must also have a defined priority and a time not before
     * the previous good entry in the file.
     *
     * \returns an empty string if the entry is good, or the reason it isn't ("structure" or "checksum").
     */
    static std::string checkEntry( const parsedEntry &entry,  ///< [in] the decoded entry
                                   bool havePrev,             ///< [in] true if an earlier entry in the file was good
                                   const timespecX &prevTs    ///< [in] the time of earlier entry, if havePrev
    );

    /// Search file for the place to recover at after a damaged entry.
    /** Reads the file from the damaged entry to the end into memory and searches it with the same recovery as
     * logMapRecover.
     *
     * \returns the offset of the place to carry on, from \p pos, or flatlogs::entryNotFound if there is none.
     */
    static size_t findRestart( FILE *fin,                    ///< [in] the open file
                               logVersionT version,           ///< [in] the file's format version
                               off_t pos,                    ///< [in] the offset of the damaged entry
                               const std::string &reason,    ///< [in] why the entry is damaged
                               bool havePrev,                ///< [in] true if an earlier entry in the file was good
                               const timespecX &prevTs,      ///< [in] the time of that entry, if havePrev
                               bool *endsBeforeEOF = nullptr ///< [out] [optional] true if the entry found at the
                                                                    /// restart point ends before the end of the file
    );

  public:
    /// Result of reading the next entry in a file.
    enum class readResult
    {
        entry,      ///< An entry's bytes were read.
        incomplete, ///< The file ends part way through an entry.
        end         ///< There is nothing more to read.
    };

    /// Read the next whole entry from a file into a buffer.
    /** Reads the header to get the entry's size, then reads the rest of the entry.
     * If the file ends part way through and entry, the file position is returned to the start of the entry.
     * A length so large that the size wraps around results in an entry of just the header,
     * which then fails to decode.
     *
     * \returns readResult::entry, readResult::incomplete or readResult::end.
     */
    static readResult readEntry( size_t &entrySize,            ///< [out] the size of the entry
                                 std::vector<char> &buff,      ///< [in,out] holds the entry
                                 FILE *fin,                    ///< [in] the open file
                                 logVersionT version           ///< [in] the file's version
    );

    virtual void setupConfig();

    virtual void loadConfig();

    virtual int execute();

    virtual int gettimes( std::vector<std::string> &logs );
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
    config.add( "keepGoing",
                "k",
                "keepGoing",
                argType::True,
                "",
                "keepGoing",
                false,
                "bool",
                "Continue past damaged entries and report each one, instead of stopping at the first.  If damaged entries "
                "found, the exist status is still an error.  Can't be used with --follow." );
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
        m_time = true;
    if( config.isSet( "json" ) )
        m_jsonMode = true;

    config( m_follow, "follow" );

    if( config.isSet( "keepGoing" ) )
        m_keepGoing = true;

    if( m_follow )
        m_nfiles = 1; // default to 1 if follow is set.
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

logdump::readResult logdump::readEntry( size_t &entrySize, std::vector<char> &buff, FILE *fin, logVersionT version )
{
    entrySize = 0;

    off_t start = ftello( fin );

    size_t minHead = entryMinHeadSize( version );
    if( minHead == 0 )
    {
        return readResult::end;
    }

    if( buff.size() < static_cast<size_t>( logHeaderV2::maxHeadSize ) )
    {
        buff.resize( logHeaderV2::maxHeadSize );
    }

    size_t nrd = fread( buff.data(), sizeof( char ), minHead, fin );
    if( nrd == 0 )
    {
        return readResult::end;
    }

    auto backUp = [&]()
    {
        fseeko( fin, start, SEEK_SET );
        return readResult::incomplete;
    };
    
    // A short read means the entry hasn't finished writing (or the file was truncated).
    if( nrd < minHead )
    {
        return backUp();
    }

    size_t hSz = entryHeaderSize( buff.data(), version );
    if( hSz > minHead )
    {
        nrd = fread( buff.data() + minHead, sizeof( char ), hSz - minHead, fin );
        if( nrd < hSz - minHead )
        {
            return backUp();
        }
    }

    size_t total = entryTotalSize( buff.data(), version );
    if( total < hSz )
    {
        // Assuming length is so large that the size wrapped around;
        // return just the header, which will fail to decode.
        entrySize = hSz;
        return readResult::entry;
    }

    if( total > buff.size() )
    {
        // Only grow the buffer to a size the file can hold, since a damaged length can claim anything.
        off_t fsz = mx::ioutils::fileSize( fileno( fin ) );
        if( fsz < start || static_cast<size_t>( fsz - start ) < total )
        {
            return backUp();
        }
        buff.resize( total );
    }

    nrd = fread( buff.data() + hSz, sizeof( char ), total - hSz, fin );
    if( nrd < total - hSz )
    {
        return backUp();
    }

    entrySize = total;
    return readResult::entry;
}

std::string logdump::checkEntry( const parsedEntry &entry, bool havePrev, const timespecX &prevTs )
{
    switch( entry.status )
    {
    case entryStatus::checksumOk:
    case entryStatus::noChecksum:
        break;
    case entryStatus::checksumMismatch:
        return "checksum";
    default:
        return "structure";
    }

    // Since v1 has no checksum, an unknown event code can be a sign of problems after a damaged length,
    // so a v1 entry with an unknown event code is only accepted if it has a valid priority and chronologically
    // comes after the previous entry.
    if( entry.status == entryStatus::noChecksum && !logCodeValid( entry.eventCode ) )
    {
        timespecX ts = entry.ts;
        if( !logMapPriorityValid( entry.prio ) || ( havePrev && ts < prevTs ) )
        {
            return "structure";
        }
    }

    return "";
}

size_t logdump::findRestart( FILE *fin,
                             logVersionT version,
                             off_t pos,
                             const std::string &reason,
                             bool havePrev,
                             const timespecX &prevTs,
                             bool *endsBeforeEOF )
{
    if( endsBeforeEOF )
    {
        *endsBeforeEOF = false;
    }

    off_t fsz = mx::ioutils::fileSize( fileno( fin ) );
    if( fsz <= pos )
    {
        return entryNotFound;
    }

    std::vector<char> rest( fsz - pos );

    fseeko( fin, pos, SEEK_SET );
    size_t nrd = fread( rest.data(), sizeof( char ), rest.size(), fin );
    rest.resize( nrd );

    logStep failed;
    failed.m_entry = parseEntry( rest.data(), rest.size(), version );

    // An entry that passes its checksum ends where its length says.
    size_t restart = failed.m_entry.entrySize;
    if( !( reason == "message" && failed.m_entry.status == entryStatus::checksumOk ) )
    {
        restart = logMapRecover( rest.data(), rest.size(), 0, version, failed, havePrev ? &prevTs : nullptr );
    }

    if( endsBeforeEOF && restart != entryNotFound && restart < rest.size() )
    {
        parsedEntry found = parseEntry( rest.data() + restart, rest.size() - restart, version );
        *endsBeforeEOF = ( found.entrySize > 0 && restart + found.entrySize < rest.size() );
    }

    return restart;
}

int logdump::execute()
{

    if( m_file == "" && m_prefixes.size() != 1 )
        return -1; // error message will have been printed in loadConfig.

    if( m_keepGoing && m_follow )
    {
        std::cerr << "logdump: --keepGoing can't be used with --follow.\n";
        return -1;
    }

    std::vector<std::string> subdirs, logs;

    if( m_file != "" )
    {
        if( m_file.find( '/' ) == std::string::npos )
        {

            std::string devName, YYYY, MM, DD, hh, mm, ss, nn;
            mx_error_check_rv(MagAOX::file::parseFilePath( devName, YYYY, MM, DD, hh, mm, ss, nn, m_file ),-1);

            m_file = m_dir + '/' + devName + '/' + YYYY + '_' + MM + '_' + DD + '/' + m_file;
        }

        logs.push_back( m_file );
    }
    else
    {
        mx_error_check_rv(mx::ioutils::getFileNames(subdirs, m_dir + "/" + m_prefixes[0], "", "", "" ),-1);

        if( m_follow )
        {
            mx_error_check_rv( listLogFiles<verboseT>( logs, subdirs.back(), m_prefixes[0], m_ext ), -1 );
        }
        else
        {
            for( size_t d = 0; d < subdirs.size(); ++d )
            {
                // get subdir vectors one at a time and append to build the whole list
                std::vector<std::string> tlogs;
                mx_error_check_rv( listLogFiles<verboseT>( tlogs, subdirs[d], m_prefixes[0], m_ext ), -1 );
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
        m_nfiles = logs.size();

    if( m_time )
    {
        return gettimes( logs );
    }

    bool firstRun = true; // for only showing latest entries on first run when following.

    bool anyDamaged = false; // with --keepGoing, whether any entry was damaged

    std::vector<char> buff; // holds one entry at a time, reused

    for( size_t i = logs.size() - m_nfiles; i < logs.size(); ++i )
    {
        std::string fname = logs[i];
        FILE       *fin;

        // Each file is read by the version its extension names.
        logVersionT version = logVersionForFile( fname );

        fin = fopen( fname.c_str(), "rb" );

        if( !fin )
        {
            std::cerr << "logdump: could not open " << fname << "\n";
            return -1;
        }

        //--> get size here!!
        off_t finSize = mx::ioutils::fileSize( fileno( fin ) );

        std::cerr << fname << "\n";

        off_t pos = 0; // offset of the next entry

        bool      havePrev = false; // whether an earlier entry in this file was good
        timespecX prevTs;           // the time of that entry

        std::map<eventCodeT, size_t> unknownTypes; // count of entries of each type this build doesn't know

        while( true )
        {
            size_t     entrySize = 0;
            readResult rr        = readEntry( entrySize, buff, fin, version );

            if( rr != readResult::entry )
            {
                // If we're following and on the last log file, wait for more to show up.
                if( m_follow == true && i == logs.size() - 1 )
                {
                    int  check   = 0;
                    bool newFile = false;
                    firstRun     = false; // from now on we show all logs

                    // The file's size at the last search after an incomplete entry, or -1 if none yet.
                    off_t sizeAtSearch = -1;

                    while( rr != readResult::entry )
                    {
                        std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::milli>( m_pauseTime ) );
                        clearerr( fin );
                        rr = readEntry( entrySize, buff, fin, version );
                        if( rr == readResult::entry )
                            break;

                        ++check;
                        if( check >= m_fileCheckInterval )
                        {
                            // An entry that remains incomplete may have a damaged length. If good entries follow
                            // it, take it as confirmation of damaged length and stop.
                            // Search once, then only when the file has grown.
                            off_t sizeNow = mx::ioutils::fileSize( fileno( fin ) );
                            if( rr == readResult::incomplete && sizeNow != sizeAtSearch )
                            {
                                sizeAtSearch = sizeNow;

                                // Only a restart entry that ends before the current end of the file counts.
                                bool   endsBeforeEOF = false;
                                size_t restart =
                                    findRestart( fin, version, pos, "structure", havePrev, prevTs, &endsBeforeEOF );
                                fseeko( fin, pos, SEEK_SET );
                                if( restart != entryNotFound && endsBeforeEOF )
                                {
                                    std::cerr << "Log " << fname << " failed verification at byte=" << pos
                                              << " (reason: structure). File possibly corrupt.  Exiting." << std::endl;
                                    return -1;
                                }
                            }

                            // Check if a new sub-directory exists now.
                            size_t old_subdirs_sz = subdirs.size();
                            mx_error_check_rv(mx::ioutils::getFileNames( subdirs, m_dir + "/" + m_prefixes[0], "", "", "" ),-1);
                            if( subdirs.size() > old_subdirs_sz )
                            {
                                // new subdirs(s) detected;
                                newFile = true;
                                break;
                            }

                            // Check if a new file exists now.
                            size_t old_logs_sz = logs.size();

                            mx_error_check_rv( listLogFiles<verboseT>( logs, subdirs.back(), m_prefixes[0], m_ext ),
                                               -1 );

                            if( logs.size() > old_logs_sz )
                            {
                                // new file(s) detected;
                                newFile = true;
                                break;
                            }
                            check = 0;
                        }
                    }

                    if( newFile )
                    {
                        if( rr == readResult::incomplete )
                        {
                            std::cerr << "logdump: " << fname << ": incomplete final entry at byte=" << pos
                                      << " (a newer file has started)\n";
                        }
                        break;
                    }
                }
                else
                {
                    if( rr == readResult::incomplete )
                    {
                        // Either the file is still being written, or the length of this entry is damaged.
                        // If good entries follow it take it as confirmation of damage.
                        size_t restart = findRestart( fin, version, pos, "structure", havePrev, prevTs );
                        if( restart == entryNotFound )
                        {
                            std::cerr << "logdump: " << fname << ": incomplete final entry at byte=" << pos
                                      << " (the file may still be being written)\n";
                            break;
                        }

                        if( !m_keepGoing )
                        {
                            std::cerr << "Log " << fname << " failed verification at byte=" << pos
                                      << " (reason: structure). File possibly corrupt.  Exiting." << std::endl;
                            return -1;
                        }

                        anyDamaged = true;
                        std::cerr << "Log " << fname << ": skipped damaged entry at byte=" << pos
                                  << " (reason: structure); resumed at byte=" << pos + restart << "\n";
                        pos += restart;
                        fseeko( fin, pos, SEEK_SET );
                        continue;
                    }

                    break;
                }
            }

            // When starting to follow earlier entries (cutoff set at 512 bytes) are skipped unchecked.
            // A damaged length in the skipped portion still offsets the position and is caught at the first 
            // entry that is checked.
            if( m_follow && firstRun && finSize > 512 && pos + static_cast<off_t>( entrySize ) < finSize - 512 )
            {
                pos += entrySize;
                continue;
            }

            parsedEntry entry = parseEntry( buff.data(), entrySize, version );

            std::string reason = checkEntry( entry, havePrev, prevTs );

            bool printIt = false;
            if( reason == "" )
            {
                havePrev = true;
                prevTs   = entry.ts;

                printIt = ( entry.prio <= m_level );

                if( printIt && m_codes.size() > 0 )
                {
                    bool found = false;
                    for( size_t c = 0; c < m_codes.size(); ++c )
                    {
                        if( m_codes[c] == entry.eventCode )
                        {
                            found = true;
                            break;
                        }
                    }

                    printIt = found;
                }

                // The message check, for entries of known types that are to be printed.
                if( printIt && logCodeValid( entry.eventCode ) &&
                    !logVerify( entry.eventCode, entryMessage( entry ), entry.msgLen ) )
                {
                    reason = "message";
                }
            }

            if( reason != "" )
            {
                bool decoded = ( entry.status == entryStatus::checksumOk || entry.status == entryStatus::noChecksum ||
                                 entry.status == entryStatus::checksumMismatch );

                if( !m_keepGoing )
                {
                    std::cerr << "Log " << fname << " failed verification";
                    if( decoded )
                    {
                        std::cerr << " on code=" << entry.eventCode;
                    }
                    std::cerr << " at byte=" << pos << " (reason: " << reason << "). File possibly corrupt.  Exiting."
                              << std::endl;
                    return -1;
                }

                anyDamaged = true;

                size_t restart = findRestart( fin, version, pos, reason, havePrev, prevTs );

                std::cerr << "Log " << fname << ": skipped damaged entry";
                if( decoded )
                {
                    std::cerr << " (code=" << entry.eventCode << ")";
                }
                std::cerr << " at byte=" << pos << " (reason: " << reason << ")";

                if( restart == entryNotFound )
                {
                    std::cerr << "; no further good entry in this file\n";
                    break;
                }

                std::cerr << "; resumed at byte=" << pos + restart << "\n";
                pos += restart;
                fseeko( fin, pos, SEEK_SET );
                continue;
            }

            pos += entrySize;

            if( !printIt )
            {
                continue;
            }

            if( !logCodeValid( entry.eventCode ) )
            {
                ++unknownTypes[entry.eventCode];

                // Not included in JSON, since placeholder has no data.
                if( m_jsonMode )
                {
                    continue;
                }
            }

            if( m_jsonMode )
            {
                printLogJson( entry );
            }
            else
            {
                printLogBuff( entry );
            }
        }

        for( const auto &code : unknownTypes )
        {
            std::cerr << "logdump: " << fname << ": " << code.second << " entries of log type " << code.first
                      << ", which this build doesn't know, ";
            if( m_jsonMode )
            {
                std::cerr << "were left out of the JSON output\n";
            }
            else
            {
                std::cerr << "were printed without their contents\n";
            }
        }

        fclose( fin );
    }

    if( anyDamaged )
    {
        return -1;
    }

    return 0;
}

inline void logdump::printLogBuff( const parsedEntry &entry )
{
    logPrioT lvl = entry.prio;
    eventCodeT ec  = entry.eventCode;

    if( ec == eventCodes::GIT_STATE )
    {
        if( git_state::repoName( entryMessage( entry ) ) == "MagAOX" )
        {
            for( int i = 0; i < 80; ++i )
                std::cout << '-';
            std::cout << "\n\t\t\t\t SOFTWARE RESTART\n";
            for( int i = 0; i < 80; ++i )
                std::cout << '-';
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

    if( logCodeValid( ec ) )
    {
        logStdFormat( std::cout, entry );
    }
    else
    {
        // Keep usual prefix.
        timespecX ts = entry.ts;
        std::cout << ts.ISO8601DateTimeStrX() << " " << priorityString( lvl ) << " Unknown log type: " << ec;
    }

    std::cout << "\033[0m";
    std::cout << std::endl;
}

inline void logdump::printLogJson( const parsedEntry &entry )
{
    logJsonFormat( std::cout, entry );
    std::cout << std::endl;
}

int logdump::gettimes( std::vector<std::string> &logs )
{
    for( size_t i = logs.size() - m_nfiles; i < logs.size(); ++i )
    {
        std::string fname = logs[i];
        FILE       *fin;

        // Files are read by the version indicated by their extension.
        logVersionT version = logVersionForFile( fname );
        size_t      minHead = entryMinHeadSize( version );

        std::vector<char> head( logHeaderV2::maxHeadSize );

        fin = fopen( fname.c_str(), "rb" );

        if( !fin )
        {
            std::cerr << "logdump: could not open " << fname << "\n";
            return -1;
        }

        //--> get size here!!
        // off_t finSize = mx::ioutils::fileSize( fileno(fin) );

        off_t totNrd = 0;

        // size_t buffSz = 0;

        // Read the rest of the header after the first minHead bytes, and return the size of what follows.
        auto finishHeader = [&]() -> size_t
        {
            size_t hSz = entryHeaderSize( head.data(), version );
            if( hSz > minHead )
            {
                size_t nrdh = fread( head.data() + minHead, sizeof( char ), hSz - minHead, fin );
                static_cast<void>( nrdh );
            }
            return entryTotalSize( head.data(), version ) - hSz;
        };

        auto headerTime = [&]() -> timespecX
        {
            return ( version == logVersion2 ) ? logHeaderV2::timespec( head.data() )
                                              : logHeaderV1::timespec( head.data() );
        };

        // Read first header

        int nrd;

        ///\todo check for errors on all reads . . .

        // Read next header
        nrd = fread( head.data(), sizeof( char ), minHead, fin );
        if( nrd == 0 )
        {
            std::cerr << "got no header\n";
            return 0;
        }

        size_t    rest = finishHeader();
        timespecX ts0  = headerTime();

        uint32_t nRecords = 1;
        fseek( fin, rest, SEEK_CUR );

        timespecX ts;

        while( !feof( fin ) ) //<--This should be an exit condition controlled by loop logic, not feof.
        {
            int nrd;

            // Read next header
            nrd = fread( head.data(), sizeof( char ), minHead, fin );
            if( nrd == 0 )
            {
                break;
            }
            nRecords += 1;

            // We got here without any data, probably means time to get a new file.
            if( nrd == 0 )
                break;

            totNrd += nrd;

            rest = finishHeader();
            ts   = headerTime();

            fseek( fin, rest, SEEK_CUR );
        }

        fclose( fin );

        double t0 = ts0.time_s + ts0.time_ns / 1e9;
        double t  = ts.time_s + ts.time_ns / 1e9;

        std::cout << fname << " " << ts0.ISO8601DateTimeStrX() << "Z " << ts.ISO8601DateTimeStrX() << "Z " << t - t0
                  << " " << nRecords << std::endl;
    }

    return 0;
}

#endif // logdump_hpp
