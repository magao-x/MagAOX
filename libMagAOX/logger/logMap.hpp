/** \file logMap.hpp
 * \brief Declares and defines the logMap class and related classes.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_files
 *
 * History:
 * - 2020-01-02 created by JRM
 */

#ifndef logger_logMap_hpp
#define logger_logMap_hpp

#include <mx/sys/timeUtils.hpp>
using namespace mx::sys::tscomp;

#include <mx/ioutils/fileUtils.hpp>

#include <vector>
#include <map>

#include <flatlogs/flatlogs.hpp>
#include "../file/stdFileName.hpp"
#include "generated/logCodes.hpp"

#ifndef DEBUG_CRUMB
    #ifdef DEBUG
        #define DEBUG_CRUMB( msg )                                                                                     \
            {                                                                                                          \
                std::cerr << msg << '(' << __FILE__ << ' ' << __LINE__ << "\n";                                        \
            }
    #else
        #define DEBUG_CRUMB( msg )
    #endif
#endif

namespace MagAOX
{
namespace logger
{

/// Format a flatlogs timestamp for debug messages.
inline std::string logMapDebugTime( flatlogs::timespecX ts /**< [in] timestamp to format */ )
{
    std::string tstamp;
    ts.timeStamp( tstamp );
    return tstamp + " (" + std::to_string( ts.time_s ) + "." + std::to_string( ts.time_ns ) + ")";
}

/// Check whether a flatlog priority value is one of the defined on-disk priorities.
inline bool logMapPriorityValid( flatlogs::logPrioT prio /**< [in] priority value to check */ )
{
    return ( prio >= flatlogs::logPrio::LOG_EMERGENCY && prio <= flatlogs::logPrio::LOG_DEBUG2 ) ||
           prio == flatlogs::logPrio::LOG_TELEM;
}

/// Check whether a flatlog entry's header and claimed size fit in the loaded buffer.
inline bool logMapEntryExtentValid( size_t &totalSize, /**< [out] total entry size claimed by the header */
                                    char   *buffer,    /**< [in] candidate flatlog entry */
                                    char   *bufferEnd  /**< [in] one past the loaded buffer */
)
{
    totalSize = 0;
    if( buffer == nullptr || bufferEnd == nullptr || buffer >= bufferEnd )
    {
        return false;
    }

    size_t remaining = bufferEnd - buffer;
    if( remaining < static_cast<size_t>( flatlogs::logHeader::minHeadSize ) )
    {
        return false;
    }

    size_t headerSize = flatlogs::logHeader::headerSize( buffer );
    if( headerSize == 0 || headerSize > remaining )
    {
        return false;
    }

    totalSize = flatlogs::logHeader::totalSize( buffer );
    return totalSize > 0 && totalSize <= remaining;
}

/// Check whether a flatlog entry has a sane envelope for length-chain traversal.
inline bool logMapEntrySane( size_t              &totalSize, /**< [out] total entry size claimed by the header */
                             char                *buffer,    /**< [in] candidate flatlog entry */
                             char                *bufferEnd, /**< [in] one past the loaded buffer */
                             flatlogs::timespecX *minTs = 0  /**< [in] optional minimum plausible timestamp */
)
{
    if( !logMapEntryExtentValid( totalSize, buffer, bufferEnd ) )
    {
        return false;
    }

    if( !logMapPriorityValid( flatlogs::logHeader::logLevel( buffer ) ) )
    {
        return false;
    }

    if( eventCodeName( flatlogs::logHeader::eventCode( buffer ) ) == "unknown event code" )
    {
        return false;
    }

    return minTs == 0 || !( flatlogs::logHeader::timespec( buffer ) < *minTs );
}

/// Byte-scan forward to the next sane flatlog envelope.
inline char *logMapResync( char                *buffer,    /**< [in] failed flatlog entry */
                           char                *bufferEnd, /**< [in] one past the loaded buffer */
                           flatlogs::timespecX *minTs = 0  /**< [in] optional minimum plausible timestamp */
)
{
    if( buffer == nullptr || bufferEnd == nullptr )
    {
        return nullptr;
    }

    for( char *candidate = buffer + 1; candidate < bufferEnd; ++candidate )
    {
        char                *chain      = candidate;
        size_t               chainLinks = 0;
        flatlogs::timespecX  chainMinTs;
        flatlogs::timespecX *chainMinTsPtr = minTs;
        while( chain < bufferEnd )
        {
            size_t totalSize = 0;
            if( !logMapEntrySane( totalSize, chain, bufferEnd, chainMinTsPtr ) )
            {
                break;
            }

            ++chainLinks;
            if( chainLinks >= 3 )
            {
                return candidate;
            }

            chainMinTs    = flatlogs::logHeader::timespec( chain );
            chainMinTsPtr = &chainMinTs;
            chain += totalSize;
        }

        if( chain == bufferEnd && chainLinks > 0 )
        {
            return candidate;
        }
    }

    return nullptr;
}

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Structure to hold a log file in memory, tracking when a new file needs to be opened.
struct logInMemory
{
    typedef XWC_DEFAULT_VERBOSITY verboseT;

    /// Source file range in the loaded log buffer.
    struct loadedFile
    {
        size_t      m_begin{ 0 }; ///< First byte offset from this source file.
        size_t      m_end{ 0 };   ///< One past the last byte offset from this source file.
        std::string m_name;       ///< Full source file path.
    };

    std::vector<char> m_memory; ///< The buffer holding the log data.

    flatlogs::timespecX m_startTime{ 0, 0 }; ///< Earliest timestamp covered by the loaded buffer.
    flatlogs::timespecX m_endTime{ 0, 0 };   ///< Latest timestamp covered by the loaded buffer.

    std::vector<loadedFile> m_loadedFiles; ///< Source-file provenance ranges for the loaded buffer.

    size_t m_recoverableErrors{ 0 }; ///< Count of recoverable log parsing errors encountered while loading.

    std::string m_reportPrefix; ///< Optional prefix for user-facing recoverable log parsing reports.

    /// Load one flatlog file into memory.
    int loadFile( file::stdFileName<verboseT> const &lfn /**< [in] standard file name to load */ );

    /// Get the source file name for a log entry pointer.
    std::string sourceFile( char *log /**< [in] pointer to a log entry in m_memory */ ) const;

    /// Get the byte offset in the source file for a log entry pointer.
    size_t sourceOffset( char *log /**< [in] pointer to a log entry in m_memory */ ) const;
};

/// Map of log entries by application name, mapping both to files and to loaded buffers.
template <class verboseT = XWC_DEFAULT_VERBOSITY>
struct logMap
{

    typedef file::stdSubDir<verboseT>   stdSubDirT;
    typedef file::stdFileName<verboseT> stdFileNameT;

    /// The app-name to file-name map type, for sorting the input files by application
    typedef std::map<std::string, std::set<stdFileNameT, file::compStdFileName<stdFileNameT>>> appToFileMapT;

    /// The app-name to buffer map type, for looking up the currently loaded logs for a given app.
    typedef std::map<std::string, logInMemory> appToBufferMapT;

    int m_searchDaySpan{ 100 }; ///< Maximum number of days to search for files in the past/future.

    appToFileMapT m_appToFileMap; ///< Available log files grouped by app/device name.

    appToBufferMapT m_appToBufferMap; ///< Loaded log buffers grouped by app/device name.

    std::string m_reportPrefix; ///< Optional prefix for user-facing recoverable log parsing reports.

    /// Record one recoverable log-processing error.
    void recordRecoverableError( const std::string &appName /**< [in] app/device associated with the error */ );

    /// Get the number of recoverable errors encountered in loaded logs.
    size_t recoverableErrors() const;

    /// Add a list of files to the file map
    /** This is a worker function for loadAppToFileMap
     *
     * \returns mx::error_t::noerror on success
     * \returns mx::error_t::std_exception if a std::exception is thrown
     */
    mx::error_t addFileListToFileMap( const std::string              &dev,   /**< [in] the device name to add*/
                                      const std::vector<std::string> &flist, /**< [in] the file list from which to
                                                                                       add files*/
                                      size_t n0,                             /**< [in] the first entry in the file list
                                                                                       to add*/
                                      size_t nf                              /**< [in] one past the last entry in the
                                                                                       file list to add, e.g. flist.size()*/
    );

    /// Get log file names in a directory and distribute them into the map by app-name
    /** Finding no logs is not reported as an error (no exception is thrown).  You must check
     *  the size of m_appToFileMap to check if any files were found.
     *
     */
    mx::error_t loadAppToFileMap( const std::string &dir,        /**< [in] the directory to search for files
                                                                           (contains the dev/YYYY_MM_DD subdirs)*/
                                  const std::string  &dev,       ///< [in] the device name to search for logs of
                                  const std::string  &ext,       ///< [in] the extension to search for
                                  const stdFileNameT &firstFile, ///< [in] the first file that needs coverage
                                  const stdFileNameT &lastFile   ///< [in] the last file that needs coverage
    );

    /// Get the log for an event code which is the first prior to the supplied time
    int getPriorLog( char                      *&logBefore, ///< [out] pointer to the first byte of the prior log entry
                     const std::string          &appName,   ///< [in] the name of the app specifying which log to search
                     const flatlogs::eventCodeT &ev,        ///< [in] the event code to search for
                     const flatlogs::timespecX  &ts,        ///< [in] the timestamp to be prior to
                     char                       *hint = 0   /**< [in] [optional] a hint specifying
                                                                      where to start searching.  If null
                                                                      search starts at beginning.*/
    );

    /// Get the next log with the same event code which is after the supplied time
    int getNextLog( char             *&logAfter,   ///< [out] pointer to the first byte of the prior log entry
                    char              *logCurrent, ///< [in] The log to start from
                    const std::string &appName     ///< [in] the name of the app specifying which log to search
    );

    /// Get the nearest loaded logs around the current search point.
    int getNearestLogs( flatlogs::bufferPtrT &logBefore, /**< [out] log before the search time */
                        flatlogs::bufferPtrT &logAfter,  /**< [out] log after the search time */
                        const std::string    &appName    /**< [in] app/device name to search */
    );

    /// Load files that cover a requested timestamp for an app.
    int loadFiles( const std::string         &appName,  /**< [in] MagAO-X app name for which to load files */
                   const flatlogs::timespecX &startTime /**< [in] timestamp that must be covered by loaded logs */
    );
};

template <class verboseT>
void logMap<verboseT>::recordRecoverableError( const std::string &appName )
{
    m_appToBufferMap[appName].m_recoverableErrors++;
}

template <class verboseT>
size_t logMap<verboseT>::recoverableErrors() const
{
    size_t nErrors = 0;
    for( const auto &appBuffer : m_appToBufferMap )
    {
        nErrors += appBuffer.second.m_recoverableErrors;
    }

    return nErrors;
}

template <class verboseT>
mx::error_t logMap<verboseT>::addFileListToFileMap( const std::string              &dev,
                                                    const std::vector<std::string> &flist,
                                                    size_t                          n0,
                                                    size_t                          nf )
{
    try
    {
        // clang-format off
        #ifdef XWCTEST_LOGMAP_AFLTFM_XWCE
            throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
        #endif

        #ifdef XWCTEST_LOGMAP_AFLTFM_BADALL
            throw std::bad_alloc; // LCOV_EXCL_LINE
        #endif

        #ifdef XWCTEST_LOGMAP_AFLTFM_EXCEPTION
            throw std::exception; // LCOV_EXCL_LINE
        #endif
        // clang-format on

        for( size_t n = n0; n < nf; ++n )
        {
            file::stdFileName<verboseT> sfn( flist[n] );

            DEBUG_CRUMB( "logMap add candidate dev=" + dev + " file=" + flist[n] );

            if( !sfn.valid() ) // this is just not a standard file name.
            {
                DEBUG_CRUMB( "logMap skip invalid dev=" + dev + " file=" + flist[n] );
                continue;
            }

            if( sfn.appName() != dev ) // this is just a different app
            {
                DEBUG_CRUMB( "logMap skip app mismatch dev=" + dev + " file=" + sfn.fullName() );
                continue;
            }

            m_appToFileMap[dev].insert( sfn );
            DEBUG_CRUMB( "logMap added dev=" + dev + " file=" + sfn.fullName() +
                         " timestamp=" + logMapDebugTime( sfn.timestamp() ) );
        }

        return mx::error_t::noerror;
    }
    catch( const xwcException &e )
    {
        std::throw_with_nested( xwcException( "adding file to map" ) );
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "adding file to map" ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           std::string( "adding file to map:" ) + e.what() );
    }
}

template <class verboseT>
mx::error_t logMap<verboseT>::loadAppToFileMap( const std::string                 &dir,
                                                const std::string                 &dev,
                                                const std::string                 &ext,
                                                const file::stdFileName<verboseT> &firstFile,
                                                const file::stdFileName<verboseT> &lastFile )
{
    mx::error_t errc;

    bool isdir = mx::ioutils::dir_exists_is( dir, errc );

    mx_error_check_code( errc );

    if( !isdir )
    {
        return mx::error_report<verboseT>( mx::error_t::dirnotfound, dir + " does not exist" );
    }

    // Timestamps for defining the previous log and the following log
    flatlogs::timespecX prevts = firstFile.timestamp( &errc );
    mx_error_check_code( errc );

    prevts.time_s -= 60; // Move 60 seconds in future.  This is a config setting

    flatlogs::timespecX follts = lastFile.timestamp( &errc );
    mx_error_check_code( errc );

    follts.time_s += 3600; // Move 3600 seconds in future.  This is a config setting

    DEBUG_CRUMB( "logMap loadAppToFileMap begin dev=" + dev + " dir=" + dir + " ext=" + ext +
                 " first=" + firstFile.fullName() + " firstTs=" + logMapDebugTime( firstFile.timestamp() ) +
                 " last=" + lastFile.fullName() + " lastTs=" + logMapDebugTime( lastFile.timestamp() ) +
                 " prevLimit=" + logMapDebugTime( prevts ) + " follLimit=" + logMapDebugTime( follts ) );

    // Coordinates of the previous log, after it's found
    bool            prevLogFound = false;
    file::stdSubDir prevLogSubDir;
    size_t          prevLogFile_n = 0;

    // Coordinates of the following log, after it's found
    bool            follLogFound = false;
    file::stdSubDir follLogSubDir;
    size_t          follLogFile_n;

    std::string basedir = dir + '/' + dev + '/';

    file::stdSubDir subdir = firstFile.subDir( &errc );
    mx_error_check_code( errc );

    int ndays = 0;

    while( prevLogFound == false && ndays < m_searchDaySpan )
    {
        ++ndays;

        std::vector<std::string> tmp_flist;

        isdir = mx::ioutils::dir_exists_is( basedir + subdir.path(), errc );
        mx_error_check_code( errc );

        if( !isdir ) // this subdir doesn't exist so go around
        {
            mx_error_check( subdir.subDay() );
            continue;
        }

        mx_error_check( mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext ) );

        if( tmp_flist.size() == 0 ) // this subdir has no files in it so go around
        {
            mx_error_check( subdir.subDay() );
            continue;
        }

        // Start from last file and move backwards
        for( size_t n = tmp_flist.size() - 1; n != static_cast<size_t>( -1 ); --n )
        {
            file::stdFileName<verboseT> sfn;

            try
            {
                // clang-format off
                #ifdef XWCTEST_LOGMAP_LATFM_BADALL1
                    throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
                #endif
                // clang-format on

                sfn.fullName( tmp_flist[n] );
            }
            catch( ... )
            {
                std::throw_with_nested( xwcException( "parsing filename" ) );
            }

            if( !sfn.valid() ) // on any other errors we assume it's not a valid log file and just go around
            {
                continue;
            }

            if( sfn.timestamp() <= prevts )
            {
                prevLogFound  = true;
                prevLogSubDir = subdir;
                prevLogFile_n = n;

                DEBUG_CRUMB( "logMap previous boundary dev=" + dev + " file=" + sfn.fullName() +
                             " timestamp=" + logMapDebugTime( sfn.timestamp() ) );
                break;
            }
        } // iteration over tmp_flist

        if( !prevLogFound )
        {
            mx_error_check( subdir.subDay() );
        }
    }

    if( !prevLogFound )
    {
        return mx::error_t::noerror; // this is not an error...yet.  one must check the map to see if 0 files found.
    }

    subdir = lastFile.subDir( &errc );
    mx_error_check_code( errc );

    ndays = 0;

    while( follLogFound == false && ndays < m_searchDaySpan )
    {
        try
        {
            // clang-format off
            #ifdef XWCTEST_LOGMAP_LATFM_BADALL2
                throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
            #endif
            // clang-format on

            ++ndays;

            std::vector<std::string> tmp_flist;

            isdir = mx::ioutils::dir_exists_is( basedir + subdir.path(), errc );

            if( errc != mx::error_t::noerror )
            {
                return mx::error_report<verboseT>( errc, "error from std::filesystem" );
            }

            if( !isdir ) // this subdir doesn't exist so go around
            {
                mx_error_check( subdir.addDay() );
                continue;
            }

            mx_error_check( mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext ) );

            if( tmp_flist.size() == 0 ) // this subdir has no files so go ahead
            {
                mx_error_check( subdir.addDay() );
                continue;
            }

            // Start from first file and move forward
            for( size_t n = 0; n < tmp_flist.size(); ++n )
            {
                file::stdFileName<verboseT> sfn;

                sfn.fullName( tmp_flist[n] );

                if( !sfn.valid() ) // any other errors just means it's not a standard file
                {
                    continue;
                }

                if( sfn.timestamp() >= follts )
                {
                    follLogFound  = true;
                    follLogSubDir = subdir;
                    follLogFile_n = n;
                    DEBUG_CRUMB( "logMap following boundary dev=" + dev + " file=" + sfn.fullName() +
                                 " timestamp=" + logMapDebugTime( sfn.timestamp() ) );
                    break;
                }

            } // iteration over tmp_flist

            if( !follLogFound )
            {
                mx_error_check( subdir.addDay() );
            }
        }
        catch( ... )
        {
            std::throw_with_nested( xwcException( "parsing filename" ) );
        }
    }

    // In this case we use the last log available and hope for the best
    if( !follLogFound )
    {
        DEBUG_CRUMB( "logMap following boundary not found dev=" + dev );
        follLogSubDir = lastFile.subDir( &errc );
        mx_error_check_code( errc );

        DEBUG_CRUMB( "checking for: " + basedir + follLogSubDir.path() );

        bool exists = mx::ioutils::dir_exists_is( basedir + follLogSubDir.path(), errc );

        int n = 0;
        while( !exists && n < m_searchDaySpan )
        {
            follLogSubDir.subDay();

            DEBUG_CRUMB( "checking for: " + basedir + follLogSubDir.path() );

            exists = mx::ioutils::dir_exists_is( basedir + follLogSubDir.path(), errc );
            ++n;
        }

        if( !exists )
        {
            follLogSubDir = prevLogSubDir;
        }

        follLogFile_n = static_cast<size_t>( -1 );
    }

    if( prevLogSubDir == follLogSubDir ) // special case, probably most common
    {
        DEBUG_CRUMB( "prevLogSubDir == follLogSubDir" );

        try
        {
#ifdef XWCTEST_LOGMAP_LATFM_BADALL3
            throw xwcException( "std::bad_alloc" ); // LCOV_EXCL_LINE
#endif
            // clang-format on

            subdir = prevLogSubDir;

            std::vector<std::string> tmp_flist;

            mx_error_check( mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext ) );

            if( follLogFile_n == static_cast<size_t>( -1 ) )
            {
                follLogFile_n = tmp_flist.size();
            }
            else
            {
                ++follLogFile_n;
                if( follLogFile_n > tmp_flist.size() )
                {
                    return mx::error_report<verboseT>( mx::error_t::sizeerr,
                                                       "miscounted the number of files somewhere" );
                }
            }

            mx_error_check( addFileListToFileMap( dev, tmp_flist, prevLogFile_n, follLogFile_n ) );
        }
        catch( ... )
        {
            std::throw_with_nested( xwcException( "adding file list to map" ) );
        }
    }
    else
    {
        try
        {
            // clang-format off
            #ifdef XWCTEST_LOGMAP_LATFM_XWCE4
                throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
            #endif

            #ifdef XWCTEST_LOGMAP_LATFM_BADALL4
                throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
            #endif
            // clang-format on

            subdir = prevLogSubDir;

            std::vector<std::string> tmp_flist;

            mx_error_check( mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext ) );

            mx_error_check( addFileListToFileMap( dev, tmp_flist, prevLogFile_n, tmp_flist.size() ) );
        }
        catch( ... )
        {
            std::throw_with_nested( xwcException( "adding file list to map" ) );
        }

        mx_error_check( subdir.addDay() );

        while( subdir < follLogSubDir )
        {
            try
            {
                // clang-format off
                #ifdef XWCTEST_LOGMAP_LATFM_XWCE5
                    throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
                #endif

                #ifdef XWCTEST_LOGMAP_LATFM_BADALL5
                    throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
                #endif
                // clang-format on

                if( !std::filesystem::exists( basedir + subdir.path() ) )
                {
                    mx_error_check( subdir.addDay() );
                    continue;
                }

                std::vector<std::string> tmp_flist;

                mx_error_check( mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext ) );

                mx_error_check( addFileListToFileMap( dev, tmp_flist, 0, tmp_flist.size() ) );

                mx_error_check( subdir.addDay() );
            }
            catch( ... )
            {
                std::throw_with_nested( xwcException( "adding file list to map" ) );
            }
        }

        try
        {
            // clang-format off
            #ifdef XWCTEST_LOGMAP_LATFM_XWCE6
                throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
            #endif

            #ifdef XWCTEST_LOGMAP_LATFM_BADALL6
                throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
            #endif
            // clang-format on

            std::vector<std::string> tmp_flist;

            /*mx::error_t errc = mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext );
            if(errc != mx::error_t::dirnotfound)
            {
                return mx::error_report<verboseT>(errc);
            }*/

            mx_error_check( mx::ioutils::getFileNames( tmp_flist, basedir + subdir.path(), dev, "", ext ) );

            if( errc == mx::error_t::noerror )
            {

                if( follLogFile_n == static_cast<size_t>( -1 ) )
                {
                    follLogFile_n = tmp_flist.size();
                }
                else
                {
                    ++follLogFile_n;
                    if( follLogFile_n > tmp_flist.size() )
                    {
                        return mx::error_report<verboseT>( mx::error_t::sizeerr,
                                                           "miscounted the number of files somewhere" );
                    }
                }

                mx_error_check( addFileListToFileMap( dev, tmp_flist, 0, follLogFile_n ) );
            }
        }
        catch( ... )
        {
            std::throw_with_nested( xwcException( "adding file list to map" ) );
        }
    }

    DEBUG_CRUMB( "logMap loadAppToFileMap end dev=" + dev +
                 " mappedFiles=" + std::to_string( m_appToFileMap[dev].size() ) );

    return mx::error_t::noerror;
}

template <class verboseT>
int logMap<verboseT>::getPriorLog( char                      *&logBefore,
                                   const std::string          &appName,
                                   const flatlogs::eventCodeT &ev,
                                   const flatlogs::timespecX  &ts,
                                   char                       *hint )
{
    static_cast<void>( hint );

    DEBUG_CRUMB( "" );

    if( m_appToFileMap[appName].size() == 0 )
    {
        DEBUG_CRUMB( "getPriorLog no file map app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) );
        return -1;
    }

    DEBUG_CRUMB( "getPriorLog begin app=" + appName + " ev=" + std::to_string( ev ) + " ts=" + logMapDebugTime( ts ) );

    logInMemory &lim = m_appToBufferMap[appName];

    flatlogs::timespecX et = lim.m_endTime;
    et.time_s += 30;
    if( lim.m_startTime > ts || et < ts )
    {
        DEBUG_CRUMB( "getPriorLog loading files app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) + " loadedStart=" + logMapDebugTime( lim.m_startTime ) +
                     " loadedEnd=" + logMapDebugTime( lim.m_endTime ) );

        if( loadFiles( appName, ts ) < 0 )
        {
            DEBUG_CRUMB( "getPriorLog loadFiles failed app=" + appName + " ev=" + std::to_string( ev ) +
                         " ts=" + logMapDebugTime( ts ) );
            return -1;
        }
    }

    if( lim.m_memory.size() == 0 )
    {
        DEBUG_CRUMB( "getPriorLog empty memory app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) );
        return -1;
    }

    char *buffer      = lim.m_memory.data();
    char *bufferEnd   = lim.m_memory.data() + lim.m_memory.size();
    char *priorBuffer = nullptr;

    while( buffer < bufferEnd )
    {
        size_t               totalSize = 0;
        flatlogs::timespecX  minTs;
        flatlogs::timespecX *minTsPtr = nullptr;
        if( priorBuffer != nullptr )
        {
            minTs    = flatlogs::logHeader::timespec( priorBuffer );
            minTsPtr = &minTs;
        }

        if( !logMapEntrySane( totalSize, buffer, bufferEnd, minTsPtr ) )
        {
            char *resynced = logMapResync( buffer, bufferEnd, minTsPtr );
            if( resynced != nullptr )
            {
                DEBUG_CRUMB( "getPriorLog resync app=" + appName + " ev=" + std::to_string( ev ) +
                             " offset=" + std::to_string( buffer - lim.m_memory.data() ) +
                             " resyncOffset=" + std::to_string( resynced - lim.m_memory.data() ) );
                buffer = resynced;
                continue;
            }

            DEBUG_CRUMB( "getPriorLog resync failed app=" + appName + " ev=" + std::to_string( ev ) +
                         " offset=" + std::to_string( buffer - lim.m_memory.data() ) );
            break;
        }

        if( ts < flatlogs::logHeader::timespec( buffer ) )
        {
            break;
        }

        if( flatlogs::logHeader::eventCode( buffer ) == ev )
        {
            priorBuffer = buffer;
        }

        buffer += totalSize;
    }

    if( priorBuffer == nullptr )
    {
        DEBUG_CRUMB( "getPriorLog no prior app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) + " loadedStart=" + logMapDebugTime( lim.m_startTime ) +
                     " loadedEnd=" + logMapDebugTime( lim.m_endTime ) +
                     " memoryBytes=" + std::to_string( lim.m_memory.size() ) );
        return -1;
    }

    logBefore = priorBuffer;

    DEBUG_CRUMB( "getPriorLog found app=" + appName + " ev=" + std::to_string( ev ) + " ts=" + logMapDebugTime( ts ) +
                 " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( logBefore ) ) );

    return 0;
} // getPriorLog

template <class verboseT>
int logMap<verboseT>::getNextLog( char *&logAfter, char *logCurrent, const std::string &appName )
{
    logInMemory &lim = m_appToBufferMap[appName];

    if( logCurrent == nullptr || lim.m_memory.size() == 0 )
    {
        DEBUG_CRUMB( "getNextLog missing current/buffer app=" + appName );
        return -1;
    }

    char *bufferStart = lim.m_memory.data();
    char *bufferEnd   = lim.m_memory.data() + lim.m_memory.size();
    if( logCurrent < bufferStart || logCurrent >= bufferEnd )
    {
        DEBUG_CRUMB( "getNextLog current outside buffer app=" + appName );
        return -1;
    }

    size_t currentSize = flatlogs::logHeader::totalSize( logCurrent );
    if( currentSize == 0 || logCurrent + currentSize > bufferEnd )
    {
        std::cerr << "attempt to read invalid log entry, possible log corruption.\n";
        return -1;
    }

    flatlogs::eventCodeT ev     = flatlogs::logHeader::eventCode( logCurrent );
    char                *buffer = logCurrent + currentSize;

    DEBUG_CRUMB( "getNextLog begin app=" + appName + " ev=" + std::to_string( ev ) +
                 " currentTs=" + logMapDebugTime( flatlogs::logHeader::timespec( logCurrent ) ) + " loadedStart=" +
                 logMapDebugTime( lim.m_startTime ) + " loadedEnd=" + logMapDebugTime( lim.m_endTime ) );

    while( buffer < bufferEnd )
    {
        size_t              totalSize = 0;
        flatlogs::timespecX currentTs = flatlogs::logHeader::timespec( logCurrent );
        if( !logMapEntrySane( totalSize, buffer, bufferEnd, &currentTs ) )
        {
            char *resynced = logMapResync( buffer, bufferEnd, &currentTs );
            if( resynced != nullptr )
            {
                DEBUG_CRUMB( "getNextLog resync app=" + appName + " ev=" + std::to_string( ev ) +
                             " offset=" + std::to_string( buffer - lim.m_memory.data() ) +
                             " resyncOffset=" + std::to_string( resynced - lim.m_memory.data() ) );
                buffer = resynced;
                continue;
            }

            DEBUG_CRUMB( "getNextLog resync failed app=" + appName + " ev=" + std::to_string( ev ) +
                         " offset=" + std::to_string( buffer - lim.m_memory.data() ) );
            break;
        }

        if( flatlogs::logHeader::eventCode( buffer ) == ev )
        {
            logAfter = buffer;
            DEBUG_CRUMB( "getNextLog found app=" + appName + " ev=" + std::to_string( ev ) +
                         " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( logAfter ) ) );
            return 0;
        }

        buffer += totalSize;
    }

    DEBUG_CRUMB( "getNextLog no next app=" + appName + " ev=" + std::to_string( ev ) +
                 " loadedEnd=" + logMapDebugTime( lim.m_endTime ) );
    return 1;
}

template <class verboseT>
int logMap<verboseT>::loadFiles( const std::string &appName, const flatlogs::timespecX &startTime )
{
    if( m_appToFileMap[appName].size() == 0 )
    {
        DEBUG_CRUMB( "loadFiles no file map app=" + appName + " startTime=" + logMapDebugTime( startTime ) );
        return -1;
    }

    DEBUG_CRUMB( "loadFiles begin app=" + appName + " startTime=" + logMapDebugTime( startTime ) +
                 " mappedFiles=" + std::to_string( m_appToFileMap[appName].size() ) );

    // First check if already loaded files cover this time
    if( m_appToBufferMap[appName].m_memory.size() > 0 )
    {
        if( m_appToBufferMap[appName].m_startTime <= startTime && m_appToBufferMap[appName].m_endTime >= startTime )
        {
            DEBUG_CRUMB( "loadFiles already covered app=" + appName +
                         " loadedStart=" + logMapDebugTime( m_appToBufferMap[appName].m_startTime ) +
                         " loadedEnd=" + logMapDebugTime( m_appToBufferMap[appName].m_endTime ) );
            return 0;
        }

#ifdef DEBUG
        std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

        if( m_appToBufferMap[appName].m_startTime > startTime ) // Files don't go back far enough
        {
#ifdef DEBUG
            std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

            auto last = m_appToFileMap[appName].begin();
            while( last->timestamp() < m_appToBufferMap[appName].m_startTime )
            {
                ++last;
                if( last == m_appToFileMap[appName].end() )
                    break;
            }
            // Now last is the last file to open in the for loop sense.
            auto first = last;

            while( first->timestamp() > startTime )
            {
                --first;
                if( first == m_appToFileMap[appName].begin() )
                    break;
            }

            // Now open each of these files, in reverse
            --last;
            --first;
            for( auto it = last; it != first; --it )
            {
                DEBUG_CRUMB( "loadFiles append backward app=" + appName + " file=" + it->fullName() +
                             " timestamp=" + logMapDebugTime( it->timestamp() ) );
                m_appToBufferMap[appName].m_reportPrefix = m_reportPrefix;
                m_appToBufferMap[appName].loadFile( *it );
            }

            return 0;
        }
        else
        {
            auto first = m_appToFileMap[appName].end();
            --first;

            while( first->timestamp() > m_appToBufferMap[appName].m_endTime )
            {
                --first;
                if( first == m_appToFileMap[appName].begin() )
                    break;
            }
            ++first;
            auto last = first;
            while( last->timestamp() < startTime )
            {
                ++last;
                if( last == m_appToFileMap[appName].end() )
                    break;
            }

            // Now open each of these files
            for( auto it = first; it != last; ++it )
            {
                DEBUG_CRUMB( "loadFiles append forward app=" + appName + " file=" + it->fullName() +
                             " timestamp=" + logMapDebugTime( it->timestamp() ) );
                m_appToBufferMap[appName].m_reportPrefix = m_reportPrefix;
                m_appToBufferMap[appName].loadFile( *it );
            }
            return 0;
        }
    }

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    auto before = m_appToFileMap[appName].begin();

    for( ; before != m_appToFileMap[appName].end(); ++before )
    {
        if( !( before->timestamp() < startTime ) )
        {
            break;
        }
    }

#ifdef debug
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    if( before == m_appToFileMap[appName].begin() )
    {
        DEBUG_CRUMB( "loadFiles no prior file app=" + appName + " startTime=" + logMapDebugTime( startTime ) );
        return -1;
    }
    --before;

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    m_appToBufferMap.emplace( std::pair<std::string, logInMemory>( appName, logInMemory() ) );

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    m_appToBufferMap[appName].m_reportPrefix = m_reportPrefix;
    m_appToBufferMap[appName].loadFile( *before );
    DEBUG_CRUMB( "loadFiles initial before app=" + appName + " file=" + before->fullName() +
                 " timestamp=" + logMapDebugTime( before->timestamp() ) );
    if( ++before != m_appToFileMap[appName].end() )
    {
        m_appToBufferMap[appName].m_reportPrefix = m_reportPrefix;
        m_appToBufferMap[appName].loadFile( *before );
        DEBUG_CRUMB( "loadFiles initial after app=" + appName + " file=" + before->fullName() +
                     " timestamp=" + logMapDebugTime( before->timestamp() ) );
    }

    DEBUG_CRUMB( "loadFiles end app=" + appName +
                 " loadedStart=" + logMapDebugTime( m_appToBufferMap[appName].m_startTime ) +
                 " loadedEnd=" + logMapDebugTime( m_appToBufferMap[appName].m_endTime ) +
                 " memoryBytes=" + std::to_string( m_appToBufferMap[appName].m_memory.size() ) );

    return 0;
}

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

extern template class logMap<XWC_DEFAULT_VERBOSITY>;

} // namespace logger
} // namespace MagAOX

#endif // logger_logMap_hpp
