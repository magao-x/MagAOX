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

#ifndef DEBUG_CRUMB
#define DEBUG_CRUMB(msg) {std::cerr << msg << '(' << __FILE__ << ' ' << __LINE__ << "\n";}
#endif

namespace MagAOX
{
namespace logger
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Structure to hold a log file in memory, tracking when a new file needs to be opened.
struct logInMemory
{
    typedef XWC_DEFAULT_VERBOSITY verboseT;

    std::vector<char> m_memory; ///< The buffer holding the log data.

    flatlogs::timespecX m_startTime{ 0, 0 };
    flatlogs::timespecX m_endTime{ 0, 0 };

    int loadFile( file::stdFileName<verboseT> const &lfn );
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

    int m_searchDaySpan {100}; ///< Maximum number of days to search for files in the past/future.

    appToFileMapT m_appToFileMap;

    appToBufferMapT m_appToBufferMap;

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

    int getNearestLogs( flatlogs::bufferPtrT &logBefore, flatlogs::bufferPtrT &logAfter, const std::string &appName );

    int loadFiles( const std::string         &appName,  ///< MagAO-X app name for which to load files
                   const flatlogs::timespecX &startTime ///<
    );
};

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

            if( !sfn.valid() ) // this is just not a standard file name.
            {
                continue;
            }

            if( sfn.appName() != dev ) // this is just a different app
            {
                continue;
            }

            m_appToFileMap[dev].insert( sfn );
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

                std::cerr << "found previous log: " << tmp_flist[n] << '\n';
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
        follLogSubDir = lastFile.subDir( &errc );
        mx_error_check_code( errc );

        DEBUG_CRUMB("checking for: " + basedir + follLogSubDir.path());

        bool exists = mx::ioutils::dir_exists_is(basedir + follLogSubDir.path(), errc);

        int n =0;
        while(!exists && n < m_searchDaySpan)
        {
            follLogSubDir.subDay();

            DEBUG_CRUMB("checking for: " + basedir + follLogSubDir.path());

            exists = mx::ioutils::dir_exists_is(basedir + follLogSubDir.path(), errc);
            ++n;
        }

        follLogFile_n = static_cast<size_t>( -1 );
    }

    if( prevLogSubDir == follLogSubDir ) // special case, probably most common
    {
        DEBUG_CRUMB("prevLogSubDir == follLogSubDir");

        try
        {
            #ifdef XWCTEST_LOGMAP_LATFM_BADALL3
                throw xwcException("std::bad_alloc"); // LCOV_EXCL_LINE
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
        std::cerr << "prevLogSubDir != follLogSubDir\n";
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


            if(errc == mx::error_t::noerror)
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

    return mx::error_t::noerror;
}

template <class verboseT>
int logMap<verboseT>::getPriorLog( char                      *&logBefore,
                                   const std::string          &appName,
                                   const flatlogs::eventCodeT &ev,
                                   const flatlogs::timespecX  &ts,
                                   char                       *hint )
{
    flatlogs::eventCodeT evL;

    DEBUG_CRUMB("");

    if( m_appToFileMap[appName].size() == 0 )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " getPriorLog empty map\n";
        return -1;
    }

    DEBUG_CRUMB("");

    logInMemory &lim = m_appToBufferMap[appName];

    flatlogs::timespecX et = lim.m_endTime;
    et.time_s += 30;
    if( lim.m_startTime > ts || et < ts )
    {
        DEBUG_CRUMB("");

        if( loadFiles( appName, ts ) < 0 )
        {
            std::cerr << __FILE__ << " " << __LINE__ << " error returned from loadfiles\n";
            return -1;
        }
    }

    char *buffer, *priorBuffer;

    if( hint )
    {
        if( flatlogs::logHeader::timespec( hint ) <= ts )
        {
            buffer = hint;
        }
        else
        {
            buffer = lim.m_memory.data();
        }
    }
    else
    {
        buffer = lim.m_memory.data();
    }

    priorBuffer = buffer;
    evL         = flatlogs::logHeader::eventCode( buffer );

    while( evL != ev )
    {
        priorBuffer = buffer;
        buffer += flatlogs::logHeader::totalSize( buffer );
        if( buffer >= lim.m_memory.data() + lim.m_memory.size() )
            break;
        evL = flatlogs::logHeader::eventCode( buffer );
    }

    if( evL != ev )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " Event code not found.\n";
        return -1;
    }

    if( flatlogs::logHeader::timespec( buffer ) < ts )
    {
        while( flatlogs::logHeader::timespec( buffer ) < ts ) // Loop until buffer is after the timestamp we want
        {
            if( buffer > lim.m_memory.data() + lim.m_memory.size() )
            {
                std::cerr << __FILE__ << " " << __LINE__
                          << " attempt to read too mach data, possible log corruption.\n";
                return -1;
            }

            if( buffer == lim.m_memory.data() + lim.m_memory.size() )
            {
                std::cerr << __FILE__ << " " << __LINE__ << " did not find following log for " << appName
                          << " -- need to load more data.\n";
                // Proper action here is to load the next file if possible...
                return 1;
            }

            priorBuffer = buffer;

            buffer += flatlogs::logHeader::totalSize( buffer );

            evL = flatlogs::logHeader::eventCode( buffer );

            while( evL != ev ) // Find the next log with the event code we want.
            {
                if( buffer > lim.m_memory.data() + lim.m_memory.size() )
                {
                    std::cerr << __FILE__ << " " << __LINE__
                              << " attempt to read too mach data, possible log corruption.\n";
                    return -1;
                }

                if( buffer == lim.m_memory.data() + lim.m_memory.size() )
                {
                    std::cerr << __FILE__ << " " << __LINE__ << " did not find following log for " << appName
                              << " -- need to load more data.\n";
                    // Proper action here is to load the next file if possible...
                    return 1;
                }

                buffer += flatlogs::logHeader::totalSize( buffer );
                evL = flatlogs::logHeader::eventCode( buffer );
            }
        }
    }

    logBefore = priorBuffer;

    return 0;
} // getPriorLog

template <class verboseT>
int logMap<verboseT>::getNextLog( char *&logAfter, char *logCurrent, const std::string &appName )
{
    flatlogs::eventCodeT ev, evL;

    logInMemory &lim = m_appToBufferMap[appName];

    char *buffer;

    ev = flatlogs::logHeader::eventCode( logCurrent );

    buffer = logCurrent;

    buffer += flatlogs::logHeader::totalSize( buffer );
    if( buffer >= lim.m_memory.data() + lim.m_memory.size() )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " Reached end of data for " << appName
                  << " -- need to load more data\n";
        // propoer action is to load the next file if possible.
        return 1;
    }

    evL = flatlogs::logHeader::eventCode( buffer );

    while( evL != ev )
    {
        buffer += flatlogs::logHeader::totalSize( buffer );
        if( buffer >= lim.m_memory.data() + lim.m_memory.size() )
        {
            std::cerr << __FILE__ << " " << __LINE__ << " Reached end of data for " << appName
                      << "-- need to load more data\n";
            // propoer action is to load the next file if possible.
            return 1;
        }
        evL = flatlogs::logHeader::eventCode( buffer );
    }

    if( evL != ev )
    {
        std::cerr << "Event code not found.\n";
        return -1;
    }

    logAfter = buffer;

    return 0;
}

template <class verboseT>
int logMap<verboseT>::loadFiles( const std::string &appName, const flatlogs::timespecX &startTime )
{
    if( m_appToFileMap[appName].size() == 0 )
    {
        std::cerr << "*************************************\n\n";
        std::cerr << "No files for " << appName << "\n";
        std::cerr << "*************************************\n\n";
        return -1;
    }

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    // First check if already loaded files cover this time
    if( m_appToBufferMap[appName].m_memory.size() > 0 )
    {
        if( m_appToBufferMap[appName].m_startTime <= startTime && m_appToBufferMap[appName].m_endTime >= startTime )
        {
            std::cerr << "good!\n";
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
            std::cerr << "open earlier files!\n";
            --last;
            --first;
            for( auto it = last; it != first; --it )
            {
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
            std::cerr << "open later file for " << appName << "!\n";
            for( auto it = first; it != last; ++it )
            {
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
        std::cerr << "No files in range for " << appName << "\n";
    }
    --before;

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    m_appToBufferMap.emplace( std::pair<std::string, logInMemory>( appName, logInMemory() ) );

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    m_appToBufferMap[appName].loadFile( *before );
    if( ++before != m_appToFileMap[appName].end() )
    {
        m_appToBufferMap[appName].loadFile( *before );
    }

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    return 0;
}

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

extern template class logMap<XWC_DEFAULT_VERBOSITY>;

} // namespace logger
} // namespace MagAOX

#endif // logger_logMap_hpp
