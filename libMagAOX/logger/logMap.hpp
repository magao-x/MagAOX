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

#include <algorithm>
#include <limits>
#include <vector>
#include <map>
#include <set>

#include <flatlogs/flatlogs.hpp>
#include "../file/stdFileName.hpp"
#include "generated/logCodes.hpp"
#include "logExtensions.hpp"

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

/// Check if log type of given event code is known.
inline bool logMapEventCodeKnown( flatlogs::eventCodeT ev /**< [in] event code to check */ )
{
    return eventCodeName( ev ) != "unknown event code";
}

/// Result of integrity check on a log entry.
/** Meaning for each version:
 * - v2: \ref verified if the checksum matches, \ref failedChecksum if not, and \ref failedStructure if the entry
 *   doesn't fit in the file or its version byte is wrong.
 * - v1: since v1 has no checksum, \ref notChecked if the entry fits, its priority is a defined one and its time 
 *   is not before the previous entry's in the same file. Otherwise \ref failedStructure.
 *
 * Unknown event codes aren't included here.
 */
enum class logIntegrity : uint8_t
{
    verified,        ///< v2: the checksum matches, so the bytes are as they were written.
    failedChecksum,  ///< v2: the entry fits, but its checksum doesn't match.
    notChecked,      ///< v1: passed the plausibility checks but there is no checksum to check.
    failedStructure  ///< The entry doesn't fit, has the wrong version byte, or, for v1, fails the plausibility checks.
};

/// Map integrity result to string.
/**
 * \returns "verified", "not checked", "structure" or "checksum".
 */
inline const char *logIntegrityName( logIntegrity integrity /**< [in] the result to name */ )
{
    switch( integrity )
    {
    case logIntegrity::verified:
        return "verified";
    case logIntegrity::failedChecksum:
        return "checksum";
    case logIntegrity::notChecked:
        return "not checked";
    case logIntegrity::failedStructure:
        return "structure";
    }

    return "unknown";
}

/// The outcome of reading one entry while walking through a file.
struct logStep
{
    flatlogs::parsedEntry m_entry; ///< The decoded entry.

    logIntegrity m_integrity{ logIntegrity::failedStructure }; ///< Integrity check result.

    /// Check if the entry can be used.
    /**
     * \returns true if the verdict is \ref logIntegrity::verified or \ref logIntegrity::notChecked.
     */
    bool usable() const;
};

inline bool logStep::usable() const
{
    return m_integrity == logIntegrity::verified || m_integrity == logIntegrity::notChecked;
}

/// Read and check integrity of entry at one position in a file's bytes.
/** Both versions check integrity here.
 * Checking if the log type is known is not included here and is left to the callers that need 
 * it for efficiency reasons.
 */
inline void logMapStepEntry( logStep &step,                        /**< [out] the entry with its integrity result */
                             const char *fileBytes,                /**< [in] the start of the file's bytes */
                             size_t fileSize,                      /**< [in] the number of bytes in the file */
                             size_t pos,                           /**< [in] offset of the entry in the file */
                             flatlogs::logVersionT version,        /**< [in] the file's format version */
                             const flatlogs::timespecX *prevTs     /**< [in] time of the previous usable entry in
                                                                             this file, or null if none */
)
{
    step.m_integrity = logIntegrity::failedStructure;

    if( fileBytes == nullptr || pos >= fileSize )
    {
        step.m_entry = flatlogs::parsedEntry();
        step.m_entry.version = version;
        return;
    }

    step.m_entry = flatlogs::parseEntry( fileBytes + pos, fileSize - pos, version );

    switch( step.m_entry.status )
    {
    case flatlogs::entryStatus::checksumOk:
        step.m_integrity = logIntegrity::verified;
        break;
    case flatlogs::entryStatus::checksumMismatch:
        step.m_integrity = logIntegrity::failedChecksum;
        break;
    case flatlogs::entryStatus::noChecksum:
    {
        flatlogs::timespecX ts = step.m_entry.ts;
        if( !logMapPriorityValid( step.m_entry.prio ) || ( prevTs != nullptr && ts < *prevTs ) )
        {
            step.m_integrity = logIntegrity::failedStructure;
        }
        else
        {
            step.m_integrity = logIntegrity::notChecked;
        }
        break;
    }
    default:
        step.m_integrity = logIntegrity::failedStructure;
        break;
    }
}

/// Check whether a v1 entry is plausible enough to be part of a restart chain after damage.
/** Stricter than \ref logMapStepEntry: the event code must also be known, so that damaged bytes that happen to look
 * like an entry of an unknown type are not chosen as the place to restart.
 *
 * \returns true if the entry can be a link in the chain.
 */
inline bool logMapV1RestartLink( flatlogs::parsedEntry &entry,         /**< [out] the decoded entry */
                                 const char *fileBytes,                /**< [in] the start of the file's bytes */
                                 size_t fileSize,                      /**< [in] the number of bytes in the file */
                                 size_t pos,                           /**< [in] offset of the entry in the file */
                                 const flatlogs::timespecX *minTs      /**< [in] earliest plausible time, or null */
)
{
    entry = flatlogs::logHeaderV1::parse( fileBytes + pos, fileSize - pos );

    if( entry.status != flatlogs::entryStatus::noChecksum )
    {
        return false;
    }

    if( !logMapPriorityValid( entry.prio ) )
    {
        return false;
    }

    if( !logMapEventCodeKnown( entry.eventCode ) )
    {
        return false;
    }

    flatlogs::timespecX ts = entry.ts;
    return minTs == nullptr || !( ts < *minTs );
}

/// Within one v1 file, byte-scan forward to the next place where a chain of plausible entries starts.
/** An entry is accepted when three plausible entries follow each other from it, or when the plausible entries
 * from it run exactly to the end of the file.
 * Never reads past the end of the file.
 *
 * \returns the offset of the place to restart, or flatlogs::entryNotFound if there is none in this file.
 */
inline size_t logMapResyncV1( const char *fileBytes,                /**< [in] the start of the file's bytes */
                              size_t fileSize,                      /**< [in] the number of bytes in the file */
                              size_t failedAt,                      /**< [in] offset of the failed entry */
                              const flatlogs::timespecX *minTs      /**< [in] earliest plausible time, or null */
)
{
    if( fileBytes == nullptr )
    {
        return flatlogs::entryNotFound;
    }

    for( size_t candidate = failedAt + 1; candidate < fileSize; ++candidate )
    {
        size_t chain = candidate;
        size_t chainLinks = 0;
        flatlogs::timespecX chainMinTs;
        const flatlogs::timespecX *chainMinTsPtr = minTs;
        while( chain < fileSize )
        {
            flatlogs::parsedEntry entry;
            if( !logMapV1RestartLink( entry, fileBytes, fileSize, chain, chainMinTsPtr ) )
            {
                break;
            }

            ++chainLinks;
            if( chainLinks >= 3 )
            {
                return candidate;
            }

            chainMinTs = entry.ts;
            chainMinTsPtr = &chainMinTs;
            chain += entry.entrySize;
        }

        if( chain == fileSize && chainLinks > 0 )
        {
            return candidate;
        }
    }

    return flatlogs::entryNotFound;
}

/// Find where to carry on after a failed entry.
/** v2 searches for the next entry that passes its checksum.
 * v1 searches for a chain of plausible entries.
 *
 * \returns the offset of the place to carry on, or flatlogs::entryNotFound if none in this file.
 */
inline size_t logMapRecover( const char *fileBytes,                /**< [in] the start of the file's bytes */
                             size_t fileSize,                      /**< [in] the number of bytes in the file */
                             size_t failedAt,                      /**< [in] offset of the failed entry */
                             flatlogs::logVersionT version,        /**< [in] the file's version */
                             const logStep &failed,                /**< [in] the result of reading the failed entry */
                             const flatlogs::timespecX *prevTs     /**< [in] time of the previous usable entry in
                                                                             this file, or null if none */
)
{
    if( version == flatlogs::logVersion1 )
    {
        return logMapResyncV1( fileBytes, fileSize, failedAt, prevTs );
    }

    if( version == flatlogs::logVersion2 )
    {
        return flatlogs::findNextEntryV2( fileBytes, fileSize, failedAt, failed.m_entry );
    }

    return flatlogs::entryNotFound;
}

/// Result of stop during walk through file.
enum class logWalkEvent
{
    entry,  ///< A usable entry.
    damage, ///< An entry that failed.
    end     ///< The end of the file.
};

/// Walks through the entries of one file, recovering from damage.
/** Each file is walked by the rules of its own version.
 */
struct logFileWalk
{
    const char *m_bytes{ nullptr }; ///< The start of the file's bytes. Not owned.

    size_t m_size{ 0 }; ///< The number of bytes in the file.

    flatlogs::logVersionT m_version{ flatlogs::logVersion1 }; ///< The file's version.

    size_t m_pos{ 0 }; ///< Offset where the next entry is expected.

    bool m_havePrevTs{ false }; ///< True once a usable entry has been found in this walk.

    flatlogs::timespecX m_prevTs; ///< Time of the previous usable entry, valid if m_havePrevTs is true.

    /// Construct a walk that has nothing to walk.
    logFileWalk();

    /// Construct a walk through a file.
    logFileWalk( const char *bytes,               /**< [in] the start of the file's bytes */
                 size_t size,                     /**< [in] the number of bytes in the file */
                 flatlogs::logVersionT version,   /**< [in] the file's version */
                 size_t pos = 0                   /**< [in] offset of the first entry to read */
    );

    /// Move to the next entry and evaluate.
    /**
     * \returns logWalkEvent::entry if a usable entry was read; the walk has moved past it.
     * \returns logWalkEvent::damage if the entry failed; the walk has moved to \p resumeAt.
     * \returns logWalkEvent::end at the end of the file; \p step, \p at and \p resumeAt are unchanged.
     */
    logWalkEvent next( logStep &step,    /**< [out] the entry that was read, with its evaluation result */
                       size_t &at,       /**< [out] offset of that entry in the file */
                       size_t &resumeAt  /**< [out] offset where the walk carries on, or flatlogs::entryNotFound
                                                   if a damaged entry left nothing usable in the file */
    );
};

inline logFileWalk::logFileWalk()
{
}

inline logFileWalk::logFileWalk( const char *bytes, size_t size, flatlogs::logVersionT version, size_t pos )
    : m_bytes( bytes ), m_size( size ), m_version( version ), m_pos( pos )
{
}

inline logWalkEvent logFileWalk::next( logStep &step, size_t &at, size_t &resumeAt )
{
    if( m_bytes == nullptr || m_pos >= m_size )
    {
        return logWalkEvent::end;
    }

    const flatlogs::timespecX *prevTs = m_havePrevTs ? &m_prevTs : nullptr;

    at = m_pos;
    logMapStepEntry( step, m_bytes, m_size, m_pos, m_version, prevTs );

    if( step.usable() )
    {
        m_prevTs = step.m_entry.ts;
        m_havePrevTs = true;
        m_pos += step.m_entry.entrySize;
        resumeAt = m_pos;
        return logWalkEvent::entry;
    }

    resumeAt = logMapRecover( m_bytes, m_size, at, m_version, step, prevTs );
    m_pos = ( resumeAt == flatlogs::entryNotFound ) ? m_size : resumeAt;

    return logWalkEvent::damage;
}

/// A log entry found by a lookup.
/** It holds its own copy of the entry and the copy stays valid whatever is loaded afterwards.
 */
struct logEntryRecord
{
    /// A copy of the whole entry as stored in the file. Empty if no entry is held.
    /** The padding places the message on an 8-byte boundary (the vector's storage is at least that aligned), so log
     * types reading it as a flatbuffer don't read unaligned.
     */
    std::vector<char> m_bytes;

    size_t m_entrySize{ 0 }; ///< The size of the entry, without the padding.

    size_t m_messageOffset{ 0 }; ///< Where the message starts in m_bytes.

    flatlogs::logVersionT m_version{ 0 }; ///< The version the entry was read as.

    flatlogs::logPrioT m_prio{ 0 }; ///< The entry's priority.

    flatlogs::eventCodeT m_eventCode{ 0 }; ///< The entry's event code.

    flatlogs::timespecX m_ts; ///< The entry's timestamp.

    flatlogs::msgLenT m_msgLen{ 0 }; ///< The length of the entry's message.

    logIntegrity m_integrity{ logIntegrity::failedStructure }; ///< Entry's integrity result.

    bool m_unsupportedType{ false }; ///< True if this build doesn't know the entry's event code.

    std::string m_sourceFile; ///< Full path of the file the entry came from.

    size_t m_sourceOffset{ std::numeric_limits<size_t>::max() }; ///< Byte offset of the entry in its file.

    /// Check whether this record holds no entry.
    /**
     * \returns true if no entry has been copied into this record, as when a lookup finds nothing.
     */
    bool empty() const;

    /// Retrieve entry's message to pass to a log type.
    /**
     * \returns a pointer to the message in m_bytes, or nullptr if no entry is held.
     */
    void *message();
};

inline bool logEntryRecord::empty() const
{
    return m_bytes.empty();
}

inline void *logEntryRecord::message()
{
    if( m_bytes.empty() )
    {
        return nullptr;
    }

    return m_bytes.data() + m_messageOffset;
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
        size_t                m_begin{ 0 }; ///< First byte offset from this source file.
        size_t                m_end{ 0 };   ///< One past the last byte offset from this source file.
        std::string           m_name;       ///< Full source file path.
        flatlogs::logVersionT m_version{ flatlogs::logVersion1 }; ///< Version this file is read as.
    };

    std::vector<char> m_memory; ///< The buffer holding the log data.

    flatlogs::timespecX m_startTime{ 0, 0 }; ///< Earliest timestamp covered by the loaded buffer.
    flatlogs::timespecX m_endTime{ 0, 0 };   ///< Latest timestamp covered by the loaded buffer.

    std::vector<loadedFile> m_loadedFiles; ///< Source-file provenance ranges for the loaded buffer.

    size_t m_recoverableErrors{ 0 }; ///< Count of recoverable log parsing errors encountered while loading.

    std::string m_reportPrefix; ///< Optional prefix for user-facing recoverable log parsing reports.

    /// Full paths of files that failed to load.
    /** A file that failed to load is not read or reported again to avoid redundancy.
     * This includes failures contingent on what is already loaded (overlapping files, or a file that would
     * go in the middle).
     */
    std::set<std::string> m_rejectedFiles;

    /// Load one flatlog file into memory.
    /** On load, entries are read and evaluated. Damaged entries are reported and skipped.
     * Entries of a log type this build doesn't know are reported once per file and event code.
     *
     * \returns 0 on success
     * \returns -1 if the file can't be read, has no usable entry, can't be placed among the loaded files, or failed
     *          to load before
     */
    int loadFile( file::stdFileName<verboseT> const &lfn /**< [in] standard file name to load */ );

    /// Reject file that previously failed to load.
    /**
     * \returns -1, for loadFile to return.
     */
    int rejectFile( file::stdFileName<verboseT> const &lfn /**< [in] the file that failed to load */ );

    /// Start a walk through loaded file.
    /**
     * \returns a walk through the file's bytes in m_memory.
     */
    logFileWalk fileWalk( size_t file,      /**< [in] index of the file in m_loadedFiles */
                          size_t offset = 0 /**< [in] offset in the file of the first entry to read */
    ) const;

    /// Copy a usable entry found by a walk into a record.
    /**
     * \returns the record holding the entry.
     */
    logEntryRecord record( size_t file,           /**< [in] index of the entry's file in m_loadedFiles */
                           size_t offset,         /**< [in] offset of the entry in its file */
                           const logStep &step    /**< [in] the entry as read by the walk */
    ) const;

    /// Find a loaded file by its full path.
    /**
     * \returns the index of the file in m_loadedFiles or m_loadedFiles.size() if file not loaded.
     */
    size_t findFile( const std::string &name /**< [in] full path of the file */ ) const;
};

/// Structure to walk in order the usable entries of all loaded files of one app.
/** Damaged entries are skipped silently (reported when file was loaded).
 */
struct logMemoryWalk
{
    const logInMemory *m_lim{ nullptr }; ///< The loaded files being walked. Not owned.

    size_t m_file{ 0 }; ///< Index in m_lim->m_loadedFiles of the file being walked.

    logFileWalk m_walk; ///< The walk through the current file.

    /// Construct a walk starting at an entry of one loaded file.
    logMemoryWalk( const logInMemory &lim,        /**< [in] the loaded files to walk */
                   size_t file = 0,               /**< [in] index of the file to start in */
                   size_t offset = 0              /**< [in] offset in that file of the first entry to read */
    );

    /// Move to the next usable entry.
    /** At the end of each file moves to the next file.
     *
     * \returns true if a usable entry was found
     * \returns false at the end of the last loaded file
     */
    bool next( logStep &step,  /**< [out] the entry that was read, with its verdict */
               size_t &file,   /**< [out] index of the entry's file in m_loadedFiles */
               size_t &offset  /**< [out] offset of the entry in its file */
    );
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

    /// List an app's log files from one directory, for every extension implied by a base extension.
    /** The files are listed, merged, and sorted by name (i.e. in time order).
     *
     * \returns mx::error_t::noerror on success
     * \returns errors from mx::ioutils::getFileNames
     */
    mx::error_t listLogFiles( std::vector<std::string> &flist, /**< [out] the sorted list of file paths */
                              const std::string &dir,          /**< [in] the directory to list */
                              const std::string &dev,          /**< [in] the device name, a prefix of the file names */
                              const std::string &ext           /**< [in] the base extension */
    );

    /// Get log file names in a directory and distribute them into the map by app-name
    /** Finding no logs is not reported as an error (no exception is thrown).  You must check
     *  the size of m_appToFileMap to check if any files were found.  Reporting that is left to the caller.
     *  Files of both versions are found.
     */
    mx::error_t loadAppToFileMap( const std::string &dir,        /**< [in] the directory to search for files
                                                                           (contains the dev/YYYY_MM_DD subdirs)*/
                                  const std::string  &dev,       ///< [in] the device name to search for logs of
                                  const std::string  &ext,       ///< [in] the base extension to search for
                                  const stdFileNameT &firstFile, ///< [in] the first file that needs coverage
                                  const stdFileNameT &lastFile   ///< [in] the last file that needs coverage
    );

    /// Get the last entry with an event code at or before a time.
    /** Loads files as needed. The result holds its own copy of the entry.
     *
     * The search stops at the first entry later than \p ts. v2 entries are not checked for time order, so, if the clock
     * stepped backwards, a verified v2 entry written after that is not found.
     *
     * \returns 0 on success
     * \returns -1 if no such entry is loaded
     */
    int getPriorEntry( logEntryRecord &before,              /**< [out] the entry found */
                       const std::string &appName,          /**< [in] the name of the app whose logs are searched */
                       const flatlogs::eventCodeT &ev,      /**< [in] the event code to search for */
                       const flatlogs::timespecX &ts        /**< [in] the time being compared against */
    );

    /// Get the next entry with the same event code as current entry.
    /**
     * \returns 0 on success
     * \returns 1 if there is no later entry with that event code in the loaded files
     * \returns -1 if \p current can't be found in the loaded files
     */
    int getNextEntry( logEntryRecord &after,         /**< [out] the entry found */
                      const logEntryRecord &current, /**< [in] the entry to start from */
                      const std::string &appName     /**< [in] the name of the app whose logs are searched */
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
mx::error_t logMap<verboseT>::listLogFiles( std::vector<std::string> &flist,
                                            const std::string        &dir,
                                            const std::string        &dev,
                                            const std::string        &ext )
{
    flist.clear();

    for( const std::string &oneExt : logExtensionsFor( ext ) )
    {
        std::vector<std::string> extList;
        mx_error_check( mx::ioutils::getFileNames( extList, dir, dev, "", oneExt ) );
        flist.insert( flist.end(), extList.begin(), extList.end() );
    }

    std::sort( flist.begin(), flist.end() );

    return mx::error_t::noerror;
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

        mx_error_check( listLogFiles( tmp_flist, basedir + subdir.path(), dev, ext ) );

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

            mx_error_check( listLogFiles( tmp_flist, basedir + subdir.path(), dev, ext ) );

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

            mx_error_check( listLogFiles( tmp_flist, basedir + subdir.path(), dev, ext ) );

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

            mx_error_check( listLogFiles( tmp_flist, basedir + subdir.path(), dev, ext ) );

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

                mx_error_check( listLogFiles( tmp_flist, basedir + subdir.path(), dev, ext ) );

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

            mx_error_check( listLogFiles( tmp_flist, basedir + subdir.path(), dev, ext ) );

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
int logMap<verboseT>::getPriorEntry( logEntryRecord             &before,
                                     const std::string          &appName,
                                     const flatlogs::eventCodeT &ev,
                                     const flatlogs::timespecX  &ts )
{
    if( m_appToFileMap[appName].size() == 0 )
    {
        DEBUG_CRUMB( "getPriorEntry no file map app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) );
        return -1;
    }

    logInMemory &lim = m_appToBufferMap[appName];

    flatlogs::timespecX et = lim.m_endTime;
    et.time_s += 30;
    if( lim.m_startTime > ts || et < ts )
    {
        DEBUG_CRUMB( "getPriorEntry loading files app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) + " loadedStart=" + logMapDebugTime( lim.m_startTime ) +
                     " loadedEnd=" + logMapDebugTime( lim.m_endTime ) );

        if( loadFiles( appName, ts ) < 0 )
        {
            return -1;
        }
    }

    bool found = false;
    size_t foundFile = 0;
    size_t foundOffset = 0;
    logStep foundStep;

    logMemoryWalk walk( lim );
    logStep step;
    size_t file = 0;
    size_t offset = 0;
    while( walk.next( step, file, offset ) )
    {
        if( ts < step.m_entry.ts )
        {
            break;
        }

        if( step.m_entry.eventCode == ev )
        {
            found = true;
            foundFile = file;
            foundOffset = offset;
            foundStep = step;
        }
    }

    if( !found )
    {
        DEBUG_CRUMB( "getPriorEntry no prior app=" + appName + " ev=" + std::to_string( ev ) +
                     " ts=" + logMapDebugTime( ts ) + " loadedStart=" + logMapDebugTime( lim.m_startTime ) +
                     " loadedEnd=" + logMapDebugTime( lim.m_endTime ) );
        return -1;
    }

    before = lim.record( foundFile, foundOffset, foundStep );

    DEBUG_CRUMB( "getPriorEntry found app=" + appName + " ev=" + std::to_string( ev ) + " ts=" + logMapDebugTime( ts ) +
                 " logTs=" + logMapDebugTime( before.m_ts ) );

    return 0;
}

template <class verboseT>
int logMap<verboseT>::getNextEntry( logEntryRecord &after, const logEntryRecord &current, const std::string &appName )
{
    auto appBuffer = m_appToBufferMap.find( appName );
    if( appBuffer == m_appToBufferMap.end() || current.empty() )
    {
        DEBUG_CRUMB( "getNextEntry missing current/buffer app=" + appName );
        return -1;
    }

    logInMemory &lim = appBuffer->second;

    size_t currentFile = lim.findFile( current.m_sourceFile );
    if( currentFile >= lim.m_loadedFiles.size() )
    {
        DEBUG_CRUMB( "getNextEntry current file not loaded app=" + appName + " file=" + current.m_sourceFile );
        return -1;
    }

    // The walk starts by re-reading the current entry.
    logMemoryWalk walk( lim, currentFile, current.m_sourceOffset );
    logStep step;
    size_t file = 0;
    size_t offset = 0;
    if( !walk.next( step, file, offset ) || file != currentFile || offset != current.m_sourceOffset )
    {
        std::cerr << "attempt to read invalid log entry, possible log corruption.\n";
        return -1;
    }

    while( walk.next( step, file, offset ) )
    {
        if( step.m_entry.eventCode == current.m_eventCode )
        {
            after = lim.record( file, offset, step );
            DEBUG_CRUMB( "getNextEntry found app=" + appName + " ev=" + std::to_string( current.m_eventCode ) +
                         " logTs=" + logMapDebugTime( after.m_ts ) );
            return 0;
        }
    }

    DEBUG_CRUMB( "getNextEntry no next app=" + appName + " ev=" + std::to_string( current.m_eventCode ) +
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
