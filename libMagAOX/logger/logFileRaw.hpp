/** \file logFileRaw.hpp
 * \brief Manage a raw log file.
 * \ingroup logger_files
 */

#ifndef logger_logFileRaw_hpp
#define logger_logFileRaw_hpp

#include <atomic>
#include <iostream>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/ioutils/stringUtils.hpp>

#include <flatlogs/flatlogs.hpp>

#include "../file/fileTimes.hpp"

namespace MagAOX
{
namespace logger
{

/// A class to manage raw binary log files
/** Manages a binary file containing MagAO-X logs.
 *
 * A new file is created when any of the following are true:
 * - the next entry would cause the file to exceed a configurable maximum size, \ref m_maxLogSize
 * - the next entry falls in a different wall-clock interval than the current file, \ref m_maxLogTime
 * - a rotation has been requested with \ref requestRotation
 *
 * Time-based rotation is aligned to the wall clock: the timeline is divided into consecutive intervals of
 * \ref m_maxLogTime minutes measured from the Unix epoch, and a new file begins when an entry falls in a
 * different interval than the one the open file began in.
 *
 * Rotation is evaluated lazily when an entry is actually written, so an idle app never creates an
 * empty file. A gap in the file timestamps therefore indicates a gap in the logged data.
 *
 * The size limit acts as a backstop and it can also trigger a new file for a verbose app, even if 
 * the time limit has not been reached.
 *
 * Filenames have a standard form of: `[path]/[name]/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext]` where fields in [] are
 * configurable.
 *
 * The timestamp in the file name is from the first entry of the file.
 *
 */
template <class verboseT = XWC_DEFAULT_VERBOSITY>
class logFileRaw
{

  protected:
    /** \name Configurable Parameters
     *@{
     */
    std::string m_logPath{ "." };                  ///< The base path for the log files.
    std::string m_logName{ "xlog" };               ///< The base name for the log files.
    std::string m_logExt{ MAGAOX_default_logExt }; ///< The extension for the log files.

    size_t m_maxLogSize{ MAGAOX_default_max_logSize }; ///< The maximum file size in bytes. Default is 10 MB.

    /// Atomic because it can be changed at runtime from an INDI callback, while the log thread is reading it.
    std::atomic<unsigned> m_maxLogTime{ MAGAOX_default_maxLogTime }; ///< The maximum time span of a file in minutes. Default is 1440 (24 hours). 0 disables time-based rotation.
    ///@}

    /** \name Internal State
     *@{
     */

    FILE *m_fout{ 0 }; ///< The file pointer

    size_t m_currFileSize{ 0 }; ///< The current file size.

    uint64_t m_currFileStartSec{ 0 }; ///< The timestamp in seconds of the first entry in the current file.

    /// This is atomic because it is set from an INDI callback thread, while the log thread is reading it.
    std::atomic<bool> m_rotateRequested{ false }; ///< Flag indicating that a new file has been requested at runtime.

    ///@}

  public:
    /// Default constructor
    /** Currently does nothing.
     */
    logFileRaw();

    /// Destructor
    /** Closes the file if open
     */
    ~logFileRaw();

    /// Set the path.
    /**
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t logPath( const std::string &newPath /**< [in] the new value of _path */ );

    /// Get the path.
    /**
     * \returns the current value of m_logPath.
     */
    std::string logPath();

    /// Set the log name
    /**
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t logName( const std::string &newName /**< [in] the new value of m_logName */ );

    /// Get the name
    /**
     * \returns the current value of _name.
     */
    std::string logName();

    /// Set the log extension
    /**
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t logExt( const std::string &newExt /**< [in] the new value of m_logExt */ );

    /// Get the log extension
    /**
     * \returns the current value of m_logExt.
     */
    std::string logExt();

    /// Set the maximum file size
    /**
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t maxLogSize( size_t newMaxFileSize /**< [in] the new value of _maxLogSize */ );

    /// Get the maximum file size
    /**
     * \returns the current value of m_maxLogSize
     */
    size_t maxLogSize();

    /// Set the maximum file time span
    /** A value of 0 disables time-based rotation, leaving only the size limit.
     *
     * \returns mx::error_t::noerror
     */
    mx::error_t maxLogTime( unsigned newMaxLogTime /**< [in] the new value of m_maxLogTime, in minutes */ );

    /// Get the maximum file time span
    /**
     * \returns the current value of m_maxLogTime, in minutes
     */
    unsigned maxLogTime();

    /// Request that a new file be created on the next log entry
    /** This only sets a flag, so is safe to call from any thread.
      * The new file is created by the next call to \ref writeLog.  If no further entries are written,
      * no new file is created.
      */
    void requestRotation();

    /// Write a log entry to the file
    /** Opens a new file if this write would exceed m_maxLogSize, if this entry falls in a different
     * wall-clock interval than the current file, or if a rotation has been requested.
     * The new file will have the timestamp of this log entry.
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t writeLog( flatlogs::bufferPtrT &data /**< [in] the log entry to write to disk */ );

    /// Flush the stream
    /** Calls `fflush`. See issue #192
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t flush();

    /// Close the file pointer
    /** Sets \ref m_fout to nullptr after calling fclose regardless of error.
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t close();

  protected:
    /// Create a new file
    /** Closes the current file if open.  Then creates a new file with a name of the form
     * [path]/[name]/YYYY_MM_DD/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext]
     *
     *
     * \returns mx::error_t::noerror on success
     * \returns an error code on error
     */
    mx::error_t createFile( flatlogs::timespecX &ts /**< [in] A MagAOX timespec, used to set the timestamp */ );
};

template <class verboseT>
logFileRaw<verboseT>::logFileRaw()
{
}

template <class verboseT>
logFileRaw<verboseT>::~logFileRaw()
{
    close();
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::logPath( const std::string &newPath )
{
    try
    {
        m_logPath = newPath;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "string assignment" ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report( mx::error_t::std_exception, std::string( "string assignment: " ) + e.what() );
    }

    return mx::error_t::noerror;
}

template <class verboseT>
std::string logFileRaw<verboseT>::logPath()
{
    return m_logPath;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::logName( const std::string &newName )
{
    try
    {
        m_logName = newName;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "string assignment" ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report( mx::error_t::std_exception, std::string( "string assignment: " ) + e.what() );
    }

    return mx::error_t::noerror;
}

template <class verboseT>
std::string logFileRaw<verboseT>::logName()
{
    return m_logName;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::logExt( const std::string &newExt )
{
    try
    {
        m_logExt = newExt;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "string assignment" ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report( mx::error_t::std_exception, std::string( "string assignment: " ) + e.what() );
    }

    return mx::error_t::noerror;
}

template <class verboseT>
std::string logFileRaw<verboseT>::logExt()
{
    return m_logExt;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::maxLogSize( size_t newMaxFileSize )
{
    try
    {
        m_maxLogSize = newMaxFileSize;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "string assignment" ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report( mx::error_t::std_exception, std::string( "string assignment: " ) + e.what() );
    }

    return mx::error_t::noerror;
}

template <class verboseT>
size_t logFileRaw<verboseT>::maxLogSize()
{
    return m_maxLogSize;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::maxLogTime( unsigned newMaxLogTime )
{
    m_maxLogTime = newMaxLogTime;

    return mx::error_t::noerror;
}

template <class verboseT>
unsigned logFileRaw<verboseT>::maxLogTime()
{
    return m_maxLogTime;
}

template <class verboseT>
void logFileRaw<verboseT>::requestRotation()
{
    m_rotateRequested = true;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::writeLog( flatlogs::bufferPtrT &data )
{
    size_t N = flatlogs::logHeader::totalSize( data );

    flatlogs::timespecX ts = flatlogs::logHeader::timespec( data );

    bool rotateNow = m_rotateRequested.exchange( false );

    uint64_t period = m_maxLogTime;
    period *= 60; // minutes to seconds

    // Recompute wall-clock interval for both the new entry and the open file
    bool newInterval =
        ( period != 0 && static_cast<uint64_t>( ts.time_s ) / period != m_currFileStartSec / period );

    // Check if we need a new file
    if( m_fout == 0 || rotateNow || newInterval || m_currFileSize + N > m_maxLogSize )
    {
        mx_error_check( createFile( ts ) );
    }

    size_t nwr = fwrite( data.get(), sizeof( char ), N, m_fout );

    if( nwr != N * sizeof( char ) )
    {
        return mx::error_report<verboseT>( mx::errno2error_t( errno ), "Error from fwrite" );
    }

    m_currFileSize += N;

    return mx::error_t::noerror;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::flush()
{
    ///\todo this probably should be fsync, with appropriate error handling (see fsyncgate) [issue #192]

    if( m_fout )
    {
        if( fflush( m_fout ) != 0 )
        {
            return mx::error_report<verboseT>( mx::errno2error_t( errno ), "Error from fflush" );
        }
    }
    return mx::error_t::noerror;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::close()
{
    if( m_fout )
    {
        errno = 0;

        if( fclose( m_fout ) != 0 )
        {
            m_fout = nullptr;

            return mx::error_report<verboseT>( mx::errno2error_t( errno ), "Error from fflush" );
        }

        m_fout = nullptr;
    }

    return mx::error_t::noerror;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::createFile( flatlogs::timespecX &ts )
{
    std::string fileName;
    std::string relPath;

    try
    {
        mx::error_t errc = file::fileTimeRelPath( fileName, relPath, m_logName, m_logExt, ts.time_s, ts.time_ns );

        if( !!errc )
        {
            return mx::error_report<verboseT>( errc );
        }
    }
    catch( ... )
    {
        std::throw_with_nested( mx::exception<verboseT>(mx::error_t::exception));
    }

    std::string fullPath = m_logPath + '/' + relPath + '/';

    // Create directory
    mx::error_t errc = mx::ioutils::createDirectories( fullPath );

    if( !!errc )
    {
        return mx::error_report<verboseT>( errc, "creating directory" );
    }

    fullPath += fileName;

    if( mx::ioutils::exists( fullPath, errc ) )
    {
        return mx::error_report<verboseT>( mx::error_t::eexist, "file " + fullPath + " exists" );
    }

    if( !!errc )
    {
        return mx::error_report<verboseT>( errc, "checking directory" );
    }

    // Close current file if it's open
    errc = close();
    if( errc != mx::error_t::noerror )
    {
        mx::error_report<verboseT>( errc, "Error from close, attempting to continue");
    }

    errno = 0;

    m_fout = fopen( fullPath.c_str(), "wb" );

    if( m_fout == 0 )
    {
        return mx::error_report<verboseT>( mx::errno2error_t( errno ), "Error from fopen on " + fullPath );
    }

    // Reset counters.
    m_currFileSize = 0;
    m_currFileStartSec = ts.time_s;

    return mx::error_t::noerror;
}

extern template class logFileRaw<XWC_DEFAULT_VERBOSITY>;

} // namespace logger
} // namespace MagAOX

#endif // logger_logFileRaw_hpp
