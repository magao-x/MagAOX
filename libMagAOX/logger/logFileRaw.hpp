/** \file logFileRaw.hpp
 * \brief Manage a raw log file.
 * \ingroup logger_files
 */

#ifndef logger_logFileRaw_hpp
#define logger_logFileRaw_hpp

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/ioutils/stringUtils.hpp>

#include <flatlogs/flatlogs.hpp>

#include "../file/fileTimes.hpp"

namespace MagAOX
{
namespace logger
{

static_assert( MAGAOX_default_maxLogTime <= MAGAOX_max_maxLogTime,
               "MAGAOX_default_maxLogTime exceeds MAGAOX_max_maxLogTime" );

/// A class to manage raw binary log files
/** Manages a binary file containing MagAO-X logs.
 *
 * A new file is created when an entry is written and any of the following are true:
 * - no file is open
 * - the entry would cause the file to exceed a configurable maximum size, \ref m_maxLogSize
 * - the entry falls in a later wall-clock interval than the current file, \ref m_maxLogTime
 * - a rotation has been requested via INDI with \ref requestRotation
 *
 * Time-based rotation is aligned to the wall clock: the timeline is divided into consecutive intervals of
 * \ref m_maxLogTime minutes measured from the Unix epoch. Rotation only moves forward in time, and never
 * past the current clock's interval, so entries which arrive late or carry a future timestamp are written
 * to the open file rather than starting a new one.
 *
 * Files are only created when an entry is written, so an idle app never creates an empty file, and a gap
 * in the file timestamps indicates a gap in the logged data. Separately, \ref closeIfIntervalElapsed
 * closes an open file once its interval has ended, so a file is not held open while nothing is logged.
 *
 * The size limit acts as a backstop for a verbose app, triggering a new file even if the time limit has 
 * not been reached.
 *
 * Filenames have a standard form of: `[path]/[name]/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext]` where fields in [] are
 * configurable.
 *
 * The timestamp in the file name is from the first entry of the file. It is never later than the
 * time the file was created. If a file of that name already exists the file is named from the current time.
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
     * \returns mx::error_t::noerror on success
     * \returns mx::error_t::erange if newMaxLogTime exceeds \ref MAGAOX_max_maxLogTime, leaving the value unchanged
     */
    mx::error_t maxLogTime( unsigned newMaxLogTime /**< [in] the new value of m_maxLogTime, in minutes */ );

    /// Get the maximum file time span
    /**
     * \returns the current value of m_maxLogTime, in minutes
     */
    unsigned maxLogTime();

    /// Parse a maximum file time span from a string, as given in a config file or over INDI
    /** Accepts any number from 0 to \ref MAGAOX_max_maxLogTime in decimal or scientific notation, optionally
     * surrounded by whitespace, e.g. "15" or "15.0". Decimals are rounded to the nearest minute, apart from 
     * a value which rounds to 0, which is rejected since 0 turns time-based rotation off. The range is
     * checked before converting to unsigned, so a negative value is rejected rather than wrapping. 
     *
     * \returns mx::error_t::noerror on success, with val set
     * \returns mx::error_t::invalidarg if str is not a number, with val unchanged
     * \returns mx::error_t::erange if the value is negative, rounds to 0 without being 0, or exceeds
     *          \ref MAGAOX_max_maxLogTime, with val unchanged
     */
    static mx::error_t parseMaxLogTime( unsigned &val,          ///< [out] the parsed value, in minutes
                                        const std::string &str  ///< [in] the string to parse
    );

    /// Request that a new file be created on the next log entry
    /** This only sets a flag, so is safe to call from any thread.
      * The new file is created by the next call to \ref writeLog. If no further entries are written,
      * no new file is created.
      */
    void requestRotation();

    /// Write a log entry to the file
    /** Opens a new file if no file is open, if this write would exceed m_maxLogSize, if this entry falls in a
     * later wall-clock interval than the current file but not later than the current clock's, or if a rotation
     * has been requested. The new file is named from this entry's timestamp, but never later than the current
     * time, and from the current time if a file of that name already exists.
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

    /// Close the current file if the wall-clock interval it began in has elapsed
    /** This does not create a new file. The next call to \ref writeLog will create one, stamped with
      * that entry's timestamp. Does nothing if no file is open, or if time-based rotation is disabled.
      *
      * \returns mx::error_t::noerror on success
      * \returns an error code on error
      */
    mx::error_t closeIfIntervalElapsed();

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
    if( newMaxLogTime > MAGAOX_max_maxLogTime )
    {
        return mx::error_report<verboseT>( mx::error_t::erange,
                                           "maxLogTime " + std::to_string( newMaxLogTime ) + " exceeds maximum of " +
                                               std::to_string( MAGAOX_max_maxLogTime ) );
    }

    m_maxLogTime = newMaxLogTime;

    return mx::error_t::noerror;
}

template <class verboseT>
unsigned logFileRaw<verboseT>::maxLogTime()
{
    return m_maxLogTime;
}

template <class verboseT>
mx::error_t logFileRaw<verboseT>::parseMaxLogTime( unsigned &val, const std::string &str )
{
    size_t first = str.find_first_not_of( " \t" );

    if( first == std::string::npos )
    {
        return mx::error_t::invalidarg;
    }

    size_t last = str.find_last_not_of( " \t" );

    std::string num = str.substr( first, last - first + 1 );

    // Decimal or scientific notation only
    if( num.find_first_not_of( "0123456789.+-eE" ) != std::string::npos )
    {
        return mx::error_t::invalidarg;
    }

    errno = 0;
    char *end = nullptr;

    double v = std::strtod( num.c_str(), &end );

    if( end != num.c_str() + num.size() ) // not entirely a number
    {
        return mx::error_t::invalidarg;
    }

    if( errno == ERANGE || v < 0 || v > MAGAOX_max_maxLogTime )
    {
        return mx::error_t::erange;
    }

    long r = std::lround( v );

    // Reject a positive decimal that rounds down to 0 since it would silently turn rotation off
    if( r == 0 && v > 0 )
    {
        return mx::error_t::erange;
    }

    val = static_cast<unsigned>( r );

    return mx::error_t::noerror;
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

    flatlogs::timespecX now;
    now.gettime();

    // Only move forward to a later interval and never past the current clock's. Entries which arrive
    // late, or which carry a future timestamp, go into the open file instead of starting a new one.
    bool newInterval = false;
    if( period != 0 )
    {
        uint64_t entryInterval = static_cast<uint64_t>( ts.time_s ) / period;

        newInterval = ( entryInterval > m_currFileStartSec / period &&
                        entryInterval <= static_cast<uint64_t>( now.time_s ) / period );
    }

    // Check if we need a new file
    if( m_fout == 0 || rotateNow || newInterval || m_currFileSize + N > m_maxLogSize )
    {
        // Never name a file after a time later than now, so a file's name is not later than its entries
        flatlogs::timespecX nameTs = ( ts > now ) ? now : ts;

        mx::error_t errc = createFile( nameTs );

        // The name can already be taken, e.g. by an earlier entry with the same timestamp. Instead of failing,
        // which would stop the log thread, name the file from the current time.
        if( errc == mx::error_t::eexist )
        {
            nameTs.gettime();
            errc = createFile( nameTs );
        }

        mx_error_check_code( errc );
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
mx::error_t logFileRaw<verboseT>::closeIfIntervalElapsed()
{
    if( m_fout == 0 )
    {
        return mx::error_t::noerror;
    }

    uint64_t period = m_maxLogTime;
    period *= 60; // minutes to seconds

    if( period == 0 )
    {
        return mx::error_t::noerror;
    }

    flatlogs::timespecX now;
    now.gettime();

    if( static_cast<uint64_t>( now.time_s ) / period != m_currFileStartSec / period )
    {
        return close();
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

            return mx::error_report<verboseT>( mx::errno2error_t( errno ), "Error from fclose" );
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
