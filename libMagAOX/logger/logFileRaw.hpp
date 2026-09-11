/** \file logFileRaw.hpp
 * \brief Manage a raw log file.
 * \ingroup logger_files
 */

#ifndef logger_logFileRaw_hpp
#define logger_logFileRaw_hpp

#include <iostream>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/ioutils/stringUtils.hpp>

#include <flatlogs/flatlogs.hpp>

#include "../file/fileTimes.hpp"
// Test-only fault hooks. Every XWCTEST_IF_ macro expands to an empty statement unless
// a test defines the matching XWCTEST_ name before including this header.
#include "tests/testMacros.hpp"

namespace MagAOX
{
namespace logger
{

// Test-only. A test can define XWCTEST_NAMESPACE and compile this file a second time
// inside that namespace with one XWCTEST_ fault macro enabled. The faulted copy runs the
// real error handling code, and its hits count toward these same source lines.
// Production builds never define XWCTEST_NAMESPACE.
#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// A class to manage raw binary log files
/** Manages a binary file containing MagAO-X logs.
 *
 * The log entries are written as a binary stream of a configurable
 * maximum size.  If this size will be exceed by the next entry, then a new file is created.
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
    ///@}

    /** \name Internal State
     *@{
     */

    FILE *m_fout{ 0 }; ///< The file pointer

    size_t m_currFileSize{ 0 }; ///< The current file size.

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
     * \returns 0 on success
     * \returns -1 on error
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
     * \returns 0 on success
     * \returns -1 on error
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
     * \returns 0 on success
     * \returns -1 on error
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
     * \returns 0 on success
     * \returns -1 on error
     */
    mx::error_t maxLogSize( size_t newMaxFileSize /**< [in] the new value of _maxLogSize */ );

    /// Get the maximum file size
    /**
     * \returns the current value of m_maxLogSize
     */
    size_t maxLogSize();

    /// Write a log entry to the file
    /** Checks if this write will exceed m_maxLogSize, and if so opens a new file.
     * The new file will have the timestamp of this log entry.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    mx::error_t writeLog( flatlogs::bufferPtrT &data /**< [in] the log entry to write to disk */ );

    /// Flush the stream
    /** Calls `fflush`. See issue #192
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    mx::error_t flush();

    /// Close the file pointer
    /** Sets \ref m_fout to nullptr after calling fclose regardless of error.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    mx::error_t close();

  protected:
    /// Create a new file
    /** Closes the current file if open.  Then creates a new file with a name of the form
     * [path]/[name]/YYYY_MM_DD/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext]
     *
     *
     * \returns 0 on success
     * \returns -1 on error
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
        // Test hooks. Each throw runs only when a test enables it, so the handlers below run for real.
        XWCTEST_IF_LOGFILERAW_LOGPATH_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_LOGFILERAW_LOGPATH_EXCEPTION( throw std::exception() );

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
        XWCTEST_IF_LOGFILERAW_LOGNAME_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_LOGFILERAW_LOGNAME_EXCEPTION( throw std::exception() );

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
        XWCTEST_IF_LOGFILERAW_LOGEXT_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_LOGFILERAW_LOGEXT_EXCEPTION( throw std::exception() );

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
mx::error_t logFileRaw<verboseT>::writeLog( flatlogs::bufferPtrT &data )
{
    size_t N = flatlogs::logHeader::totalSize( data );

    // Check if we need a new file
    if( m_currFileSize + N > m_maxLogSize || m_fout == 0 )
    {
        flatlogs::timespecX ts = flatlogs::logHeader::timespec( data );

        mx_error_check( createFile( ts ) );
    }

    size_t nwr = fwrite( data.get(), sizeof( char ), N, m_fout );

    // Test hook. Pretends fwrite wrote nothing.
    XWCTEST_IF_LOGFILERAW_WRITELOG_FWRITE_FAIL( nwr = 0 );

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
        // The return value is kept in a variable so the test hook below can override it.
        int frv = fflush( m_fout );

        // Test hook. Pretends fflush failed with an I/O error.
        XWCTEST_IF_LOGFILERAW_FLUSH_FFLUSH_FAIL( ( errno = EIO, frv = -1 ) );

        if( frv != 0 )
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

        // The return value is kept in a variable so the test hook below can override it.
        int crv = fclose( m_fout );

        // Test hook. Pretends fclose failed with an I/O error.
        XWCTEST_IF_LOGFILERAW_CLOSE_FCLOSE_FAIL( ( errno = EIO, crv = -1 ) );

        if( crv != 0 )
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

        // Test hook. Throws so the catch below wraps and rethrows it.
        XWCTEST_IF_LOGFILERAW_CREATEFILE_EXCEPTION( throw std::bad_alloc() );

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

    // Test hook. Pretends the exists check reported a permission error.
    XWCTEST_IF_LOGFILERAW_CREATEFILE_EXISTS_ERRC( errc = mx::error_t::eacces );

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

    // Test hook. Closes the file that fopen just opened and pretends fopen failed.
    XWCTEST_IF_LOGFILERAW_CREATEFILE_FOPEN_FAIL( if( m_fout != 0 ) { fclose( m_fout ); } m_fout = 0; errno = EACCES; );

    if( m_fout == 0 )
    {
        return mx::error_report<verboseT>( mx::errno2error_t( errno ), "Error from fopen on " + fullPath );
    }

    // Reset counters.
    m_currFileSize = 0;

    return mx::error_t::noerror;
}

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

extern template class logFileRaw<XWC_DEFAULT_VERBOSITY>;

} // namespace logger
} // namespace MagAOX

#endif // logger_logFileRaw_hpp
