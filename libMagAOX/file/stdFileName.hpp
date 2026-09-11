/** \file stdFileName.hpp
 * \brief The stdFileName class for managing standard file names
 * \ingroup file_files
 *
 */

#ifndef file_stdFileName_hpp
#define file_stdFileName_hpp

#include <filesystem>
#include <chrono>
#include <format>

#include <flatlogs/flatlogs.hpp>

#include "../common/defaults.hpp"

#include "../common/exceptions.hpp"

#include "stdSubDir.hpp"
#include "fileTimes.hpp"

// Test-only fault hooks. Every XWCTEST_IF_ macro expands to an empty statement unless
// a test defines the matching XWCTEST_ name before including this header.
#include "tests/testMacros.hpp"

namespace MagAOX
{
namespace file
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Organize and analyze the name of a standard file name
template <typename verboseT = XWC_DEFAULT_VERBOSITY>
class stdFileName
{

  protected:
    std::string m_fullName; ///< The full name of the file, including path
    std::string m_baseName; ///< The base name of the file, not including path

    std::string m_extension; ///< The extension of the file

    std::string m_appName; ///< The name of the application which wrote the file

    stdSubDir<verboseT> m_subDir; ///< The subdirectory of the file

    int m_hour{ 0 };   ///< The hour of the timestamp
    int m_minute{ 0 }; ///< The minute of the timestamp
    int m_second{ 0 }; ///< The second of the timestamp
    int m_nsec{ 0 };   ///< The nanosecond of the timestamp

    flatlogs::timespecX m_timestamp{ 0, 0 }; ///< The timestamp

    bool m_valid{ false }; ///< Whether or not the filename parsed correctly and the components are valid

  public:
    /// Default c'tor
    stdFileName();

    /// Construct from a full name
    /** This calls fullName(const std::string &), which parses the input and
     * populates all fields.
     *
     * On success, sets `m_valid=true`
     *
     *
     *
     * \throws nested MagAOX::xwcException on an exception from fullName.
     *
     */
    explicit stdFileName( const std::string &fullName /**< [in] The new full name of the file (including the path)*/ );

    /// Assignment operator from string
    /** Sets the full name, which is the only way to set any of the values.  This parses the input and
     * populates all fields.
     *
     * On success, sets `m_valid=true`
     *
     * On error, sets `m_valid=false`
     *
     * \returns a reference the `this`
     *
     * \throws nested MagAOX::xwcException on an exception from fullName.
     *
     */
    stdFileName &operator=( const std::string &fullname /**< [in] The new full name of the file
                                                                  (including the path)*/
    );

    /// Sets the full name
    /** Setting the full name is the only way to set any of the values.  This parses the input and
     * populates all fields.
     *
     * \throws nested MagAOX::xwcException on a bad_alloc exception.
     *
     */
    mx::error_t fullName( const std::string &fullName /**< [in] The new full name of the file (including the path)*/ );

    /// Get the current value of m_fullName
    /**
     * \returns the current value of m_fullName
     *
     */
    const std::string &fullName( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_baseName
    /**
     * \returns the current value of m_baseName
     *
     */
    const std::string &baseName( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of
    /**
     * \returns the current value of
     *
     */
    const std::string &extension( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_appName
    /**
     * \returns the current value of m_appName
     *
     */
    const std::string &appName( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_subDir
    /**
     * \returns the current value of m_subDir
     *
     */
    const stdSubDir<verboseT> &subDir( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the year
    /**
     * \returns year() from m_subDir
     *
     */
    int year( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the month
    /**
     * \returns month from m_subDir
     *
     */
    unsigned month( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the day
    /**
     * \returns day from m_subDir
     *
     */
    unsigned day( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_hour
    /**
     * \returns the current value of m_hour
     *
     */
    int hour( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_minute
    /**
     * \returns the current value of m_minute
     *
     */
    int minute( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_second
    /**
     * \returns the current value of m_second
     *
     */
    int second( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_nsec
    /**
     * \returns the current value of m_nsec
     *
     */
    int nsec( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of m_valid
    /**
     * \returns the current value of m_valid
     *
     */
    flatlogs::timespecX timestamp( mx::error_t *errc = nullptr /**< [in] [optional] error code */ ) const;

    /// Get the current value of
    /**
     * \returns the current value of
     *
     */
    bool valid() const;

    /// Set all stored values to invalid values
    void invalidate();
};

template <class verboseT>
stdFileName<verboseT>::stdFileName()
{
    return;
}

template <class verboseT>
stdFileName<verboseT>::stdFileName( const std::string &fn )
{
    try
    {
        mx::error_t errc = fullName( fn );
        if( errc != mx::error_t::noerror )
        {
            mx::error_report( errc, "from fullName" );
        }
    }
    catch( ... )
    {
        std::throw_with_nested( xwcException( "from fullName" ) );
    }
}

template <class verboseT>
stdFileName<verboseT> &stdFileName<verboseT>::operator=( const std::string &fn )
{
    try
    {
        mx::error_t errc = fullName( fn );

        if( errc != mx::error_t::noerror )
        {
            mx::error_report<verboseT>( errc, "from fullName" );
        }

        return *this;
    }
    catch( ... ) // will be bad_alloc
    {
        std::throw_with_nested( xwcException( "from fullName" ) );
    }
}

template <class verboseT>
mx::error_t stdFileName<verboseT>::fullName( const std::string &fn )
{
    // assume it's false beginning at any modification
    invalidate();

    try
    {
        // Test hooks. Each throw runs only when a test enables it, so the handlers below run for real.
        XWCTEST_IF_STDFILENAME_FULLNAME_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDFILENAME_FULLNAME_EXCEPTION( throw std::exception() );

        m_fullName = fn;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "from std::string" ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception, "from std::string" );
    }

    try
    {
        XWCTEST_IF_STDFILENAME_FULLNAME_FS_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDFILENAME_FULLNAME_FS_FILESYSTEM_ERROR(
            throw std::filesystem::filesystem_error( "test", std::error_code( 10, std::system_category() ) ) );
        XWCTEST_IF_STDFILENAME_FULLNAME_FS_EXCEPTION( throw std::exception() );

        std::filesystem::path p( m_fullName );

        m_baseName  = p.filename();
        m_extension = p.extension();
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "extracting basename and extension" ) );
    }
    catch( const std::filesystem::filesystem_error &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_filesystem_error,
                                           "extracting basename and extension " + m_fullName );
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           "extracting basename and extension from " + m_fullName );
    }

    if( m_extension == "" )
    {
        return mx::error_report<verboseT>( mx::error_t::invalidarg, "No extension found in: " + m_fullName );
    }

    std::string YYYY, MM, DD, hh, mm, ss, nn;

    try
    {
        mx_error_check( parseFilePath( m_appName, YYYY, MM, DD, hh, mm, ss, nn, m_baseName ) );
    }
    // Bug fix. parseFilePath() throws mx::exception<verboseT> when it wraps a
    // std::bad_alloc. The old handler caught xwcException, which is a different type,
    // so it never ran and the nested xwcException below was never thrown.
    catch( const mx::exception<verboseT> &e ) // a bad_alloc
    {
        std::throw_with_nested( xwcException( "parsing filename" ) );
    }

    mx::error_t errc;
    int         year = mx::ioutils::stoT<int>( YYYY, &errc );
    mx_error_check_code( errc );

    unsigned int month = mx::ioutils::stoT<unsigned int>( MM, &errc );
    mx_error_check_code( errc );

    unsigned int day = mx::ioutils::stoT<unsigned int>( DD, &errc );
    mx_error_check_code( errc );

    m_hour = mx::ioutils::stoT<int>( hh, &errc );
    mx_error_check_code( errc );

    m_minute = mx::ioutils::stoT<int>( mm, &errc );
    mx_error_check_code( errc );

    m_second = mx::ioutils::stoT<int>( ss, &errc );
    mx_error_check_code( errc );

    m_nsec = mx::ioutils::stoT<int>( nn, &errc );
    mx_error_check_code( errc );

    try
    {
        mx_error_check( m_subDir.ymd( year, month, day ) );
    }
    catch( ... )
    {
        std::throw_with_nested( xwcException( "from stdSubDir::ymd" ) );
    }

    tm tmst;
    tmst.tm_year = year - 1900;
    tmst.tm_mon  = month - 1;
    tmst.tm_mday = day;
    tmst.tm_hour = m_hour;
    tmst.tm_min  = m_minute;
    tmst.tm_sec  = m_second;

    errno      = 0;
    time_t tgm = timegm( &tmst );

    // Test hooks. Pretend timegm failed, once with errno set and once without.
    XWCTEST_IF_STDFILENAME_FULLNAME_TIMEGM( ( tgm = static_cast<time_t>( -1 ), errno = EOVERFLOW ) );
    XWCTEST_IF_STDFILENAME_FULLNAME_TIMEGM_OTHER( tgm = static_cast<time_t>( -1 ) );

    if( tgm == static_cast<time_t>( -1 ) )
    {
        if( errno != 0 )
        {
            return mx::error_report<verboseT>( mx::errno2error_t( errno ), "error from timegm" );
        }

        return mx::error_report<verboseT>( mx::error_t::error, "error from timegm" );
    }

    m_timestamp.time_s  = tgm;
    m_timestamp.time_ns = m_nsec;

    m_valid = true;

    return mx::error_t::noerror;
}

template <class verboseT>
const std::string &stdFileName<verboseT>::fullName( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_fullName;
}

template <class verboseT>
const std::string &stdFileName<verboseT>::baseName( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_baseName;
}

template <class verboseT>
const std::string &stdFileName<verboseT>::extension( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_extension;
}

template <class verboseT>
const std::string &stdFileName<verboseT>::appName( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_appName;
}

template <class verboseT>
const stdSubDir<verboseT> &stdFileName<verboseT>::subDir( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_subDir;
}

template <class verboseT>
int stdFileName<verboseT>::year( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_subDir.year();
}

template <class verboseT>
unsigned int stdFileName<verboseT>::month( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<unsigned int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_subDir.month();
}

template <class verboseT>
unsigned int stdFileName<verboseT>::day( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<unsigned int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_subDir.day();
}

template <class verboseT>
int stdFileName<verboseT>::hour( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_hour;
}

template <class verboseT>
int stdFileName<verboseT>::minute( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_minute;
}

template <class verboseT>
int stdFileName<verboseT>::second( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_second;
}

template <class verboseT>
int stdFileName<verboseT>::nsec( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return std::numeric_limits<int>::max();
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_nsec;
}

template <class verboseT>
flatlogs::timespecX stdFileName<verboseT>::timestamp( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return flatlogs::timespecX( 0, 0 );
    }
    else if( errc )
    {
        *errc = mx::error_t::noerror;
    }

    return m_timestamp;
}

template <class verboseT>
bool stdFileName<verboseT>::valid() const
{
    return m_valid;
}

template <class verboseT>
void stdFileName<verboseT>::invalidate()
{
    m_appName.clear();
    m_baseName.clear();
    m_extension.clear();
    m_baseName.clear();
    m_fullName.clear();

    m_subDir.invalidate();

    m_valid = false;
}

#ifndef XWCTEST_NAMESPACE
extern template class stdFileName<XWC_DEFAULT_VERBOSITY>;
#endif

/// Sort predicate for stdFileNames
/** Sorting is on 'fullName()'
 */
template <class stdFileNameT>
struct compStdFileName
{
    /// Comparison operator.
    /** \returns true if a < b
     * \returns false otherwise
     */
    bool operator()( const stdFileNameT &a, const stdFileNameT &b ) const
    {
        return ( a.baseName() < b.baseName() );
    }
};

#ifdef XWCTEST_NAMESPACE
}
#endif

} // namespace file
} // namespace MagAOX

#endif // file_stdFileName_hpp
