/** \file stdSubDir.hpp
 * \brief The stdSubDir class for managing subdirectories
 *
 * \ingroup file_files
 *
 */

#ifndef file_stdSubDir_hpp
#define file_stdSubDir_hpp

#include <string>
#include <chrono>

#include <mx/mxlib.hpp>
#include <mx/ioutils/stringUtils.hpp>
#include "../common/exceptions.hpp"

#include "../common/defaults.hpp"

// Test-only fault hooks. Every XWCTEST_IF_ macro expands to an empty statement unless
// a test defines the matching XWCTEST_ name before including this header. Each hook
// below throws inside a try block so the real handler after it runs.
#include "tests/testMacros.hpp"

namespace MagAOX
{
namespace file
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Manage a standard subdirectory
/** MagAO-X data storage subdirectories have the format YYYY_MM_DD
 *  This class provides parsing and date arithmetic.
 */
template <typename verboseT = XWC_DEFAULT_VERBOSITY>
class stdSubDir
{

  protected:
    std::chrono::sys_days m_sysday; ///< System time representation of the date of this sub directory

    mutable bool m_pathMade{ false }; ///< Whether or not the path string has been constructed

    mutable std::string m_path; ///< The path string.  Only constructed on demand.

    bool m_valid{ false }; ///< Whether or not the components are valid

  public:
    /// Default c'tor
    stdSubDir();

    /// Construct from std::sys_days
    /**
     * On success, sets `m_valid=true`
     *
     * On error, sets `m_valid=false`
     */
    explicit stdSubDir( const std::chrono::sys_days &sysday /**< [in] The new year*/ );

    /// Construct from components
    /**
     * On success, sets `m_valid=true`
     *
     * On error, sets `m_valid=false`
     */
    stdSubDir( int      year,  /**< [in] The new year*/
               unsigned month, /**< [in] The new month*/
               unsigned day    /**< [in] The new day*/
    );

    /// Construct from a string
    /**
     * On success, sets `m_valid=true`
     *
     * On error, sets `m_valid=false`
     */
    stdSubDir( const std::string &subdir /**< [in] The string of format YYYY_MM_DD*/ );

    /// Setup from components
    /**
     * On success, sets `m_valid=true`
     *
     * On error, sets `m_valid=false`
     */
    mx::error_t ymd( int      year,  /**< [in] The new year*/
                     unsigned month, /**< [in] The new month*/
                     unsigned day    /**< [in] The new day*/
    );

    /// Set the subdirectory
    /**
     * Parses the string and sets the time point
     */
    mx::error_t path( const std::string &subdir );

    /// Get the current value of m_subDir
    /**
     * \returns the current value of m_subDir if valid and path construction succeeds
     * \returns empty string "" if invalid or path construction fails
     */
    std::string path( mx::error_t *errc = nullptr /**< [in] [optional] error code set during path creation */ ) const;

    /// Get the current value of m_year
    /**
     * \returns the current value of m_year if valid
     * \returns std::numeric_limits<int>::max() if invalid
     */
    int year( mx::error_t *errc = nullptr /**< [in] [optional] error code set during path creation */ ) const;

    /// Get the current value of m_month
    /**
     * \returns the current value of m_month if valid
     * \returns std::numeric_limits<unsigned int>::max() if invalid
     */
    unsigned int month( mx::error_t *errc = nullptr /**< [in] [optional] error code set during path creation */ ) const;

    /// Get the current value of m_day
    /**
     * \returns the current value of m_day if valid
     * \returns std::numeric_limits<unsigned int>::max() if invalid
     */
    unsigned int day( mx::error_t *errc = nullptr /**< [in] [optional] error code set during path creation */ ) const;

    /// Get the current value of m_valid
    /**
     * \returns the current value of m_valid
     */
    bool valid() const;

    // Manipulations

    /// Add a day to this subdirectory
    mx::error_t addDay();

    /// Subtract a day from this subdirectory
    mx::error_t subDay();

    /// Get the previous day's subdirectory
    stdSubDir previousSubdir( mx::error_t *errc = nullptr /**< [in] [optional] error code set during path creation */ );

    /// Get the following day's subdirectory
    stdSubDir followingSubdir( mx::error_t *errc = nullptr /**< [in] [optional] error code set during path creation */
    );

    /// Compare two subdirectories for equality by timestamp
    /** Two subdirectories are equal if and only if their timestamps are equal
     *
     */
    bool operator==( const stdSubDir &comp ) const;

    /// Compare two subdirectories for inequality by timestamp
    /** Two subdirectories are equal if and only if their timestamps are equal
     *
     */
    bool operator!=( const stdSubDir &comp ) const;

    /// Compare two subdirectories for less-than by timestamp
    /** A subdirectory is less than if and only if its timestamp is less-than
     *
     */
    bool operator<( const stdSubDir &comp ) const;

    /// Compare two subdirectories for less-than-or-equal by timestamp
    /** A subdirectory is less than or equal if and only if its timestamp is less-than-or-equal
     *
     */
    bool operator<=( const stdSubDir &comp ) const;

    /// Compare two subdirectories for greater-than by timestamp
    /** A subdirectory is greater than if and only if its timestamp is greater-than
     *
     */
    bool operator>( const stdSubDir &comp ) const;

    /// Compare two subdirectories for greater-than-or-equal by timestamp
    /** A subdirectory is greater than if and only if its timestamp is greater-than-or-equal
     *
     */
    bool operator>=( const stdSubDir &comp ) const;

    /// Invalidate this instance
    void invalidate();

};

template <typename verboseT>
stdSubDir<verboseT>::stdSubDir()
{
    return;
}

template <typename verboseT>
stdSubDir<verboseT>::stdSubDir( const std::chrono::sys_days &sysday )
{
    try
    {
        XWCTEST_IF_STDSUBDIR_CTORSYSDAYS_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_CTORSYSDAYS_EXCEPTION( throw std::exception() );

        m_sysday = sysday;
        m_valid  = true;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        mx::error_report<verboseT>( mx::error_t::std_exception, std::string( "chrono operations: " ) + e.what() );
    }
}

template <typename verboseT>
stdSubDir<verboseT>::stdSubDir( int year, unsigned month, unsigned day )
{
    ymd( year, month, day );
}

template <typename verboseT>
stdSubDir<verboseT>::stdSubDir( const std::string &subdir )
{
    path( subdir );
}

template <typename verboseT>
mx::error_t stdSubDir<verboseT>::ymd( int year, unsigned month, unsigned day )
{
    // we technically could be subtle about this and wait for changes to occur
    // but we'll just propagate any errors as invalidating this instance
    invalidate();

    try
    {
        XWCTEST_IF_STDSUBDIR_YMD_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_YMD_EXCEPTION( throw std::exception() );

        std::chrono::year_month_day ymd{
            std::chrono::year( year ), std::chrono::month( month ), std::chrono::day( day ) };

        m_sysday = ymd;

        m_valid = true;
        return mx::error_t::noerror;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           std::string( "chrono operations: " ) + e.what() );
    }
}

template <typename verboseT>
mx::error_t stdSubDir<verboseT>::path( const std::string &subdir )
{
    // we technically could be subtle about this and wait for changes to occur
    // but we'll just propagate any errors as invalidating this instance
    invalidate();

    if( subdir.length() != 10 )
    {
        return mx::error_report<verboseT>( mx::error_t::invalidarg, "subdir " + subdir + " is not 10 chars long " );
    }

    for( size_t n : { 4, 7 } )
    {
        if( subdir[n] != '_' )
        {
            return mx::error_report<verboseT>( mx::error_t::invalidarg, "subdir " + subdir + " is missing _ " );
        }
    }

    for( size_t n : { 0, 1, 2, 3, 5, 6, 8, 9 } )
    {
        if( !isdigit( subdir[n] ) )
        {
            return mx::error_report<verboseT>( mx::error_t::invalidarg,
                                               "subdir " + subdir + " has non-digit at " + std::to_string( n ) );
        }
    }

    mx::error_t errc;

    int          year;
    unsigned int month, day;

    try
    {
        XWCTEST_IF_STDSUBDIR_SETPATH_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_SETPATH_OUT_OF_RANGE( throw std::out_of_range( "test" ) );
        XWCTEST_IF_STDSUBDIR_SETPATH_EXCEPTION( throw std::exception() );

        year = mx::ioutils::stoT<int>( subdir.substr( 0, 4 ), errc );
        mx_error_check_code( errc );

        month = mx::ioutils::stoT<unsigned int>( subdir.substr( 5, 2 ), errc );
        mx_error_check_code( errc );

        day = mx::ioutils::stoT<unsigned int>( subdir.substr( 8, 2 ), errc );
        mx_error_check_code( errc );
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "parsing subdir", -6 ) );
    }
    catch( const std::out_of_range &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_out_of_range, std::string( "parsing subdir" ) + e.what() );
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception, std::string( "parsing subdir" ) + e.what() );
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_SETPATH_BAD_ALLOC2( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_SETPATH_EXCEPTION2( throw std::exception() );

        std::chrono::year_month_day ymd{
            std::chrono::year( year ), std::chrono::month( month ), std::chrono::day( day ) };

        m_sysday = ymd;

        m_path     = subdir;
        m_pathMade = true;

        m_valid = true;

        return mx::error_t::noerror;
    }
    catch( const std::bad_alloc &e )
    {
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           std::string( "chrono operations: " ) + e.what() );
    }
}

template <typename verboseT>
std::string stdSubDir<verboseT>::path( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access path while invalid" );
        return "";
    }

    if( !m_pathMade )
    {
        try
        {
            XWCTEST_IF_STDSUBDIR_MAKEPATH_BAD_ALLOC( throw std::bad_alloc() );
            XWCTEST_IF_STDSUBDIR_MAKEPATH_FORMAT_ERROR( throw std::format_error( "test" ) );
            XWCTEST_IF_STDSUBDIR_MAKEPATH_EXCEPTION( throw std::exception() );

            m_path = std::format( "{:%Y_%m_%d}", m_sysday );

            m_pathMade = true;
        }
        catch( const std::bad_alloc &e )
        {
            std::throw_with_nested( xwcException( "bad_alloc from std::format", -12 ) );
        }
        catch( const std::format_error &e )
        {
            if( errc )
            {
                *errc = mx::error_t::std_format_error;
            }
            mx::error_report<verboseT>( mx::error_t::std_format_error, std::string( "from std::format: " ) + e.what() );
            return "";
        }
        catch( const std::exception &e )
        {
            if( errc )
            {
                *errc = mx::error_t::std_exception;
            }
            mx::error_report<verboseT>( mx::error_t::std_exception, std::string( "from std::format: " ) + e.what() );
            return "";
        }
    }

    if( errc )
    {
        *errc = mx::error_t::noerror;
    }
    return m_path;
}

template <typename verboseT>
int stdSubDir<verboseT>::year( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access year while invalid" );
        return std::numeric_limits<int>::max();
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_GYMD_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_GYMD_EXCEPTION( throw std::exception() );

        std::chrono::year_month_day ymd{ m_sysday };

        if( errc )
        {
            *errc = mx::error_t::noerror;
        }

        return static_cast<int>( ymd.year() );
    }
    catch( const std::bad_alloc &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_bad_alloc;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_exception;
        }
        mx::error_report<verboseT>( mx::error_t::std_exception, std::string( "chrono operations: " ) + e.what() );
        return std::numeric_limits<int>::max();
    }
}

template <typename verboseT>
unsigned int stdSubDir<verboseT>::month( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access month while invalid" );
        return std::numeric_limits<unsigned int>::max();
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_GYMD_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_GYMD_EXCEPTION( throw std::exception() );

        std::chrono::year_month_day ymd{ m_sysday };

        if( errc )
        {
            *errc = mx::error_t::noerror;
        }
        return static_cast<unsigned>( ymd.month() );
    }
    catch( const std::bad_alloc &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_bad_alloc;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_exception;
        }
        mx::error_report<verboseT>( mx::error_t::std_exception, std::string( "chrono operations: " ) + e.what() );
        return std::numeric_limits<unsigned int>::max();
    }
}

template <typename verboseT>
unsigned int stdSubDir<verboseT>::day( mx::error_t *errc ) const
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access day while invalid" );
        return std::numeric_limits<unsigned int>::max();
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_GYMD_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_GYMD_EXCEPTION( throw std::exception() );

        std::chrono::year_month_day ymd{ m_sysday };

        if( errc )
        {
            *errc = mx::error_t::noerror;
        }
        return static_cast<unsigned>( ymd.day() );
    }
    catch( const std::bad_alloc &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_bad_alloc;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_exception;
        }
        mx::error_report<verboseT>( mx::error_t::std_exception, std::string( "chrono operations: " ) + e.what() );
        return std::numeric_limits<unsigned int>::max();
    }
}

template <typename verboseT>
bool stdSubDir<verboseT>::valid() const
{
    return m_valid;
}

template <typename verboseT>
mx::error_t stdSubDir<verboseT>::addDay()
{
    if( !m_valid )
    {
        return mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_INC_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_INC_EXCEPTION( throw std::exception() );

        m_pathMade = false;
        ++m_sysday;
    }
    catch( const std::bad_alloc &e )
    {
        invalidate();
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        invalidate();
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           std::string( "chrono operations: " ) + e.what() );
    }

    return mx::error_t::noerror;
}

template <typename verboseT>
mx::error_t stdSubDir<verboseT>::subDay()
{
    if( !m_valid )
    {
        return mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_INC_BAD_ALLOC( throw std::bad_alloc() );
        XWCTEST_IF_STDSUBDIR_INC_EXCEPTION( throw std::exception() );

        --m_sysday;

        m_pathMade = false;

        return mx::error_t::noerror;
    }
    catch( const std::bad_alloc &e )
    {
        invalidate();
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const std::exception &e )
    {
        invalidate();
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           std::string( "chrono operations: " ) + e.what() );
    }
}

template <typename verboseT>
stdSubDir<verboseT> stdSubDir<verboseT>::previousSubdir( mx::error_t *errc )
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return stdSubDir();
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_PREVFOLL_BAD_ALLOC( throw std::bad_alloc() );

        stdSubDir std = *this;

        // invalidate
        XWCTEST_IF_STDSUBDIR_PREV_INVAL( std.path( "" ) );

        if( !std.valid() )
        {
            if( errc )
            {
                *errc = mx::error_t::error;
            }
            mx::error_report<verboseT>( mx::error_t::error, "an error occurred creating new subdir" );
            return stdSubDir();
        }

        mx::error_t _errc = std.subDay();

        if( _errc != mx::error_t::noerror )
        {
            mx::error_report<verboseT>( _errc, "from subDay" );
        }

        if( errc )
        {
            *errc = _errc;
        }

        return std;
    }
    catch( const std::bad_alloc &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_bad_alloc;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const xwcException &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_exception;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
}

template <typename verboseT>
stdSubDir<verboseT> stdSubDir<verboseT>::followingSubdir( mx::error_t *errc )
{
    if( !m_valid )
    {
        if( errc )
        {
            *errc = mx::error_t::invalidconfig;
        }
        mx::error_report<verboseT>( mx::error_t::invalidconfig, "attempt to access while invalid" );
        return stdSubDir();
    }

    try
    {
        XWCTEST_IF_STDSUBDIR_PREVFOLL_BAD_ALLOC( throw std::bad_alloc() );

        stdSubDir std = *this;

        // invalidate
        XWCTEST_IF_STDSUBDIR_PREV_INVAL( std.path( "" ) );

        if( !std.valid() )
        {
            if( errc )
            {
                *errc = mx::error_t::error;
            }
            mx::error_report<verboseT>( mx::error_t::error, "an error occurred creating new subdir" );
            return stdSubDir();
        }

        mx::error_t _errc = std.addDay();

        if( _errc != mx::error_t::noerror )
        {
            mx::error_report<verboseT>( _errc, "from subDay" );
        }

        if( errc )
        {
            *errc = _errc;
        }

        return std;
    }
    catch( const std::bad_alloc &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_bad_alloc;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
    catch( const xwcException &e )
    {
        if( errc )
        {
            *errc = mx::error_t::std_exception;
        }
        std::throw_with_nested( xwcException( "chrono operations", -6 ) );
    }
}

template <typename verboseT>
bool stdSubDir<verboseT>::operator==( const stdSubDir &comp ) const
{
    return ( m_sysday == comp.m_sysday );
}

template <typename verboseT>
bool stdSubDir<verboseT>::operator!=( const stdSubDir &comp ) const
{
    return ( m_sysday != comp.m_sysday );
}

template <typename verboseT>
bool stdSubDir<verboseT>::operator<( const stdSubDir &comp ) const
{
    return ( m_sysday < comp.m_sysday );
}

template <typename verboseT>
bool stdSubDir<verboseT>::operator<=( const stdSubDir &comp ) const
{
    return ( m_sysday <= comp.m_sysday );
}

template <typename verboseT>
bool stdSubDir<verboseT>::operator>( const stdSubDir &comp ) const
{
    return ( m_sysday > comp.m_sysday );
}

template <typename verboseT>
bool stdSubDir<verboseT>::operator>=( const stdSubDir &comp ) const
{
    return ( m_sysday >= comp.m_sysday );
}

template <typename verboseT>
void stdSubDir<verboseT>::invalidate()
{
    m_path.clear();

    m_pathMade = false;
    m_valid = false;
}

#ifndef XWCTEST_NAMESPACE
extern template class stdSubDir<XWC_DEFAULT_VERBOSITY>;
#endif

#ifdef XWCTEST_NAMESPACE
}
#endif

} // namespace file
} // namespace MagAOX

#endif // file_stdSubDir_hpp
