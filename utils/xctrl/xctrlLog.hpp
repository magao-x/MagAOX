/** \file xctrlLog.hpp
 * \brief Console output and MagAO-X binary logging for xctrl
 *
 * Every message xctrl prints also goes, timestamped, to the MagAO-X binary log of xctrl, as a standard `text_log`
 * entry, so `logdump xctrl` reads it like any app's log.  Only the flatlogs record format and the generated
 * `text_log` code and schema of libMagAOX are used, not its logger, so the setuid xctrl stays free of mxlib.
 *
 * \ingroup xctrl_files
 */

#ifndef xctrlLog_hpp
#define xctrlLog_hpp

#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <grp.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <flatlogs/logHeader.hpp>

#include "../../libMagAOX/common/defaults.hpp"
#include "../../libMagAOX/logger/generated/logCodes.hpp"
#include "../../libMagAOX/logger/types/generated/string_log_generated.h"

namespace xctrl
{

/// The kinds of xctrl output, each with its console color and log priority
/**
 * \ingroup xctrl
 */
enum class msgLevel
{
    info,    ///< plain information, priority INFO
    note,    ///< a note, printed as "note: ...", priority NOTICE
    action,  ///< a command xctrl runs, e.g. systemctl, blue, priority INFO
    success, ///< something done, green, priority INFO
    warning, ///< printed as "warning: ...", yellow, on stderr, priority WARNING
    error    ///< printed as "error: ...", red, on stderr, priority ERROR
};

/// The MagAO-X `text_log` type, for flatlogs::logHeader::createLog, without libMagAOX's logger
/**
 * \ingroup xctrl
 */
struct textLog
{
    static const flatlogs::eventCodeT eventCode    = MagAOX::logger::eventCodes::TEXT_LOG; ///< the text_log code
    static const flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_INFO;          ///< used for LOG_DEFAULT

    /// A message: the text, serialized as a String_log_fb flatbuffer
    struct messageT
    {
        flatbuffers::FlatBufferBuilder builder; ///< holds the serialized message

        /// Serialize the text
        explicit messageT( const std::string &msg /**< [in] the text */ );
    };

    /// The size of the serialized message
    static flatlogs::msgLenT length( const messageT &msg /**< [in] the message */ );

    /// Copy the serialized message into a log entry
    /**
     * \returns 0
     */
    static int format( void           *msgBuffer, /**< [out] the entry's message buffer, of length( msg ) */
                       const messageT &msg /**< [in] the message */ );
};

/// The console output and binary log of one xctrl run
/** Errors and warnings go to stderr, everything else to stdout, colored when that is a terminal.  Once open(), every
 * message is also logged.
 *
 * \ingroup xctrl
 */
class xctrlLog
{
  protected:
    /** \name State - Data
     * @{
     */

    int  m_fd{ -1 };          ///< the log file, opened for appending, -1 if not logging
    bool m_colorOut{ false }; ///< color stdout
    bool m_colorErr{ false }; ///< color stderr

    ///@}

  public:
    /// Choose whether to color the console output
    void colors( bool out, /**< [in] color stdout */
                 bool err /**< [in] color stderr */ );

    /// Open today's log file under logsDir/xctrl/YYYY_MM_DD/, appending to the newest unless it is full
    /** Directories and a new file are created as needed, owned by xsup and group magaox when we are root.  No path
     * component is followed if it is a symbolic link.
     *
     * \returns 0 on success, -1 if the log can not be opened, after a warning on stderr
     */
    int open( const std::string &logsDir /**< [in] the MagAO-X logs directory */ );

    /// Whether messages are being logged
    bool isOpen() const;

    /// Print a message to the console and log it
    void message( msgLevel           level, /**< [in] the kind of message */
                  const std::string &msg,   /**< [in] the text, without a trailing newline */
                  bool               console = true /**< [in] [optional] false to only log it */ );

  protected:
    /// Append one text_log entry to the log
    void logEntry( msgLevel           level, /**< [in] the kind of message */
                   const std::string &msg /**< [in] the text */ );
};

/// The output of this xctrl run
xctrlLog &xlog();

/// Builds one message with <<, and writes it with xlog() when it goes out of scope
/** For example `say( msgLevel::error ) << name << " is not registered";`
 *
 * \ingroup xctrl
 */
class say
{
  protected:
    msgLevel           m_level; ///< the kind of message
    std::ostringstream m_text;  ///< the text so far

  public:
    /// Start a message
    explicit say( msgLevel level /**< [in] the kind of message */ );

    /// Write the message
    ~say();

    /// Append to the message
    template <class T>
    say &operator<<( const T &v /**< [in] what to append */ );
};

// ---------------------------------------------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------------------------------------------

inline textLog::messageT::messageT( const std::string &msg )
{
    auto s = builder.CreateString( msg );
    builder.Finish( MagAOX::logger::CreateString_log_fb( builder, s ) );
}

inline flatlogs::msgLenT textLog::length( const messageT &msg )
{
    return msg.builder.GetSize();
}

inline int textLog::format( void *msgBuffer, const messageT &msg )
{
    memcpy( msgBuffer, msg.builder.GetBufferPointer(), msg.builder.GetSize() );
    return 0;
}

inline void xctrlLog::colors( bool out, bool err )
{
    m_colorOut = out;
    m_colorErr = err;
}

inline int xctrlLog::open( const std::string &logsDir )
{
    if( m_fd >= 0 )
    {
        return 0;
    }

    // as the MagAO-X logger names them, in UTC: YYYY_MM_DD/xctrl_YYYYMMDDhhmmssnnnnnnnnn.binlog
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    tm ut;
    gmtime_r( &ts.tv_sec, &ut );
    char day[16], stamp[40];
    strftime( day, sizeof( day ), "%Y_%m_%d", &ut );
    strftime( stamp, sizeof( stamp ), "%Y%m%d%H%M%S", &ut );
    std::string fileName = std::string( "xctrl_" ) + stamp + std::to_string( 1000000000L + ts.tv_nsec ).substr( 1 ) +
                           "." + MAGAOX_default_logExt;

    bool           root = geteuid() == 0;
    struct passwd *pw   = getpwnam( "xsup" );
    struct group  *gr   = getgrnam( "magaox" );
    uid_t          uid  = pw ? pw->pw_uid : -1;
    gid_t          gid  = gr ? gr->gr_gid : -1;

    // a directory under parent, made if missing, never through a symbolic link
    auto subdir = [&]( int parent, const char *name ) -> int
    {
        bool made = mkdirat( parent, name, 0755 ) == 0;
        int  fd   = openat( parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC );
        if( fd >= 0 && made && root && fchown( fd, uid, gid ) < 0 )
        {
            // the log still works, owned by root
        }
        return fd;
    };

    int top = ::open( logsDir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC );
    int dir = top < 0 ? -1 : subdir( top, "xctrl" );
    int dfd = dir < 0 ? -1 : subdir( dir, day );
    if( top >= 0 )
    {
        close( top );
    }
    if( dir >= 0 )
    {
        close( dir );
    }
    if( dfd < 0 )
    {
        message( msgLevel::warning,
                 "not logging: cannot open " + logsDir + "/xctrl/" + day + ": " + strerror( errno ) );
        return -1;
    }

    // the newest file of the day, unless it is full
    std::string newest;
    int         lfd = dup( dfd );
    DIR        *d   = lfd < 0 ? nullptr : fdopendir( lfd );
    if( d )
    {
        for( dirent *e; ( e = readdir( d ) ); )
        {
            // the names differ only in their timestamps, so the newest sorts last
            std::string n = e->d_name;
            if( n.size() == fileName.size() && n.compare( 0, 6, "xctrl_" ) == 0 &&
                n.compare( 29, std::string::npos, fileName, 29, std::string::npos ) == 0 && n > newest )
            {
                newest = n;
            }
        }
        closedir( d );
    }
    struct stat st;
    if( !newest.empty() && fstatat( dfd, newest.c_str(), &st, AT_SYMLINK_NOFOLLOW ) == 0 && S_ISREG( st.st_mode ) &&
        st.st_size < static_cast<off_t>( MAGAOX_default_max_logSize ) )
    {
        m_fd = openat( dfd, newest.c_str(), O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC );
    }
    if( m_fd < 0 )
    {
        m_fd = openat( dfd, fileName.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644 );
        if( m_fd >= 0 && root && fchown( m_fd, uid, gid ) < 0 )
        {
            // the log still works, owned by root
        }
    }
    int err = errno;
    close( dfd );
    if( m_fd < 0 )
    {
        message( msgLevel::warning,
                 "not logging: cannot open a log file in " + logsDir + "/xctrl/" + day + ": " + strerror( err ) );
        return -1;
    }
    return 0;
}

inline bool xctrlLog::isOpen() const
{
    return m_fd >= 0;
}

inline void xctrlLog::message( msgLevel level, const std::string &msg, bool console )
{
    static const char *colors[] = { "", "", "\033[34m", "\033[32m", "\033[33m", "\033[31m" };
    static const char *labels[] = { "", "note: ", "", "", "warning: ", "error: " };
    int                l        = static_cast<int>( level );

    if( console )
    {
        bool          err   = level == msgLevel::warning || level == msgLevel::error;
        bool          color = ( err ? m_colorErr : m_colorOut ) && colors[l][0];
        std::ostream &os    = err ? std::cerr : std::cout;
        if( err )
        {
            std::cout.flush(); // keep the order of stdout and stderr lines
        }
        os << ( color ? colors[l] : "" ) << labels[l] << msg << ( color ? "\033[0m" : "" ) << std::endl;
    }
    logEntry( level, msg ); // the log has its own priority, so no label
}

inline void xctrlLog::logEntry( msgLevel level, const std::string &msg )
{
    if( m_fd < 0 )
    {
        return;
    }
    static const flatlogs::logPrioT prios[] = { flatlogs::logPrio::LOG_INFO,
                                                flatlogs::logPrio::LOG_NOTICE,
                                                flatlogs::logPrio::LOG_INFO,
                                                flatlogs::logPrio::LOG_INFO,
                                                flatlogs::logPrio::LOG_WARNING,
                                                flatlogs::logPrio::LOG_ERROR };

    flatlogs::timespecX  ts;
    flatlogs::bufferPtrT buf;
    ts.gettime();
    flatlogs::logHeader::createLog<textLog>( buf, ts, textLog::messageT( msg ), prios[static_cast<int>( level )] );

    // one write per entry: with O_APPEND, entries of concurrent xctrl runs do not interleave
    size_t len = flatlogs::logHeader::totalSize( buf );
    if( write( m_fd, buf.get(), len ) != static_cast<ssize_t>( len ) )
    {
        close( m_fd );
        m_fd = -1;
        message( msgLevel::warning, std::string( "stopped logging: " ) + strerror( errno ) );
    }
}

inline xctrlLog &xlog()
{
    static xctrlLog log;
    return log;
}

inline say::say( msgLevel level ) : m_level( level )
{
}

inline say::~say()
{
    xlog().message( m_level, m_text.str() );
}

template <class T>
say &say::operator<<( const T &v )
{
    m_text << v;
    return *this;
}

} // namespace xctrl

#endif // xctrlLog_hpp
