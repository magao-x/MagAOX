/** \file resuctrl.hpp
 * \brief Manage the MagAO-X apps of a role as systemd services
 *
 * \ingroup resuctrl_files
 */

#ifndef resuctrl_hpp
#define resuctrl_hpp

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <grp.h>
#include <pwd.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../../libMagAOX/common/paths.hpp"

/** \defgroup resuctrl resuctrl: MagAO-X systemd process control
 * \brief Run the apps listed in a role's proclist as systemd services.
 *
 * Each app runs as an instance of one service template per role, `magaox-<role>-app@<name>.service`, with a
 * drop-in generated from the proclist and the app's configuration.  The apps enabled for automatic startup are
 * wanted by `magaox-<role>.target`.
 *
 * \ingroup utils
 */

/** \defgroup resuctrl_files resuctrl Files
 * \ingroup resuctrl
 */

namespace resuctrl
{

/// One process from a proclist line
/**
 * \ingroup resuctrl
 */
struct appEntry
{
    /// The app name, passed to the executable as `-n name`, and the systemd instance name
    std::string name;

    /// The executable, a file name in the MagAO-X bin directory
    std::string exec;

    /// The proclist prefix of the name: "" or "py:" for apps that notify systemd, "-" or "nhb:" for those that do not
    std::string prefix;

    /// The proclist line number, for messages
    int line{ 0 };

    /// Whether the app sends READY=1 and WATCHDOG=1 notifications to systemd
    bool notifies() const;

    /// Whether this is the role's INDI server
    bool isServer() const;

    /// Whether this is an ssh tunnel, which the INDI server starts after
    bool isTunnel() const;
};

/// The result of parsing a proclist
/**
 * \ingroup resuctrl
 */
struct proclist
{
    /// The processes, in proclist order
    std::vector<appEntry> apps;

    /// Problems found, one message each; the lines concerned are not in apps
    std::vector<std::string> errors;
};

/// Key-value settings from a configuration file, keyed by "section.key", or "key" before any section
typedef std::map<std::string, std::string> confMap;

/// systemd directives and their values, keyed by directive name
typedef std::map<std::string, std::string> directiveMap;

/// Kinds of setting values, for validation
enum class valueKind
{
    seconds, ///< a non-negative number of seconds, or infinity
    count,   ///< a non-negative integer
    restart, ///< a systemd Restart= policy
    signal   ///< a signal name allowed for KillSignal=
};

/// One `[systemd]` configuration key and the systemd directive it sets
/**
 * \ingroup resuctrl
 */
struct settingDef
{
    const char *key;         ///< key in the `[systemd]` section of an app or magaox.conf
    const char *directive;   ///< the systemd directive it sets
    bool        unitSection; ///< true if the directive belongs in [Unit], false for [Service]
    valueKind   kind;        ///< how its value is validated
};

/// The `[systemd]` configuration keys resuctrl understands
/**
 * \ingroup resuctrl
 */
inline const std::vector<settingDef> &settingDefs()
{
    static const std::vector<settingDef> defs = {
        { "restart", "Restart", false, valueKind::restart },
        { "restartSec", "RestartSec", false, valueKind::seconds },
        { "startLimitBurst", "StartLimitBurst", true, valueKind::count },
        { "startLimitIntervalSec", "StartLimitIntervalSec", true, valueKind::seconds },
        { "watchdogSec", "WatchdogSec", false, valueKind::seconds },
        { "timeoutStartSec", "TimeoutStartSec", false, valueKind::seconds },
        { "timeoutStopSec", "TimeoutStopSec", false, valueKind::seconds },
        { "killSignal", "KillSignal", false, valueKind::signal } };
    return defs;
}

/// The template defaults, before magaox.conf `[systemd]` settings
/**
 * \ingroup resuctrl
 */
inline directiveMap defaultDirectives()
{
    return { { "Restart", "always" },
             { "RestartSec", "5" },
             { "StartLimitBurst", "10" },
             { "StartLimitIntervalSec", "600" },
             { "WatchdogSec", "10" },
             { "TimeoutStartSec", "120" },
             { "TimeoutStopSec", "30" },
             { "KillSignal", "SIGTERM" } };
}

/// Return s in lower case
std::string lower( const std::string &s /**< [in] the string */ );

/// Return s without leading and trailing white space
std::string trim( const std::string &s /**< [in] the string */ );

/// Whether s is a valid app or executable name: letters, digits, `_`, `.` and `-`, starting with a letter or digit
bool validName( const std::string &s /**< [in] the name */ );

/// Whether value is valid for a setting of the given kind
bool validValue( valueKind          kind, /**< [in] the kind of setting */
                 const std::string &value /**< [in] the value */ );

/// Parse a proclist, as resurrector_indi did
/** Each line with exactly two white-space separated tokens, NAME and EXECUTABLE, is a process.  Lines starting
 * with `#` are comments.  NAME may have a prefix: `-` or `nhb:` for processes that send no heartbeat, `py:` for
 * Python apps; any other `prefix:` is part of the name, which is then invalid.
 *
 * \returns the processes and any errors found
 */
proclist parseProclist( std::istream &in /**< [in] the proclist contents */ );

/// Parse a MagAO-X or TOML-style configuration file into section.key=value pairs
/** Sections are `[name]` lines, values are `key = value` lines, `#` starts a comment.  Quotes and `[ ]` array
 * brackets are stripped from values.  A repeated key keeps its last value.
 *
 * \returns the settings
 */
confMap parseConf( std::istream &in /**< [in] the configuration contents */ );

/// Read and parse a configuration file
/**
 * \returns true if the file was read, false if it could not be opened
 */
bool readConf( confMap           &conf, /**< [out] the settings */
               const std::string &path /**< [in] the file */ );

/// Collect systemd directives from the `[systemd]` section of a configuration
/** Invalid values and unknown keys are reported in errors and skipped.
 */
void systemdSettings( directiveMap             &dirs,   /**< [in.out] directives, updated with the settings found */
                      std::vector<std::string> &errors, /**< [in.out] problems found, appended to */
                      const confMap            &conf,   /**< [in] the configuration */
                      const std::string        &source /**< [in] the configuration's name, for messages */ );

/// Collect the systemd directives of one app from its configuration
/** `[resurrectee] timeout = run[,startup]` sets WatchdogSec and TimeoutStartSec, as it set the resurrector's
 * heartbeat timeouts; a negative run timeout disables the watchdog.  `[systemd]` settings override it.  Apps
 * that do not notify get Type=simple and no watchdog.
 *
 * \returns the directives that differ from the template
 */
directiveMap appDirectives( const appEntry           &app,  /**< [in] the app */
                            const confMap            &conf, /**< [in] the app's configuration */
                            std::vector<std::string> &errors /**< [in.out] problems found, appended to */ );

/// Names of the units of a role
/**
 * \ingroup resuctrl
 */
struct unitNames
{
    std::string role;  ///< the role, as given
    std::string lrole; ///< the role in lower case, used in unit and directory names

    /// Set the role
    explicit unitNames( const std::string &r /**< [in] the role */ );

    /// The service template, magaox-<role>-app@.service
    std::string templ() const;

    /// The target wanting the apps enabled for automatic startup, magaox-<role>.target
    std::string target() const;

    /// The service instance of an app, magaox-<role>-app@<name>.service
    std::string instance( const std::string &name /**< [in] the app name */ ) const;

    /// The app name of a service instance of this role, or "" if unit is not one
    std::string appName( const std::string &unit /**< [in] a unit name */ ) const;
};

/// Append "Directive=value" lines to os for the directives of one section, Type first
void appendDirectives( std::ostringstream &os,   /**< [in.out] the unit being generated */
                       const directiveMap &dirs, /**< [in] the directives */
                       bool                unitSection /**< [in] true for [Unit], false for [Service] */ );

/// Generate the service template of a role
/**
 * \returns the unit file contents
 */
std::string templateUnit( const unitNames    &units, /**< [in] the role's unit names */
                          const directiveMap &dirs,  /**< [in] the default directives, with magaox.conf settings */
                          const std::string  &envFile /**< [in] optional environment file for all apps */ );

/// Generate the target of a role
/**
 * \returns the unit file contents
 */
std::string targetUnit( const unitNames &units /**< [in] the role's unit names */ );

/// Generate the drop-in of one app
/**
 * \returns the drop-in file contents
 */
std::string dropIn( const unitNames                &units,  /**< [in] the role's unit names */
                    const appEntry                 &app,    /**< [in] the app */
                    const std::string              &binDir, /**< [in] directory of the executables */
                    const directiveMap             &dirs,   /**< [in] the app's directives */
                    const std::string              &server, /**< [in] the role's INDI server, "" if none */
                    const std::vector<std::string> &tunnels /**< [in] the role's ssh tunnel apps */ );

/// Generate the registry entry of one app
/**
 * \returns the registry file contents
 */
std::string registryEntry( const appEntry &app /**< [in] the app */ );

/// Write content to path if it differs from what is there, creating directories as needed
/**
 * \returns 1 if written, 0 if unchanged, -1 on error with a message on stderr
 */
int writeIfChanged( const std::string &path,    /**< [in] the file */
                    const std::string &content, /**< [in] its new contents */
                    mode_t             mode /**< [in] its permissions */ );

/// Run a command, without a shell
/** When resuctrl runs setuid root, the command runs either as the calling user, or fully as root.
 *
 * \returns the exit status, or -1 if it could not be run
 */
int runCommand( const std::vector<std::string> &argv, /**< [in] the command and its arguments */
                std::string *output   = nullptr,      /**< [out] [optional] its standard output, else inherited */
                bool         asCaller = false /**< [in] [optional] run as the calling user, not as root */ );

/// How an app's state is reported
/**
 * \ingroup resuctrl
 */
struct appState
{
    int         rank;  ///< sort order: 0 running, 1 disabled, 2 not started, 3 changing, 4 dead
    std::string label; ///< the state in a few words
    const char *color; ///< ANSI color sequence for the state
};

/// Classify an app's service state, with the old resuctrl's colors
/** Running is green, disabled yellow, not started cyan, starting or stopping magenta, failed or dead red.
 *
 * \returns the state
 */
appState classify( const std::map<std::string, std::string> &props, /**< [in] systemctl show properties of the app */
                   bool                                      registered, /**< [in] whether the app is registered */
                   bool                                      enabled /**< [in] whether it is enabled for autostart */ );

/// Command-line interface to manage a role's apps as systemd services
/**
 * \ingroup resuctrl
 */
class resuCtrl
{
  public:
    /// Run resuctrl with its command-line arguments
    /**
     * \returns the process exit status
     */
    int main( int    argc, /**< [in] argument count */
              char **argv /**< [in] arguments */ );

  protected:
    /** \name Options - Data
     * @{
     */

    std::string m_role;               ///< the MagAO-X role, from -r or MAGAOX_ROLE
    std::string m_top{ MAGAOX_path }; ///< the MagAO-X directory, holding bin, config and sys
    std::string m_sysroot;            ///< prefix for /etc files, for testing; systemctl then only enables and disables
    bool        m_dryRun{ false };    ///< print commands and changes instead of making them
    bool        m_all{ false };       ///< --all was given
    bool        m_keepRunning{ false }; ///< reset leaves apps removed from the proclist running
    bool        m_verbose{ false };     ///< status shows a table with systemd and resource details
    int         m_lines{ -1 };          ///< log lines for peek and inspect, -1 for their defaults
    std::string m_prog{ "resuctrl" };   ///< the name we were run as, resuctrl or xctrl
    bool        m_color{ false };       ///< color the output, when it is a terminal and NO_COLOR is not set
    std::string m_verb;                 ///< the command
    std::vector<std::string> m_names;   ///< the app names given

    ///@}

    /** \name Paths
     * @{
     */

    /// Directory of the systemd units
    std::string unitDir() const;

    /// Directory of the app registry, /etc/magaox-<role>/apps
    std::string regDir() const;

    /// The optional environment file for all apps of the role, as seen by systemd
    std::string envFile() const;

    /// The proclist of the role
    std::string proclistPath() const;

    /// The last applied proclist
    std::string activePath() const;

    ///@}

    /// Print usage
    void usage( std::ostream &os /**< [in] where to print */ ) const;

    /// Whether the calling user may start and stop apps: root, xsup, or a member of group magaox
    bool operatorAllowed() const;

    /// Settle privileges for the command before running it
    /** Commands that change which apps are configured need root, i.e. sudo.  Starting and stopping apps needs
     * operatorAllowed(); resuctrl is installed setuid root for that.  Everything else, and any testing option,
     * runs as the calling user.
     *
     * \returns 0 if the command may run, -1 with a message on stderr if not
     */
    int settlePrivileges();

    /// Query systemd for the state of apps
    /**
     * \returns properties of each app's service instance, keyed by app name
     */
    std::map<std::string, std::map<std::string, std::string>>
    showProps( const std::vector<std::string> &names /**< [in] the app names */ );

    /// Parse the command line
    /**
     * \returns 0 if ok, -1 with a message on stderr if not
     */
    int parseArgs( int    argc, /**< [in] argument count */
                   char **argv /**< [in] arguments */ );

    /// Run systemctl, honoring --dry-run and --sysroot
    /**
     * \returns systemctl's exit status, 0 if it was only printed
     */
    int systemctl( const std::vector<std::string> &args, /**< [in] the systemctl arguments */
                   bool offline = false /**< [in] true if it works on --sysroot, e.g. enable */ );

    /// Write a generated file if it changed, or only show it with --dry-run
    /**
     * \returns 1 if written, 0 if unchanged, -1 on error
     */
    int writeFile( const std::string &path,    /**< [in] the file */
                   const std::string &content, /**< [in] its contents */
                   mode_t             mode /**< [in] its permissions */ );

    /// The names of the registered apps
    std::vector<std::string> registered() const;

    /// Read the registry entry of an app
    /**
     * \returns true if the app is registered
     */
    bool registryRead( appEntry          &app, /**< [out] the app */
                       const std::string &name /**< [in] its name */ ) const;

    /// Whether an app is enabled for automatic startup
    bool enabled( const std::string &name /**< [in] the app */ ) const;

    /// Check that the given names are registered apps
    /**
     * \returns the apps, empty with messages on stderr if any is not registered
     */
    std::vector<appEntry> namedApps();

    /// The service instances of the role that are active or activating
    std::vector<std::string> activeApps();

    /// Start apps, the INDI server after the others
    /**
     * \returns 0 if all started, else non-zero
     */
    int startApps( const std::vector<appEntry> &apps /**< [in] the apps */ );

    /// Stop apps, the INDI server after the others
    /**
     * \returns 0 if all stopped, else non-zero
     */
    int stopApps( const std::vector<std::string> &names /**< [in] the app names */ );

    /** \name Commands
     * @{
     */

    int cmdReset();    ///< Register the proclist's apps with systemd and reconcile what changed
    int cmdStartup();  ///< Start enabled apps, or enable and start named ones
    int cmdShutdown(); ///< Stop all apps, or disable and stop named ones
    int cmdStart();    ///< Start named apps
    int cmdStop();     ///< Stop named apps
    int cmdRestart();  ///< Restart named apps, or all
    int cmdStatus();   ///< Report the apps' service states, one line each, or a detailed table with -v
    int cmdPeek();     ///< Report state and the last log lines of apps
    int cmdInspect();  ///< Follow an app's console output in the journal
    int cmdEnable();   ///< Enable or disable named apps for automatic startup
    int cmdDelete();   ///< Stop and deregister named apps
    int cmdBoot();     ///< Start the role at boot, or not

    ///@}
};

// ---------------------------------------------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------------------------------------------

inline bool appEntry::notifies() const
{
    return prefix != "-" && prefix != "nhb:";
}

inline bool appEntry::isServer() const
{
    return exec == "xindiserver";
}

inline bool appEntry::isTunnel() const
{
    return exec == "sshDigger";
}

inline std::string lower( const std::string &s )
{
    std::string r( s );
    for( auto &c : r )
    {
        c = static_cast<char>( tolower( static_cast<unsigned char>( c ) ) );
    }
    return r;
}

inline std::string trim( const std::string &s )
{
    size_t b = s.find_first_not_of( " \t\r\n" );
    if( b == std::string::npos )
    {
        return "";
    }
    size_t e = s.find_last_not_of( " \t\r\n" );
    return s.substr( b, e - b + 1 );
}

inline bool validName( const std::string &s )
{
    if( s.empty() || s.size() > 128 || !isalnum( static_cast<unsigned char>( s[0] ) ) )
    {
        return false;
    }
    for( char c : s )
    {
        if( !isalnum( static_cast<unsigned char>( c ) ) && c != '_' && c != '.' && c != '-' )
        {
            return false;
        }
    }
    return true;
}

inline bool validValue( valueKind kind, const std::string &v )
{
    static const std::set<std::string> restarts = {
        "no", "always", "on-success", "on-failure", "on-abnormal", "on-abort", "on-watchdog" };
    static const std::set<std::string> signals = {
        "SIGTERM", "SIGINT", "SIGQUIT", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGKILL" };

    switch( kind )
    {
    case valueKind::seconds:
    {
        if( v == "infinity" )
        {
            return true;
        }
        bool dot = false;
        for( size_t i = 0; i < v.size(); ++i )
        {
            if( v[i] == '.' && !dot && i > 0 && i + 1 < v.size() )
            {
                dot = true;
            }
            else if( !isdigit( static_cast<unsigned char>( v[i] ) ) )
            {
                return false;
            }
        }
        return !v.empty();
    }
    case valueKind::count:
        return !v.empty() && v.size() < 10 &&
               std::all_of( v.begin(), v.end(), []( char c ) { return isdigit( static_cast<unsigned char>( c ) ); } );
    case valueKind::restart:
        return restarts.count( v ) > 0;
    case valueKind::signal:
        return signals.count( v ) > 0;
    }
    return false;
}

inline proclist parseProclist( std::istream &in )
{
    proclist              pl;
    std::set<std::string> seen;
    std::string           line;
    int                   n = 0;

    while( std::getline( in, line ) )
    {
        ++n;
        std::istringstream       ss( line );
        std::string              tok, extra;
        std::vector<std::string> toks;
        while( ss >> tok )
        {
            toks.push_back( tok );
        }

        if( toks.empty() || toks[0][0] == '#' || ( toks.size() > 1 && toks[1][0] == '#' ) )
        {
            continue; // blank line or comment
        }

        if( toks.size() != 2 )
        {
            pl.errors.push_back( "proclist line " + std::to_string( n ) +
                                 ": expected NAME EXECUTABLE, ignored: " + trim( line ) );
            continue;
        }

        appEntry app;
        app.line = n;
        app.exec = toks[1];
        app.name = toks[0];
        if( app.name[0] == '-' )
        {
            app.prefix = "-";
            app.name.erase( 0, 1 );
        }
        else if( app.name.compare( 0, 3, "py:" ) == 0 || app.name.compare( 0, 4, "nhb:" ) == 0 )
        {
            size_t c   = app.name.find( ':' );
            app.prefix = app.name.substr( 0, c + 1 );
            app.name.erase( 0, c + 1 );
        }

        if( !validName( app.name ) )
        {
            pl.errors.push_back( "proclist line " + std::to_string( n ) + ": invalid name [" + toks[0] +
                                 "]: use letters, digits, _ . and -, with a py:, nhb: or - prefix if needed" );
            continue;
        }
        if( !validName( app.exec ) )
        {
            pl.errors.push_back( "proclist line " + std::to_string( n ) + ": invalid executable [" + app.exec +
                                 "]: give a file name in the MagAO-X bin directory" );
            continue;
        }
        if( !seen.insert( app.name ).second )
        {
            pl.errors.push_back( "proclist line " + std::to_string( n ) + ": duplicate name [" + app.name + "]" );
            continue;
        }

        pl.apps.push_back( app );
    }

    return pl;
}

inline confMap parseConf( std::istream &in )
{
    confMap     conf;
    std::string section, line;

    while( std::getline( in, line ) )
    {
        size_t h = line.find( '#' );
        if( h != std::string::npos )
        {
            line.erase( h );
        }
        line = trim( line );
        if( line.empty() )
        {
            continue;
        }

        if( line.front() == '[' && line.back() == ']' )
        {
            section = trim( line.substr( 1, line.size() - 2 ) );
            continue;
        }

        size_t eq = line.find( '=' );
        if( eq == std::string::npos )
        {
            continue;
        }

        std::string key = trim( line.substr( 0, eq ) );
        std::string val = trim( line.substr( eq + 1 ) );
        if( val.size() >= 2 &&
            ( ( val.front() == '"' && val.back() == '"' ) || ( val.front() == '\'' && val.back() == '\'' ) ||
              ( val.front() == '[' && val.back() == ']' ) ) )
        {
            val = trim( val.substr( 1, val.size() - 2 ) );
        }

        conf[section.empty() ? key : section + "." + key] = val;
    }

    return conf;
}

inline bool readConf( confMap &conf, const std::string &path )
{
    std::ifstream f( path );
    if( !f )
    {
        return false;
    }
    conf = parseConf( f );
    return true;
}

inline void
systemdSettings( directiveMap &dirs, std::vector<std::string> &errors, const confMap &conf, const std::string &source )
{
    std::set<std::string> known;
    for( const auto &d : settingDefs() )
    {
        known.insert( std::string( "systemd." ) + d.key );

        auto it = conf.find( std::string( "systemd." ) + d.key );
        if( it == conf.end() )
        {
            continue;
        }
        if( !validValue( d.kind, it->second ) )
        {
            errors.push_back( source + ": invalid [systemd] " + d.key + " = " + it->second );
        }
        else
        {
            dirs[d.directive] = it->second;
        }
    }

    for( const auto &kv : conf )
    {
        if( kv.first.compare( 0, 8, "systemd." ) == 0 && known.count( kv.first ) == 0 )
        {
            errors.push_back( source + ": unknown setting [systemd] " + kv.first.substr( 8 ) );
        }
    }
}

inline directiveMap appDirectives( const appEntry &app, const confMap &conf, std::vector<std::string> &errors )
{
    directiveMap dirs;
    std::string  source = app.name + ".conf";

    // the resurrector's heartbeat timeouts: running, then optionally startup
    auto it = conf.find( "resurrectee.timeout" );
    if( it != conf.end() )
    {
        std::vector<long> t;
        std::stringstream ss( it->second );
        std::string       item;
        bool              ok = true;
        while( std::getline( ss, item, ',' ) )
        {
            item      = trim( item );
            char *end = nullptr;
            long  v   = strtol( item.c_str(), &end, 10 );
            ok        = ok && !item.empty() && *end == '\0';
            t.push_back( v );
        }

        if( !ok || t.empty() )
        {
            errors.push_back( source + ": invalid [resurrectee] timeout = " + it->second );
        }
        else
        {
            // as the resurrectee: negative disables the heartbeat, below 1 means the default of 10
            dirs["WatchdogSec"] = t[0] < 0 ? "0" : std::to_string( t[0] < 1 ? 10 : t[0] );
            if( t.size() > 1 && t[1] > 0 )
            {
                dirs["TimeoutStartSec"] = std::to_string( t[1] );
            }
        }
    }

    systemdSettings( dirs, errors, conf, source );

    if( !app.notifies() )
    {
        dirs["Type"]        = "simple";
        dirs["WatchdogSec"] = "0";
    }

    return dirs;
}

inline unitNames::unitNames( const std::string &r ) : role( r ), lrole( lower( r ) )
{
}

inline std::string unitNames::templ() const
{
    return "magaox-" + lrole + "-app@.service";
}

inline std::string unitNames::target() const
{
    return "magaox-" + lrole + ".target";
}

inline std::string unitNames::instance( const std::string &name ) const
{
    return "magaox-" + lrole + "-app@" + name + ".service";
}

inline std::string unitNames::appName( const std::string &unit ) const
{
    std::string pre = "magaox-" + lrole + "-app@", suf = ".service";
    if( unit.size() <= pre.size() + suf.size() || unit.compare( 0, pre.size(), pre ) != 0 ||
        unit.compare( unit.size() - suf.size(), suf.size(), suf ) != 0 )
    {
        return "";
    }
    return unit.substr( pre.size(), unit.size() - pre.size() - suf.size() );
}

inline void appendDirectives( std::ostringstream &os, const directiveMap &dirs, bool unitSection )
{
    // Type first, it reads best there
    if( !unitSection && dirs.count( "Type" ) )
    {
        os << "Type=" << dirs.at( "Type" ) << "\n";
    }
    for( const auto &d : settingDefs() )
    {
        if( d.unitSection == unitSection && dirs.count( d.directive ) )
        {
            os << d.directive << "=" << dirs.at( d.directive ) << "\n";
        }
    }
}

inline std::string templateUnit( const unitNames &units, const directiveMap &dirs, const std::string &envFile )
{
    std::ostringstream os;
    os << "# Generated by resuctrl reset for MAGAOX_ROLE=" << units.role << ".  Do not edit: set [systemd] in\n"
       << "# magaox.conf, or in an app's .conf for that app, then rerun resuctrl reset.\n"
       << "[Unit]\n"
       << "Description=MagAO-X app %i (role " << units.role << ")\n"
       << "PartOf=" << units.target() << "\n"
       << "Wants=network-online.target\n"
       << "After=network-online.target\n";
    appendDirectives( os, dirs, true );
    os << "\n[Service]\n"
       << "Type=notify\n"
       << "NotifyAccess=main\n"
       << "User=xsup\n"
       << "Environment=MAGAOX_ROLE=" << units.role << "\n"
       << "EnvironmentFile=-" << envFile << "\n"
       << "# Each registered app's drop-in sets its ExecStart; an unregistered instance fails\n"
       << "ExecStart=/bin/false\n";
    appendDirectives( os, dirs, false );
    os << "KillMode=control-group\n"
       << "SyslogIdentifier=%i\n"
       << "\n[Install]\n"
       << "WantedBy=" << units.target() << "\n";
    return os.str();
}

inline std::string targetUnit( const unitNames &units )
{
    std::ostringstream os;
    os << "# Generated by resuctrl reset for MAGAOX_ROLE=" << units.role << ".  Do not edit.\n"
       << "# Wants the apps enabled for automatic startup; resuctrl boot on|off starts it at boot or not.\n"
       << "[Unit]\n"
       << "Description=MagAO-X apps of role " << units.role << "\n"
       << "\n[Install]\n"
       << "WantedBy=multi-user.target\n";
    return os.str();
}

inline std::string dropIn( const unitNames                &units,
                           const appEntry                 &app,
                           const std::string              &binDir,
                           const directiveMap             &dirs,
                           const std::string              &server,
                           const std::vector<std::string> &tunnels )
{
    std::ostringstream os;
    os << "# Generated by resuctrl reset from proclist_" << units.role << ".txt and " << app.name << ".conf.\n"
       << "# Do not edit: change those, then rerun resuctrl reset.\n"
       << "[Unit]\n";
    if( app.isServer() )
    {
        // the INDI server's remote links go through the tunnels
        for( const auto &t : tunnels )
        {
            os << "After=" << units.instance( t ) << "\n";
        }
    }
    else if( !server.empty() && !app.isTunnel() )
    {
        // so apps stop before the INDI server, and can say goodbye through it
        os << "After=" << units.instance( server ) << "\n";
    }
    appendDirectives( os, dirs, true );
    os << "\n[Service]\n"
       << "ExecStart=\n"
       << "ExecStart=" << binDir << "/" << app.exec << " -n " << app.name << "\n";
    appendDirectives( os, dirs, false );
    return os.str();
}

inline std::string registryEntry( const appEntry &app )
{
    return "# resuctrl registry entry, written by resuctrl reset\nname=" + app.name + "\nexec=" + app.exec +
           "\nprefix=" + app.prefix + "\n";
}

inline int writeIfChanged( const std::string &path, const std::string &content, mode_t mode )
{
    {
        std::ifstream f( path );
        if( f )
        {
            std::stringstream ss;
            ss << f.rdbuf();
            if( ss.str() == content )
            {
                return 0;
            }
        }
    }

    std::error_code ec;
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path(), ec );
    if( ec )
    {
        std::cerr << "resuctrl: creating directory for " << path << ": " << ec.message() << "\n";
        return -1;
    }

    std::string tmp = path + ".resuctrl.tmp";
    {
        std::ofstream f( tmp, std::ios::trunc );
        f << content;
        if( !f.good() )
        {
            std::cerr << "resuctrl: writing " << tmp << ": " << strerror( errno ) << "\n";
            return -1;
        }
    }
    if( chmod( tmp.c_str(), mode ) < 0 || rename( tmp.c_str(), path.c_str() ) < 0 )
    {
        std::cerr << "resuctrl: installing " << path << ": " << strerror( errno ) << "\n";
        unlink( tmp.c_str() );
        return -1;
    }
    return 1;
}

inline int runCommand( const std::vector<std::string> &argv, std::string *output, bool asCaller )
{
    int pfd[2] = { -1, -1 };
    if( output && pipe( pfd ) < 0 )
    {
        return -1;
    }

    pid_t pid = fork();
    if( pid < 0 )
    {
        return -1;
    }
    if( pid == 0 )
    {
        if( output )
        {
            dup2( pfd[1], 1 );
            close( pfd[0] );
            close( pfd[1] );
        }
        if( geteuid() == 0 )
        {
            uid_t u = asCaller ? getuid() : 0;
            gid_t g = asCaller ? getgid() : 0;
            if( setresgid( g, g, g ) < 0 || setresuid( u, u, u ) < 0 )
            {
                _exit( 126 );
            }
        }
        std::vector<char *> args;
        for( const auto &a : argv )
        {
            args.push_back( const_cast<char *>( a.c_str() ) );
        }
        args.push_back( nullptr );
        execvp( args[0], args.data() );
        std::cerr << "resuctrl: running " << argv[0] << ": " << strerror( errno ) << "\n";
        _exit( 127 );
    }

    if( output )
    {
        close( pfd[1] );
        char    buf[4096];
        ssize_t nr;
        output->clear();
        while( ( nr = read( pfd[0], buf, sizeof( buf ) ) ) != 0 )
        {
            if( nr < 0 )
            {
                if( errno == EINTR )
                {
                    continue;
                }
                break;
            }
            output->append( buf, nr );
        }
        close( pfd[0] );
    }

    int status;
    while( waitpid( pid, &status, 0 ) < 0 )
    {
        if( errno != EINTR )
        {
            return -1;
        }
    }
    return WIFEXITED( status ) ? WEXITSTATUS( status ) : -1;
}

inline appState classify( const std::map<std::string, std::string> &props, bool registered, bool enabled )
{
    static const char *green = "\033[32m", *yellow = "\033[33m", *cyan = "\033[36m", *magenta = "\033[35m",
                      *red = "\033[31m";

    auto get = [&]( const char *k ) -> std::string
    {
        auto it = props.find( k );
        return it == props.end() ? "" : it->second;
    };
    std::string active = get( "ActiveState" ), sub = get( "SubState" ), result = get( "Result" );

    if( active.empty() )
    {
        return { 2, "unknown", cyan };
    }
    if( active == "active" && sub == "running" )
    {
        return registered ? appState{ 0, "running", green } : appState{ 3, "running, not registered", magenta };
    }
    if( active == "activating" )
    {
        return { 3, "starting", magenta };
    }
    if( active == "deactivating" )
    {
        return { 3, "stopping", magenta };
    }
    if( active == "failed" || ( !result.empty() && result != "success" ) )
    {
        return { 4, "dead (" + result + ")", red };
    }
    if( active == "inactive" )
    {
        return enabled ? appState{ 2, "not started", cyan } : appState{ 1, "disabled", yellow };
    }
    return { 3, active + "/" + sub, magenta };
}

inline std::string resuCtrl::unitDir() const
{
    return m_sysroot + "/etc/systemd/system";
}

inline std::string resuCtrl::regDir() const
{
    return m_sysroot + "/etc/magaox-" + lower( m_role ) + "/apps";
}

inline std::string resuCtrl::envFile() const
{
    return "/etc/magaox-" + lower( m_role ) + "/environment";
}

inline std::string resuCtrl::proclistPath() const
{
    return m_top + "/" + MAGAOX_configRelPath + "/proclist_" + m_role + ".txt";
}

inline std::string resuCtrl::activePath() const
{
    return m_top + "/" + MAGAOX_sysRelPath + "/resurrector_proclist/proclist_" + m_role + ".txt.active";
}

inline void resuCtrl::usage( std::ostream &os ) const
{
    os << "usage: " << m_prog << " [-r ROLE] [options] COMMAND [NAME ...]\n"
       << "\n"
          "MagAO-X process control: run the apps of a role (MAGAOX_ROLE, or -r) as systemd services, one\n"
          "instance of magaox-<role>-app@.service per app in /opt/MagAOX/config/proclist_<role>.txt.\n"
          "resuctrl and xctrl are the same program.\n"
          "\n"
          "Any user can look; xsup and members of group magaox can start and stop apps:\n"
          "  status [NAME...]       state and pid of the apps; -v adds autostart, restarts, uptime,\n"
          "                         last watchdog ping, CPU, memory and GPU memory\n"
          "  peek [NAME...]         state and last log lines of the apps (-n LINES, default 10)\n"
          "  inspect NAME           follow an app's console output (-n LINES, default 50)\n"
          "  startup [NAME...]      start the apps enabled for automatic startup, the INDI server last;\n"
          "                         with names, start those apps\n"
          "  start NAME...          start apps, enabled or not; systemd supervises them either way\n"
          "  stop NAME...           stop apps; they stay stopped until started again\n"
          "  shutdown NAME...|--all stop apps, or all running apps of the role, the INDI server last\n"
          "  restart NAME...|--all  restart apps, or stop then start everything\n"
          "  defib NAME...          same as restart; a hung app is already restarted by its watchdog\n"
          "\n"
          "Changing which apps are configured needs sudo:\n"
          "  sudo resuctrl reset    register the proclist's apps, generate their drop-ins from their .conf,\n"
          "                         enable them for automatic startup, disable (and stop) apps no longer\n"
          "                         listed; new apps start at once if the role is up\n"
          "  sudo resuctrl enable NAME...   enable apps for automatic startup, until the next reset\n"
          "  sudo resuctrl disable NAME...  disable apps for automatic startup, until the next reset\n"
          "  sudo resuctrl delete NAME...   stop, disable and deregister apps\n"
          "  sudo resuctrl boot on|off      start the role's enabled apps at boot, or not\n"
          "\n"
          "Options:\n"
          "  -r, --role ROLE      the MagAO-X role, else MAGAOX_ROLE or /etc/profile.d/magaox_role.sh\n"
          "  --all                all apps\n"
          "  --keep-running       reset leaves apps removed from the proclist running\n"
          "  -n, --lines N        log lines for peek and inspect\n"
          "  --dry-run            print what would be done, change nothing\n"
          "  --top DIR            MagAO-X directory, default " MAGAOX_path ", with --sysroot or --dry-run\n"
          "  --sysroot DIR        write /etc files under DIR and only run offline systemctl, for testing\n"
          "  -h, --help           this help\n";
}

inline bool resuCtrl::operatorAllowed() const
{
    uid_t uid = getuid();
    if( uid == 0 )
    {
        return true;
    }
    struct passwd *pw = getpwnam( "xsup" );
    if( pw && pw->pw_uid == uid )
    {
        return true;
    }
    struct group *gr = getgrnam( "magaox" );
    if( !gr )
    {
        return false;
    }
    if( getgid() == gr->gr_gid )
    {
        return true;
    }
    int n = getgroups( 0, nullptr );
    if( n <= 0 )
    {
        return false;
    }
    std::vector<gid_t> groups( n );
    n = getgroups( n, groups.data() );
    return std::find( groups.begin(), groups.begin() + std::max( n, 0 ), gr->gr_gid ) !=
           groups.begin() + std::max( n, 0 );
}

inline int resuCtrl::settlePrivileges()
{
    static const std::set<std::string> admin     = { "reset", "enable", "disable", "delete", "deregister", "boot" };
    static const std::set<std::string> operate   = { "startup", "start", "shutdown", "stop", "restart", "defib" };
    bool                               setuidRun = geteuid() == 0 && getuid() != 0;
    bool                               testing = m_dryRun || !m_sysroot.empty() || m_top != std::string( MAGAOX_path );

    // a setuid program trusts nothing from the caller's environment
    if( setuidRun )
    {
        clearenv();
        setenv( "PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/MagAOX/bin", 1 );
        setenv( "SYSTEMD_PAGER", "", 1 );
        setenv( "PAGER", "cat", 1 );
    }

    bool keepRoot = false;
    if( testing )
    {
        keepRoot = false;
    }
    else if( admin.count( m_verb ) )
    {
        if( getuid() != 0 )
        {
            std::cerr << m_prog << ": " << m_verb << " changes which apps are configured, run it with sudo: sudo "
                      << m_prog << " " << m_verb << " ...\n";
            return -1;
        }
        keepRoot = true;
    }
    else if( operate.count( m_verb ) )
    {
        if( !operatorAllowed() )
        {
            std::cerr << m_prog << ": only xsup and members of group magaox can " << m_verb << " apps\n";
            return -1;
        }
        keepRoot = true;
    }
    else
    {
        // looking: the journal is readable as root only for those who may operate the apps
        keepRoot = operatorAllowed();
    }

    if( setuidRun && !keepRoot )
    {
        gid_t g = getgid();
        uid_t u = getuid();
        if( setresgid( g, g, g ) < 0 || setresuid( u, u, u ) < 0 )
        {
            std::cerr << m_prog << ": dropping privileges: " << strerror( errno ) << "\n";
            return -1;
        }
    }
    return 0;
}

inline int resuCtrl::parseArgs( int argc, char **argv )
{
    for( int i = 1; i < argc; ++i )
    {
        std::string a     = argv[i];
        auto        value = [&]( std::string &dest ) -> bool
        {
            if( i + 1 >= argc )
            {
                std::cerr << m_prog << ": " << a << " requires a value\n";
                return false;
            }
            dest = argv[++i];
            return true;
        };

        if( a == "-h" || a == "--help" )
        {
            m_verb = "help";
            return 0;
        }
        else if( a == "-r" || a == "--role" )
        {
            if( !value( m_role ) )
                return -1;
        }
        else if( a.compare( 0, 7, "--role=" ) == 0 )
        {
            m_role = a.substr( 7 );
        }
        else if( a == "--top" )
        {
            if( !value( m_top ) )
                return -1;
        }
        else if( a == "--sysroot" )
        {
            if( !value( m_sysroot ) )
                return -1;
        }
        else if( a == "-n" || a == "--lines" )
        {
            std::string n;
            if( !value( n ) )
                return -1;
            m_lines = atoi( n.c_str() );
        }
        else if( a == "--dry-run" )
        {
            m_dryRun = true;
        }
        else if( a == "--all" )
        {
            m_all = true;
        }
        else if( a == "-v" || a == "--verbose" )
        {
            m_verbose = true;
        }
        else if( a == "--keep-running" )
        {
            m_keepRunning = true;
        }
        else if( a.size() > 1 && a[0] == '-' )
        {
            std::cerr << m_prog << ": unknown option " << a << "; try --help\n";
            return -1;
        }
        else if( m_verb.empty() )
        {
            m_verb = a;
        }
        else
        {
            m_names.push_back( a );
        }
    }

    if( m_role.empty() && getenv( "MAGAOX_ROLE" ) )
    {
        m_role = getenv( "MAGAOX_ROLE" );
    }
    if( m_role.empty() )
    {
        // sudo drops MAGAOX_ROLE, so also look where it is usually set
        std::ifstream f( "/etc/profile.d/magaox_role.sh" );
        std::string   line;
        while( std::getline( f, line ) )
        {
            size_t p = line.find( "MAGAOX_ROLE=" );
            if( p != std::string::npos )
            {
                std::string v = trim( line.substr( p + 12 ) );
                v.erase( std::remove( v.begin(), v.end(), '"' ), v.end() );
                v.erase( std::remove( v.begin(), v.end(), '\'' ), v.end() );
                m_role = trim( v.substr( 0, v.find_first_of( " ;\t" ) ) );
            }
        }
    }

    if( m_verb.empty() )
    {
        std::cerr << m_prog << ": missing command; try --help\n";
        return -1;
    }
    if( m_top != MAGAOX_path && m_sysroot.empty() && !m_dryRun )
    {
        std::cerr << m_prog << ": --top is for testing, with --sysroot or --dry-run\n";
        return -1;
    }
    if( m_role.empty() || !validName( m_role ) )
    {
        std::cerr << m_prog << ": missing or invalid role [" << m_role << "]: give -r ROLE or set MAGAOX_ROLE\n";
        return -1;
    }
    for( const auto &n : m_names )
    {
        if( !validName( n ) && m_verb != "boot" )
        {
            std::cerr << m_prog << ": invalid app name [" << n << "]\n";
            return -1;
        }
    }
    return 0;
}

inline int resuCtrl::systemctl( const std::vector<std::string> &args, bool offline )
{
    std::vector<std::string> argv{ "systemctl" };
    if( !m_sysroot.empty() )
    {
        argv.push_back( "--root=" + m_sysroot );
    }
    argv.insert( argv.end(), args.begin(), args.end() );

    std::string line;
    for( const auto &a : argv )
    {
        line += ( line.empty() ? "" : " " ) + a;
    }

    if( m_dryRun || ( !m_sysroot.empty() && !offline ) )
    {
        std::cout << ( m_dryRun ? "[dry-run] " : "[sysroot, not run] " ) << line << "\n";
        return 0;
    }
    std::cout << "+ " << line << std::endl;
    return runCommand( argv );
}

inline std::vector<std::string> resuCtrl::registered() const
{
    std::vector<std::string> names;
    std::error_code          ec;
    for( const auto &e : std::filesystem::directory_iterator( regDir(), ec ) )
    {
        if( e.path().extension() == ".conf" )
        {
            names.push_back( e.path().stem().string() );
        }
    }
    std::sort( names.begin(), names.end() );
    return names;
}

inline bool resuCtrl::registryRead( appEntry &app, const std::string &name ) const
{
    confMap c;
    if( !readConf( c, regDir() + "/" + name + ".conf" ) || c["name"] != name )
    {
        return false;
    }
    app.name   = name;
    app.exec   = c["exec"];
    app.prefix = c["prefix"];
    return true;
}

inline bool resuCtrl::enabled( const std::string &name ) const
{
    unitNames       u( m_role );
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::symlink_status( unitDir() + "/" + u.target() + ".wants/" + u.instance( name ), ec ) );
}

inline std::vector<appEntry> resuCtrl::namedApps()
{
    std::vector<appEntry> apps;
    bool                  ok = !m_names.empty();
    if( !ok )
    {
        std::cerr << m_prog << ": " << m_verb << " needs app names\n";
    }
    for( const auto &n : m_names )
    {
        appEntry app;
        if( registryRead( app, n ) )
        {
            apps.push_back( app );
        }
        else
        {
            std::cerr << m_prog << ": " << n << " is not registered: add it to " << proclistPath()
                      << " and run sudo resuctrl reset\n";
            ok = false;
        }
    }
    return ok ? apps : std::vector<appEntry>();
}

inline std::vector<std::string> resuCtrl::activeApps()
{
    std::vector<std::string> names;
    unitNames                u( m_role );
    if( m_dryRun || !m_sysroot.empty() )
    {
        return names;
    }

    std::string out;
    runCommand( { "systemctl",
                  "list-units",
                  "--all",
                  "--plain",
                  "--no-legend",
                  "--no-pager",
                  "magaox-" + u.lrole + "-app@*.service" },
                &out );
    std::istringstream ss( out );
    std::string        line;
    while( std::getline( ss, line ) )
    {
        std::istringstream ls( line );
        std::string        unit, load, active;
        ls >> unit >> load >> active;
        std::string name = u.appName( unit );
        if( !name.empty() && ( active == "active" || active == "activating" || active == "reloading" ) )
        {
            names.push_back( name );
        }
    }
    return names;
}

inline int resuCtrl::startApps( const std::vector<appEntry> &apps )
{
    unitNames                u( m_role );
    std::vector<std::string> others{ "start" }, server{ "start" }, reset{ "reset-failed" };
    for( const auto &a : apps )
    {
        ( a.isServer() ? server : others ).push_back( u.instance( a.name ) );
        reset.push_back( u.instance( a.name ) );
    }

    // an explicit start also clears a failure, e.g. after the start limit was hit
    systemctl( reset );

    // systemctl start waits for the apps to report READY=1, then the INDI server starts
    int rc = 0;
    if( others.size() > 1 )
    {
        rc |= systemctl( others );
    }
    if( server.size() > 1 )
    {
        rc |= systemctl( server );
    }
    return rc;
}

inline int resuCtrl::stopApps( const std::vector<std::string> &names )
{
    unitNames                u( m_role );
    std::vector<std::string> others{ "stop" }, server{ "stop" };
    for( const auto &n : names )
    {
        appEntry a;
        bool     isServer = registryRead( a, n ) ? a.isServer() : n.compare( 0, 2, "is" ) == 0;
        ( isServer ? server : others ).push_back( u.instance( n ) );
    }

    // the INDI server stops last, so the apps' last messages get through
    int rc = 0;
    if( others.size() > 1 )
    {
        rc |= systemctl( others );
    }
    if( server.size() > 1 )
    {
        rc |= systemctl( server );
    }
    return rc;
}

inline int resuCtrl::writeFile( const std::string &path, const std::string &content, mode_t mode )
{
    if( m_dryRun )
    {
        std::ifstream     f( path );
        std::stringstream ss;
        if( f )
        {
            ss << f.rdbuf();
        }
        if( f && ss.str() == content )
        {
            return 0;
        }
        std::cout << "[dry-run] would write " << path << ":\n" << content << "\n";
        return 1;
    }
    return writeIfChanged( path, content, mode );
}

inline int resuCtrl::cmdReset()
{
    unitNames                u( m_role );
    std::vector<std::string> errors;

    std::ifstream pf( proclistPath() );
    if( !pf )
    {
        std::cerr << m_prog << ": cannot read proclist " << proclistPath() << "\n";
        return 1;
    }
    proclist pl = parseProclist( pf );
    errors      = pl.errors;

    // role-wide defaults, from magaox.conf [systemd]
    std::string  cfgDir = m_top + "/" + MAGAOX_configRelPath;
    directiveMap gdirs  = defaultDirectives();
    confMap      gconf;
    readConf( gconf, cfgDir + "/" + MAGAOX_globalConfig );
    systemdSettings( gdirs, errors, gconf, MAGAOX_globalConfig );

    // the INDI server and the apps must agree on the control FIFO
    confMap isconf;
    if( readConf( isconf, cfgDir + "/is" + m_role + ".conf" ) && isconf.count( "indiserver.f" ) &&
        gconf.count( "indiserver_ctrl_fifo" ) && isconf["indiserver.f"] != gconf["indiserver_ctrl_fifo"] )
    {
        std::cerr << m_prog << ": warning: [indiserver] f = " << isconf["indiserver.f"] << " in is" << m_role
                  << ".conf differs from indiserver_ctrl_fifo = " << gconf["indiserver_ctrl_fifo"]
                  << " in magaox.conf\n";
    }

    // validate each app; invalid ones keep any earlier registration and state
    struct valid
    {
        appEntry     app;
        directiveMap dirs;
    };
    std::vector<valid>       apps;
    std::string              server;
    std::vector<std::string> tunnels;
    std::string              binDir = m_top + "/bin";

    for( const auto &app : pl.apps )
    {
        std::string exe = binDir + "/" + app.exec;
        struct stat st;
        if( stat( exe.c_str(), &st ) < 0 || !S_ISREG( st.st_mode ) || access( exe.c_str(), X_OK ) < 0 )
        {
            errors.push_back( app.name + ": executable " + exe + " is missing or not executable, not registered" );
            continue;
        }

        confMap c;
        if( !readConf( c, cfgDir + "/" + app.name + ".conf" ) )
        {
            std::cout << m_prog << ": note: " << app.name << " has no " << app.name << ".conf, using defaults\n";
        }
        std::vector<std::string> aerr;
        directiveMap             d = appDirectives( app, c, aerr );
        if( !aerr.empty() )
        {
            errors.insert( errors.end(), aerr.begin(), aerr.end() );
            errors.push_back( app.name + ": not registered because of its configuration errors" );
            continue;
        }

        if( app.isServer() )
        {
            if( !server.empty() )
            {
                errors.push_back( app.name + ": a second INDI server, " + server + " is the role's, not registered" );
                continue;
            }
            server = app.name;
        }
        if( app.isTunnel() )
        {
            tunnels.push_back( app.name );
        }
        apps.push_back( { app, d } );
    }

    std::vector<std::string> before = registered();
    std::set<std::string>    wasRegistered( before.begin(), before.end() );

    // the role's template and target, then each app's drop-in and registry entry
    int rc = 0;
    rc |= writeFile( unitDir() + "/" + u.templ(), templateUnit( u, gdirs, envFile() ), 0644 ) < 0;
    rc |= writeFile( unitDir() + "/" + u.target(), targetUnit( u ), 0644 ) < 0;

    std::vector<std::string> changed, toEnable;
    std::vector<appEntry>    newlyEnabled;
    for( const auto &v : apps )
    {
        int w = writeFile( unitDir() + "/" + u.instance( v.app.name ) + ".d/override.conf",
                           dropIn( u, v.app, binDir, v.dirs, server, tunnels ),
                           0644 );
        rc |= w < 0;
        rc |= writeFile( regDir() + "/" + v.app.name + ".conf", registryEntry( v.app ), 0644 ) < 0;
        if( w == 1 && wasRegistered.count( v.app.name ) )
        {
            changed.push_back( v.app.name );
        }
        if( !enabled( v.app.name ) )
        {
            toEnable.push_back( u.instance( v.app.name ) );
            newlyEnabled.push_back( v.app );
        }
    }
    if( rc )
    {
        std::cerr << m_prog << ": could not write the systemd configuration; run with sudo\n";
        return 1;
    }

    // registered apps no longer in the proclist leave automatic startup but stay registered
    std::set<std::string> listed;
    for( const auto &a : pl.apps )
    {
        listed.insert( a.name );
    }
    std::vector<std::string> removed, toDisable;
    for( const auto &n : before )
    {
        if( !listed.count( n ) && enabled( n ) )
        {
            removed.push_back( n );
            toDisable.push_back( u.instance( n ) );
        }
    }

    systemctl( { "daemon-reload" } );
    if( !toEnable.empty() )
    {
        toEnable.insert( toEnable.begin(), "enable" );
        rc |= systemctl( toEnable, true ) != 0;
    }
    if( !toDisable.empty() )
    {
        toDisable.insert( toDisable.begin(), "disable" );
        rc |= systemctl( toDisable, true ) != 0;
    }

    if( !removed.empty() && !m_keepRunning )
    {
        std::vector<std::string> act = activeApps(), stop;
        for( const auto &n : removed )
        {
            if( std::find( act.begin(), act.end(), n ) != act.end() )
            {
                stop.push_back( n );
            }
        }
        if( !stop.empty() )
        {
            rc |= stopApps( stop ) != 0;
        }
    }

    // a role that is up starts its new apps now
    bool up = !m_dryRun && m_sysroot.empty() && runCommand( { "systemctl", "is-active", "--quiet", u.target() } ) == 0;
    if( up && !newlyEnabled.empty() )
    {
        rc |= startApps( newlyEnabled ) != 0;
    }

    // record what was applied
    std::ostringstream active;
    active << "# Applied by resuctrl reset from " << proclistPath() << "; do not edit\n";
    for( const auto &v : apps )
    {
        active << v.app.prefix << v.app.name << " " << v.app.exec << "\n";
    }
    if( writeFile( activePath(), active.str(), 0664 ) < 0 )
    {
        rc = 1;
    }

    std::cout << m_prog << ": role " << m_role << ": " << apps.size() << " apps registered and enabled, "
              << newlyEnabled.size() << " newly";
    if( !removed.empty() )
    {
        std::cout << "; removed from automatic startup:";
        for( const auto &n : removed )
        {
            std::cout << " " << n;
        }
    }
    std::cout << "\n";
    for( const auto &n : changed )
    {
        std::cout << m_prog << ": " << n << ": settings changed, applied at its next start; resuctrl restart " << n
                  << " applies them now\n";
    }
    for( const auto &e : errors )
    {
        std::cerr << m_prog << ": error: " << e << "\n";
    }

    return ( rc || !errors.empty() ) ? 1 : 0;
}

inline int resuCtrl::cmdStartup()
{
    // with names, as start
    if( !m_names.empty() )
    {
        return cmdStart();
    }

    unitNames             u( m_role );
    std::vector<appEntry> apps;
    for( const auto &n : registered() )
    {
        appEntry a;
        if( enabled( n ) && registryRead( a, n ) )
        {
            apps.push_back( a );
        }
    }
    if( apps.empty() )
    {
        std::cerr << m_prog << ": no apps enabled for role " << m_role << "; run sudo resuctrl reset\n";
        return 1;
    }
    int rc = startApps( apps );
    return rc | systemctl( { "start", u.target() } );
}

inline int resuCtrl::cmdShutdown()
{
    // with names, as stop
    if( !m_names.empty() )
    {
        return cmdStop();
    }
    if( !m_all )
    {
        std::cerr << m_prog << ": specify --all to stop all apps, or give app names\n";
        return 1;
    }

    // everything running, including apps started by hand that are not enabled
    unitNames                u( m_role );
    std::vector<std::string> names = ( m_dryRun || !m_sysroot.empty() ) ? registered() : activeApps();
    int                      rc    = systemctl( { "stop", u.target() } );
    if( !names.empty() )
    {
        rc |= stopApps( names );
    }
    return rc;
}

inline int resuCtrl::cmdStart()
{
    std::vector<appEntry> apps = namedApps();
    return apps.empty() ? 1 : startApps( apps );
}

inline int resuCtrl::cmdStop()
{
    std::vector<appEntry> apps = namedApps();
    if( apps.empty() )
    {
        return 1;
    }
    std::vector<std::string> names;
    for( const auto &a : apps )
    {
        names.push_back( a.name );
    }
    return stopApps( names );
}

inline int resuCtrl::cmdRestart()
{
    if( m_verb == "defib" )
    {
        std::cout << m_prog << ": defib restarts apps; a hung app is restarted by its systemd watchdog\n";
    }

    if( m_all || m_names.empty() )
    {
        if( !m_all )
        {
            std::cerr << m_prog << ": restart needs app names, or --all\n";
            return 1;
        }
        int rc = cmdShutdown();
        return rc | cmdStartup();
    }

    std::vector<appEntry> apps = namedApps();
    if( apps.empty() )
    {
        return 1;
    }
    unitNames                u( m_role );
    std::vector<std::string> others{ "restart" }, server{ "restart" };
    for( const auto &a : apps )
    {
        ( a.isServer() ? server : others ).push_back( u.instance( a.name ) );
    }
    int rc = 0;
    if( others.size() > 1 )
    {
        rc |= systemctl( others );
    }
    if( server.size() > 1 )
    {
        rc |= systemctl( server );
    }
    return rc;
}

inline std::map<std::string, std::map<std::string, std::string>>
resuCtrl::showProps( const std::vector<std::string> &names )
{
    std::map<std::string, std::map<std::string, std::string>> props;
    if( m_dryRun || !m_sysroot.empty() || names.empty() )
    {
        return props;
    }

    // one systemctl show for all: blocks of key=value separated by blank lines
    unitNames                u( m_role );
    std::vector<std::string> argv{
        "systemctl",
        "show",
        "--no-pager",
        "-p",
        "Id,ActiveState,SubState,MainPID,NRestarts,Result,WatchdogUSec,WatchdogTimestampMonotonic,"
        "ExecMainStartTimestampMonotonic,CPUUsageNSec,MemoryCurrent,ControlGroup" };
    for( const auto &n : names )
    {
        argv.push_back( u.instance( n ) );
    }
    std::string out;
    runCommand( argv, &out );

    std::istringstream                 ss( out );
    std::string                        line;
    std::map<std::string, std::string> cur;
    auto                               flush = [&]()
    {
        if( cur.count( "Id" ) )
        {
            props[u.appName( cur["Id"] )] = cur;
        }
        cur.clear();
    };
    while( std::getline( ss, line ) )
    {
        if( line.empty() )
        {
            flush();
            continue;
        }
        size_t eq = line.find( '=' );
        if( eq != std::string::npos )
        {
            cur[line.substr( 0, eq )] = line.substr( eq + 1 );
        }
    }
    flush();
    return props;
}

inline int resuCtrl::cmdStatus()
{
    std::vector<std::string> names = m_names;
    std::set<std::string>    regs;
    for( const auto &n : registered() )
    {
        regs.insert( n );
    }
    if( names.empty() )
    {
        names.assign( regs.begin(), regs.end() );
        for( const auto &n : activeApps() )
        {
            if( !regs.count( n ) )
            {
                names.push_back( n ); // running, but deregistered
            }
        }
    }
    if( names.empty() )
    {
        std::cout << m_prog << ": no apps registered for role " << m_role << "; run sudo resuctrl reset\n";
        return 0;
    }

    auto props = showProps( names );

    // sorted by state then name, colored, as the old resuctrl
    struct row
    {
        appState    st;
        std::string name;
    };
    std::vector<row> rows;
    for( const auto &n : names )
    {
        rows.push_back( { classify( props[n], regs.count( n ) > 0, enabled( n ) ), n } );
    }
    std::stable_sort( rows.begin(), rows.end(), []( const row &a, const row &b ) { return a.st.rank < b.st.rank; } );

    const char *reset = m_color ? "\033[0m" : "";
    auto        pidOf = [&]( const std::string &n ) -> std::string
    {
        auto &p = props[n];
        return p.count( "MainPID" ) && p["MainPID"] != "0" ? p["MainPID"] : "";
    };

    if( !m_verbose )
    {
        // name: state (pid: N), as the old resuctrl and xctrl
        for( const auto &r : rows )
        {
            std::string pid = pidOf( r.name );
            std::cout << ( m_color ? r.st.color : "" ) << r.name << ": " << r.st.label
                      << ( pid.empty() ? "" : " (pid: " + pid + ")" ) << reset << "\n";
        }
        return 0;
    }

    // a number from systemctl show, or -1 if it is not set, e.g. when accounting is off
    auto num = []( std::map<std::string, std::string> &p, const char *key ) -> double
    {
        if( !p.count( key ) || p[key].empty() || !isdigit( static_cast<unsigned char>( p[key][0] ) ) ||
            p[key] == "18446744073709551615" )
        {
            return -1;
        }
        return strtod( p[key].c_str(), nullptr );
    };
    auto monoUsec = []() -> double
    {
        struct timespec ts;
        clock_gettime( CLOCK_MONOTONIC, &ts );
        return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
    };

    // CPU use now, from two samples of each app's CPU time a quarter second apart
    std::map<std::string, double> cpuPct;
    double                        t0 = monoUsec();
    if( !props.empty() )
    {
        usleep( 250000 );
        auto   again = showProps( names );
        double dt    = monoUsec() - t0;
        for( const auto &n : names )
        {
            double c0 = num( props[n], "CPUUsageNSec" ), c1 = num( again[n], "CPUUsageNSec" );
            if( c0 >= 0 && c1 >= c0 && !pidOf( n ).empty() )
            {
                cpuPct[n] = 100 * ( c1 - c0 ) / 1e3 / dt;
            }
        }
    }
    double now = monoUsec();

    // GPU memory, if this host has NVIDIA GPUs, summed over each app's processes
    std::map<std::string, double> gpuMiB;
    std::string                   smi;
    bool                          haveSmi = false;
    std::string                   path    = getenv( "PATH" ) ? getenv( "PATH" ) : "/usr/bin:/bin";
    for( size_t b = 0, e; b <= path.size() && !haveSmi; b = e + 1 )
    {
        e       = std::min( path.find( ':', b ), path.size() );
        haveSmi = access( ( path.substr( b, e - b ) + "/nvidia-smi" ).c_str(), X_OK ) == 0;
    }
    if( !props.empty() && haveSmi &&
        runCommand( { "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits" },
                    &smi,
                    true ) == 0 )
    {
        std::map<long, double> byPid;
        std::istringstream     ss( smi );
        std::string            line;
        while( std::getline( ss, line ) )
        {
            size_t c = line.find( ',' );
            if( c != std::string::npos )
            {
                byPid[atol( line.c_str() )] += strtod( line.c_str() + c + 1, nullptr );
            }
        }
        for( const auto &n : names )
        {
            // all the app's processes, from its cgroup, else just its main process
            std::set<long> pids;
            std::ifstream  procs( "/sys/fs/cgroup" + props[n]["ControlGroup"] + "/cgroup.procs" );
            long           pid;
            while( !props[n]["ControlGroup"].empty() && procs >> pid )
            {
                pids.insert( pid );
            }
            if( pids.empty() && !pidOf( n ).empty() )
            {
                pids.insert( atol( pidOf( n ).c_str() ) );
            }
            for( long q : pids )
            {
                if( byPid.count( q ) )
                {
                    gpuMiB[n] += byPid[q];
                }
            }
        }
    }

    char buf[512];
    snprintf( buf,
              sizeof( buf ),
              "%-24s %-22s %8s %-9s %8s %10s %9s %8s %7s %9s %9s\n",
              "NAME",
              "STATE",
              "PID",
              "AUTOSTART",
              "RESTARTS",
              "UPTIME(s)",
              "PING(s)",
              "WATCHDOG",
              "CPU%",
              "MEM(MiB)",
              "GPU(MiB)" );
    std::cout << buf;
    for( const auto &r : rows )
    {
        auto       &p   = props[r.name];
        std::string pid = pidOf( r.name );
        std::string wd = "-", up = "-", ping = "-", cpu = "-", mem = "-", gpu = "-";
        if( p.count( "WatchdogUSec" ) && p["WatchdogUSec"] != "0" && p["WatchdogUSec"] != "infinity" )
        {
            wd = p["WatchdogUSec"];
        }
        char v[64];
        if( !pid.empty() )
        {
            double start = num( p, "ExecMainStartTimestampMonotonic" );
            if( start > 0 )
            {
                snprintf( v, sizeof( v ), "%.0f", ( now - start ) / 1e6 );
                up = v;
            }
            double last = num( p, "WatchdogTimestampMonotonic" );
            if( last > 0 && wd != "-" )
            {
                snprintf( v, sizeof( v ), "%.1f", ( now - last ) / 1e6 ); // seconds since the last WATCHDOG=1
                ping = v;
            }
            if( cpuPct.count( r.name ) )
            {
                snprintf( v, sizeof( v ), "%.1f", cpuPct[r.name] );
                cpu = v;
            }
            double m = num( p, "MemoryCurrent" );
            if( m >= 0 )
            {
                snprintf( v, sizeof( v ), "%.1f", m / 1048576 );
                mem = v;
            }
            if( gpuMiB.count( r.name ) )
            {
                snprintf( v, sizeof( v ), "%.0f", gpuMiB[r.name] );
                gpu = v;
            }
        }
        snprintf( buf,
                  sizeof( buf ),
                  "%-24s %-22s %8s %-9s %8s %10s %9s %8s %7s %9s %9s",
                  r.name.c_str(),
                  r.st.label.c_str(),
                  pid.empty() ? "-" : pid.c_str(),
                  !regs.count( r.name ) ? "-" : ( enabled( r.name ) ? "enabled" : "disabled" ),
                  p.count( "NRestarts" ) ? p["NRestarts"].c_str() : "-",
                  up.c_str(),
                  ping.c_str(),
                  wd.c_str(),
                  cpu.c_str(),
                  mem.c_str(),
                  gpu.c_str() );
        std::cout << ( m_color ? r.st.color : "" ) << buf << reset << "\n";
    }
    return 0;
}

inline int resuCtrl::cmdPeek()
{
    unitNames                u( m_role );
    std::vector<std::string> names = m_names.empty() ? registered() : m_names;
    int                      lines = m_lines < 0 ? 10 : m_lines;
    auto                     props = showProps( names );

    for( const auto &n : names )
    {
        auto       &p      = props[n];
        std::string active = p.count( "ActiveState" ) ? p["ActiveState"] : "unknown";
        appState    st     = classify( p, true, enabled( n ) );
        std::cout << ( m_color ? st.color : "" ) << n << ": " << st.label;
        if( p.count( "MainPID" ) && p["MainPID"] != "0" )
        {
            std::cout << " (pid: " << p["MainPID"] << ")";
        }
        std::cout << ( m_color ? "\033[0m" : "" ) << std::endl;

        if( m_dryRun )
        {
            std::cout << "[dry-run] logdump -n 1 " << n << " | tail -n " << lines << "\n";
            continue;
        }

        // the app's own log, read as the caller, as xctrl did
        std::string out;
        runCommand( { "logdump", "-n", "1", n }, &out, true );
        std::vector<std::string> all;
        std::istringstream       ss( out );
        std::string              line;
        while( std::getline( ss, line ) )
        {
            all.push_back( line );
        }
        for( size_t i = all.size() > (size_t)lines ? all.size() - lines : 0; i < all.size(); ++i )
        {
            std::cout << all[i] << "\n";
        }

        // and its console output when it is not running, which says why
        if( active != "active" && active != "activating" && active != "unknown" )
        {
            std::cout << "--- console output (" << m_prog << " inspect " << n << " to follow):" << std::endl;
            runCommand( { "journalctl", "-u", u.instance( n ), "-n", std::to_string( lines ), "--no-pager", "-q" } );
        }
        std::cout << "End " << n << " logs\n\n";
    }
    return 0;
}

inline int resuCtrl::cmdInspect()
{
    if( m_names.size() != 1 )
    {
        std::cerr << m_prog << ": inspect needs one app name\n";
        return 1;
    }
    unitNames u( m_role );
    if( m_dryRun )
    {
        std::cout << "[dry-run] journalctl -u " << u.instance( m_names[0] ) << " -f\n";
        return 0;
    }
    return runCommand( { "journalctl",
                         "-u",
                         u.instance( m_names[0] ),
                         "-n",
                         std::to_string( m_lines < 0 ? 50 : m_lines ),
                         "--no-pager",
                         "-f" } );
}

inline int resuCtrl::cmdEnable()
{
    std::vector<appEntry> apps = namedApps();
    if( apps.empty() )
    {
        return 1;
    }
    unitNames                u( m_role );
    std::vector<std::string> args{ m_verb };
    for( const auto &a : apps )
    {
        args.push_back( u.instance( a.name ) );
    }
    int rc = systemctl( args, true );
    std::cout << m_prog << ": note: the next resuctrl reset enables exactly the apps in " << proclistPath() << "\n";
    return rc;
}

inline int resuCtrl::cmdDelete()
{
    std::vector<appEntry> apps = namedApps();
    if( apps.empty() )
    {
        return 1;
    }
    unitNames                u( m_role );
    std::vector<std::string> names, dis{ "disable" };
    for( const auto &a : apps )
    {
        names.push_back( a.name );
        dis.push_back( u.instance( a.name ) );
    }

    // stop while systemd still knows how
    int rc = stopApps( names );
    rc |= systemctl( dis, true );
    for( const auto &n : names )
    {
        std::string d = unitDir() + "/" + u.instance( n ) + ".d", r = regDir() + "/" + n + ".conf";
        if( m_dryRun )
        {
            std::cout << "[dry-run] would remove " << d << " and " << r << "\n";
            continue;
        }
        std::error_code ec;
        std::filesystem::remove( d + "/override.conf", ec );
        std::filesystem::remove( d, ec ); // only if nothing else is there
        std::filesystem::remove( r, ec );
        std::cout << m_prog << ": deregistered " << n << "\n";
    }
    rc |= systemctl( { "daemon-reload" } );
    return rc;
}

inline int resuCtrl::cmdBoot()
{
    if( m_names.size() != 1 || ( m_names[0] != "on" && m_names[0] != "off" ) )
    {
        std::cerr << m_prog << ": boot needs on or off\n";
        return 1;
    }
    unitNames u( m_role );
    return systemctl( { m_names[0] == "on" ? "enable" : "disable", u.target() }, true );
}

inline int resuCtrl::main( int argc, char **argv )
{
    // also installed as xctrl
    std::string invoked = argc > 0 ? argv[0] : "resuctrl";
    m_prog              = invoked.substr( invoked.find_last_of( '/' ) + 1 );

    if( parseArgs( argc, argv ) < 0 )
    {
        return 2;
    }
    if( m_verb == "help" )
    {
        usage( std::cout );
        return 0;
    }

    static const std::map<std::string, int ( resuCtrl::* )()> verbs = { { "reset", &resuCtrl::cmdReset },
                                                                        { "startup", &resuCtrl::cmdStartup },
                                                                        { "shutdown", &resuCtrl::cmdShutdown },
                                                                        { "start", &resuCtrl::cmdStart },
                                                                        { "stop", &resuCtrl::cmdStop },
                                                                        { "restart", &resuCtrl::cmdRestart },
                                                                        { "defib", &resuCtrl::cmdRestart },
                                                                        { "status", &resuCtrl::cmdStatus },
                                                                        { "lstatus", &resuCtrl::cmdStatus },
                                                                        { "peek", &resuCtrl::cmdPeek },
                                                                        { "inspect", &resuCtrl::cmdInspect },
                                                                        { "enable", &resuCtrl::cmdEnable },
                                                                        { "disable", &resuCtrl::cmdEnable },
                                                                        { "delete", &resuCtrl::cmdDelete },
                                                                        { "deregister", &resuCtrl::cmdDelete },
                                                                        { "boot", &resuCtrl::cmdBoot } };
    auto                                                      it    = verbs.find( m_verb );
    if( it == verbs.end() )
    {
        std::cerr << m_prog << ": unknown command " << m_verb << "; try --help\n";
        return 2;
    }

    // decided before settlePrivileges() clears the environment
    m_color = isatty( STDOUT_FILENO ) && getenv( "NO_COLOR" ) == nullptr;

    if( settlePrivileges() < 0 )
    {
        return 1;
    }

    return ( this->*( it->second ) )();
}

} // namespace resuctrl

#endif // resuctrl_hpp
