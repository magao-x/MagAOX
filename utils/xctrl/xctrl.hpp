/** \file xctrl.hpp
 * \brief Manage the MagAO-X apps of a role as systemd services
 *
 * \ingroup xctrl_files
 */

#ifndef xctrl_hpp
#define xctrl_hpp

#include <algorithm>
#include <cmath>
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

/** \defgroup xctrl xctrl: MagAO-X systemd process control
 * \brief Run the apps listed in a role's proclist as systemd services.
 *
 * Each app runs as an instance of one service template per role, `magaox-<role>-app@<name>.service`, with a
 * drop-in generated from the proclist and the app's configuration.  The apps enabled for automatic startup are
 * wanted by `magaox-<role>.target`.
 *
 * \ingroup utils
 */

/** \defgroup xctrl_files xctrl Files
 * \ingroup xctrl
 */

namespace xctrl
{

/// One process from a proclist line
/**
 * \ingroup xctrl
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
 * \ingroup xctrl
 */
struct proclist
{
    /// The processes, in proclist order
    std::vector<appEntry> apps;

    /// Commented-out lines that look like processes, `#NAME EXECUTABLE`: registered but not started automatically
    std::vector<appEntry> disabled;

    /// Problems found, one message each; the lines concerned are not in apps
    std::vector<std::string> errors;

    /// Disabled lines that are not `#NAME EXECUTABLE`, one message each; they are ignored
    std::vector<std::string> warnings;
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
 * \ingroup xctrl
 */
struct settingDef
{
    const char *key;         ///< key in the `[systemd]` section of an app or magaox.conf
    const char *directive;   ///< the systemd directive it sets
    bool        unitSection; ///< true if the directive belongs in [Unit], false for [Service]
    valueKind   kind;        ///< how its value is validated
};

/// The `[systemd]` configuration keys xctrl understands
/**
 * \ingroup xctrl
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
 * \ingroup xctrl
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

/// Parse a proclist
/** Each line `NAME EXECUTABLE` is a process started automatically.  A line starting with one `#`, with or without
 * spaces after it, `#NAME EXECUTABLE`, is a disabled process; one that is not `NAME EXECUTABLE` is reported in
 * warnings and ignored.  A line starting with `##` is only a comment, whatever follows.  NAME may have a prefix:
 * `-` or `nhb:` for processes that do not notify systemd, so have no watchdog, `py:` for Python apps; any other
 * `prefix:` is part of the name, which is then invalid.
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

/// Actions on an app's proclist lines
enum class proclistEdit
{
    comment,   ///< comment out its active lines
    uncomment, ///< make its first commented-out line active, or append newLine if it has no line
    remove     ///< remove all its lines, active or commented out
};

/// Edit an app's lines in proclist text, leaving every other line, and every `##` comment, as it is
/**
 * \returns the edited text
 */
std::string editProclist( const std::string &text,    /**< [in] the proclist contents */
                          const std::string &name,    /**< [in] the app name */
                          proclistEdit       edit,    /**< [in] what to do */
                          const std::string &newLine, /**< [in] the line to append for uncomment, if none */
                          int               &changed /**< [out] the number of lines changed */ );

/// Convert a systemd time span, as `systemctl show` prints it (e.g. "1min 30s", "500ms", "infinity"), to seconds
/**
 * \returns the seconds, a large number for infinity, or -1 if it cannot be parsed
 */
double timespanSeconds( const std::string &span /**< [in] the time span */ );

/// Collect systemd directives from the `[systemd]` section of a configuration
/** Invalid values and unknown keys are reported in errors and skipped.
 */
void systemdSettings( directiveMap             &dirs,   /**< [in.out] directives, updated with the settings found */
                      std::vector<std::string> &errors, /**< [in.out] problems found, appended to */
                      const confMap            &conf,   /**< [in] the configuration */
                      const std::string        &source /**< [in] the configuration's name, for messages */ );

/// Collect the systemd directives of one app from the `[systemd]` section of its configuration
/** Apps that do not notify get Type=simple and no watchdog.  A `[resurrectee]` section is an error: its timeouts are
 * now `[systemd] watchdogSec` and `timeoutStartSec`.
 *
 * \returns the directives that differ from the template
 */
directiveMap appDirectives( const appEntry           &app,  /**< [in] the app */
                            const confMap            &conf, /**< [in] the app's configuration */
                            std::vector<std::string> &errors /**< [in.out] problems found, appended to */ );

/// Names of the units of a role
/**
 * \ingroup xctrl
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
/** When xctrl runs setuid root, the command runs either as the calling user, or fully as root.
 *
 * \returns the exit status, or -1 if it could not be run
 */
int runCommand( const std::vector<std::string> &argv, /**< [in] the command and its arguments */
                std::string *output   = nullptr,      /**< [out] [optional] its standard output, else inherited */
                bool         asCaller = false /**< [in] [optional] run as the calling user, not as root */ );

/// How an app's state is reported
/**
 * \ingroup xctrl
 */
struct appState
{
    int         rank;  ///< sort order: 0 running, 1 manually started, 2 disabled, 3 not started, 4 changing, 5 dead
    std::string label; ///< the state in a few words
    const char *color; ///< ANSI color sequence for the state
};

/// Classify an app's service state, with a color for each state
/** Running is green, running with autostart off (started by hand) blue, disabled yellow, not started cyan,
 * starting or stopping magenta, failed or dead red.
 *
 * \returns the state
 */
appState classify( const std::map<std::string, std::string> &props, /**< [in] systemctl show properties of the app */
                   bool                                      registered, /**< [in] whether the app is registered */
                   bool                                      enabled /**< [in] whether it is enabled for autostart */ );

/// Command-line interface to manage a role's apps as systemd services
/**
 * \ingroup xctrl
 */
class xCtrl
{
  public:
    /// Run xctrl with its command-line arguments
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
    std::string m_prog{ "xctrl" };      ///< the name we were run as
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
     * operatorAllowed(); xctrl is installed setuid root for that.  Everything else, and any testing option,
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

    /// The role-wide directives: the template defaults with magaox.conf `[systemd]` settings
    directiveMap roleDirectives( std::vector<std::string> &errors /**< [in.out] problems found, appended to */ );

    /// Check that an app can be registered, and work out its systemd directives from its .conf
    /**
     * \returns true if it can, false with the reasons appended to errors
     */
    bool prepareApp( const appEntry           &app,    /**< [in] the app */
                     directiveMap             &dirs,   /**< [out] its directives */
                     std::vector<std::string> &errors, /**< [in.out] problems found, appended to */
                     bool                     *hasConf = nullptr /**< [out] [optional] whether it has a .conf */ );

    /// Find the role's INDI server and ssh tunnels among apps, for the drop-ins' ordering
    void serverAndTunnels( const std::vector<appEntry> &apps,   /**< [in] the apps */
                           std::string                 &server, /**< [out] the INDI server, "" if none */
                           std::vector<std::string>    &tunnels /**< [out] the ssh tunnel apps */ );

    /// Edit an app's lines in the role's proclist, in place so its owner and permissions stay
    /**
     * \returns the number of lines changed, or -1 with a message on stderr
     */
    int updateProclist( const std::string &name, /**< [in] the app name */
                        proclistEdit       edit, /**< [in] what to do */
                        const std::string &newLine = "" /**< [in] the line to append for uncomment, if none */ );

    /// Find an app in the registry, else in the proclist, active or commented out
    /**
     * \returns true if found
     */
    bool findApp( appEntry          &app, /**< [out] the app */
                  const std::string &name /**< [in] its name */ );

    /// Register one app from its .conf, without reloading systemd
    /**
     * \returns 0 if registered, -1 with messages on stderr if not
     */
    int registerApp( const appEntry &app /**< [in] the app */ );

    /// Stop, disable and deregister apps, without reloading systemd
    /**
     * \returns 0 if all went well, else non-zero
     */
    int deregister( const std::vector<std::string> &names /**< [in] the app names */ );

    /// Bring registered apps' drop-ins up to date with their .conf before they start
    /** Regenerates each app's drop-in, and reloads systemd if one changed, if systemd still has older unit
     * files loaded, or if the watchdog systemd has for an app differs from its configuration.
     *
     * \returns 0 if all went well, else non-zero
     */
    int refreshApps( const std::vector<appEntry> &apps /**< [in] the apps about to start */ );

    /** \name Commands
     * @{
     */

    int cmdReset();    ///< Register the proclist's apps with systemd and reconcile what changed
    int cmdStartup();  ///< Start enabled apps, or enable and start named ones
    int cmdShutdown(); ///< Stop all apps, or named ones
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

        if( toks.empty() || ( toks.size() > 1 && toks[1][0] == '#' ) )
        {
            continue; // blank line or comment
        }

        // "##..." is only a comment, whatever follows, e.g. "##camsci_old camsci"
        if( toks[0].compare( 0, 2, "##" ) == 0 )
        {
            continue;
        }

        // a commented-out process, e.g. "#camsci1 picamCtrl" or "# camsci1 picamCtrl"
        bool commented = toks[0][0] == '#';
        if( commented )
        {
            std::string rest = line.substr( line.find( '#' ) );
            rest.erase( 0, rest.find_first_not_of( "# \t" ) );
            std::istringstream cs( rest );
            toks.clear();
            while( cs >> tok )
            {
                toks.push_back( tok );
            }
            if( toks.size() != 2 )
            {
                pl.warnings.push_back( "proclist line " + std::to_string( n ) + ": [" + trim( line ) +
                                       "] is not #NAME EXECUTABLE, ignored; use ## for comments" );
                continue;
            }
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

        if( commented )
        {
            if( validName( app.name ) && validName( app.exec ) )
            {
                pl.disabled.push_back( app );
            }
            else
            {
                pl.warnings.push_back( "proclist line " + std::to_string( n ) + ": [" + trim( line ) +
                                       "] is not a valid #NAME EXECUTABLE, ignored; use ## for comments" );
            }
            continue;
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

    // an active line wins over a commented-out one for the same name, and the first commented-out one wins
    std::vector<appEntry> disabled;
    for( const auto &d : pl.disabled )
    {
        if( !seen.count( d.name ) )
        {
            seen.insert( d.name );
            disabled.push_back( d );
        }
    }
    pl.disabled = disabled;

    return pl;
}

inline std::string editProclist(
    const std::string &text, const std::string &name, proclistEdit edit, const std::string &newLine, int &changed )
{
    // whether line is NAME EXECUTABLE for this name, with any prefix, and whether it is commented out
    auto match = [&]( const std::string &line, bool &commented ) -> bool
    {
        std::string t = trim( line );
        commented     = !t.empty() && t[0] == '#';
        if( t.compare( 0, 2, "##" ) == 0 )
        {
            return false; // only a comment, never an app's line
        }
        t.erase( 0, t.find_first_not_of( "# \t" ) );
        std::istringstream       ss( t );
        std::vector<std::string> toks;
        std::string              tok;
        while( ss >> tok )
        {
            toks.push_back( tok );
        }
        if( toks.size() != 2 )
        {
            return false;
        }
        std::string n = toks[0];
        if( n[0] == '-' )
        {
            n.erase( 0, 1 );
        }
        else if( n.compare( 0, 3, "py:" ) == 0 || n.compare( 0, 4, "nhb:" ) == 0 )
        {
            n.erase( 0, n.find( ':' ) + 1 );
        }
        return n == name;
    };

    std::vector<std::string> lines;
    std::istringstream       in( text );
    std::string              line;
    bool                     hasActive = false;
    while( std::getline( in, line ) )
    {
        bool c;
        hasActive = hasActive || ( match( line, c ) && !c );
        lines.push_back( line );
    }

    changed = 0;
    std::ostringstream out;
    for( auto &l : lines )
    {
        bool c;
        if( !match( l, c ) )
        {
            out << l << "\n";
            continue;
        }
        if( edit == proclistEdit::remove )
        {
            ++changed;
            continue;
        }
        if( edit == proclistEdit::comment && !c )
        {
            l = "#" + l;
            ++changed;
        }
        else if( edit == proclistEdit::uncomment && c && !hasActive )
        {
            l.erase( 0, l.find_first_not_of( "# \t" ) );
            hasActive = true;
            ++changed;
        }
        out << l << "\n";
    }
    if( edit == proclistEdit::uncomment && !hasActive && !newLine.empty() )
    {
        out << newLine << "\n";
        ++changed;
    }
    return out.str();
}

inline double timespanSeconds( const std::string &span )
{
    static const std::map<std::string, double> units = { { "us", 1e-6 },
                                                         { "usec", 1e-6 },
                                                         { "ms", 1e-3 },
                                                         { "msec", 1e-3 },
                                                         { "s", 1 },
                                                         { "sec", 1 },
                                                         { "", 1 },
                                                         { "min", 60 },
                                                         { "m", 60 },
                                                         { "h", 3600 },
                                                         { "hr", 3600 },
                                                         { "d", 86400 },
                                                         { "w", 604800 },
                                                         { "M", 2629800 },
                                                         { "y", 31557600 } };
    std::string                                t     = trim( span );
    if( t == "infinity" )
    {
        return 1e18;
    }
    if( t.empty() )
    {
        return -1;
    }

    double             total = 0;
    std::istringstream ss( t );
    std::string        tok;
    while( ss >> tok )
    {
        size_t i = 0;
        while( i < tok.size() && ( isdigit( static_cast<unsigned char>( tok[i] ) ) || tok[i] == '.' ) )
        {
            ++i;
        }
        auto u = units.find( tok.substr( i ) );
        if( i == 0 || u == units.end() )
        {
            return -1;
        }
        total += strtod( tok.substr( 0, i ).c_str(), nullptr ) * u->second;
    }
    return total;
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

    // the old [resurrectee] timeouts are [systemd] settings now
    for( const auto &kv : conf )
    {
        if( kv.first.compare( 0, 12, "resurrectee." ) == 0 )
        {
            errors.push_back( source + ": [resurrectee] " + kv.first.substr( 12 ) +
                              " is replaced by [systemd] watchdogSec and timeoutStartSec" );
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
    os << "# Generated by xctrl reset for MAGAOX_ROLE=" << units.role << ".  Do not edit: set [systemd] in\n"
       << "# magaox.conf, or in an app's .conf for that app, then rerun xctrl reset.\n"
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
    os << "# Generated by xctrl reset for MAGAOX_ROLE=" << units.role << ".  Do not edit.\n"
       << "# Wants the apps enabled for automatic startup; xctrl boot on|off starts it at boot or not.\n"
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
    os << "# Generated by xctrl reset from proclist_" << units.role << ".txt and " << app.name << ".conf.\n"
       << "# Do not edit: change those, then rerun xctrl reset.\n"
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
    return "# xctrl registry entry, written by xctrl reset\nname=" + app.name + "\nexec=" + app.exec +
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
        std::cerr << "xctrl: creating directory for " << path << ": " << ec.message() << "\n";
        return -1;
    }

    std::string tmp = path + ".xctrl.tmp";
    {
        std::ofstream f( tmp, std::ios::trunc );
        f << content;
        if( !f.good() )
        {
            std::cerr << "xctrl: writing " << tmp << ": " << strerror( errno ) << "\n";
            return -1;
        }
    }
    if( chmod( tmp.c_str(), mode ) < 0 || rename( tmp.c_str(), path.c_str() ) < 0 )
    {
        std::cerr << "xctrl: installing " << path << ": " << strerror( errno ) << "\n";
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
        std::cerr << "xctrl: running " << argv[0] << ": " << strerror( errno ) << "\n";
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
    static const char *green = "\033[32m", *blue = "\033[34m", *yellow = "\033[33m", *cyan = "\033[36m",
                      *magenta = "\033[35m", *red = "\033[31m";

    auto get = [&]( const char *k ) -> std::string
    {
        auto it = props.find( k );
        return it == props.end() ? "" : it->second;
    };
    std::string active = get( "ActiveState" ), sub = get( "SubState" ), result = get( "Result" );

    if( active.empty() )
    {
        return { 3, "unknown", cyan };
    }
    if( active == "active" && sub == "running" )
    {
        if( !registered )
        {
            return { 4, "running, not registered", magenta };
        }
        return enabled ? appState{ 0, "running", green } : appState{ 1, "manually started", blue };
    }
    if( active == "activating" )
    {
        return { 4, "starting", magenta };
    }
    if( active == "deactivating" )
    {
        return { 4, "stopping", magenta };
    }
    if( active == "failed" || ( !result.empty() && result != "success" ) )
    {
        return { 5, "dead (" + result + ")", red };
    }
    if( active == "inactive" )
    {
        return enabled ? appState{ 3, "not started", cyan } : appState{ 2, "disabled", yellow };
    }
    return { 4, active + "/" + sub, magenta };
}

inline std::string xCtrl::unitDir() const
{
    return m_sysroot + "/etc/systemd/system";
}

inline std::string xCtrl::regDir() const
{
    return m_sysroot + "/etc/magaox-" + lower( m_role ) + "/apps";
}

inline std::string xCtrl::envFile() const
{
    return "/etc/magaox-" + lower( m_role ) + "/environment";
}

inline std::string xCtrl::proclistPath() const
{
    return m_top + "/" + MAGAOX_configRelPath + "/proclist_" + m_role + ".txt";
}

inline std::string xCtrl::activePath() const
{
    return m_top + "/" + MAGAOX_sysRelPath + "/sentinel_proclist/proclist_" + m_role + ".txt.active";
}

inline void xCtrl::usage( std::ostream &os ) const
{
    os << "usage: " << m_prog << " [-r ROLE] [options] COMMAND [NAME ...]\n"
       << "\n"
          "MagAO-X process control: run the apps of a role (MAGAOX_ROLE, or -r) as systemd services, one\n"
          "instance of magaox-<role>-app@.service per app in /opt/MagAOX/config/proclist_<role>.txt.\n"
          "\n"
          "Any user can look; xsup and members of group magaox can start and stop apps:\n"
          "  status [NAME...]       state and pid of the apps; -v adds autostart, restarts, uptime,\n"
          "                         last watchdog ping, CPU, memory and GPU memory\n"
          "  peek [NAME...]         state and last log lines of the apps (-n LINES, default 10)\n"
          "  inspect NAME           follow an app's console output (-n LINES, default 50)\n"
          "  startup [NAME...]      start the apps enabled for automatic startup, the INDI server last;\n"
          "                         with names, start those apps\n"
          "  start NAME...          start apps, enabled or not; systemd supervises them either way.\n"
          "                         start, startup and restart first apply changes in the apps' .conf,\n"
          "                         e.g. a new [systemd] watchdogSec\n"
          "  stop NAME...           stop apps; they stay stopped until started again\n"
          "  shutdown NAME...|--all stop apps, or all running apps of the role, the INDI server last\n"
          "  restart NAME...|--all  restart apps, or stop then start everything\n"
          "\n"
          "Changing which apps are configured needs sudo:\n"
          "  sudo xctrl reset    register the proclist's apps from their .conf and enable them for\n"
          "                         automatic startup; commented-out apps with a .conf are registered\n"
          "                         but disabled (and stopped), to start by hand; apps no longer in the\n"
          "                         proclist are stopped and deregistered; new apps start at once if the\n"
          "                         role is up\n"
          "  sudo xctrl enable NAME...   autostart on: uncomment the app in the proclist (adding it if\n"
          "                                 needed) and register it if it is not; does not start it\n"
          "  sudo xctrl disable NAME...  autostart off: comment the app out in the proclist; it stays\n"
          "                                 registered to start by hand; does not stop it\n"
          "  sudo xctrl delete NAME...   remove the app from the proclist, stop and deregister it\n"
          "  sudo xctrl boot on|off      start the role's enabled apps at boot, or not\n"
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

inline bool xCtrl::operatorAllowed() const
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

inline int xCtrl::settlePrivileges()
{
    static const std::set<std::string> admin     = { "reset", "enable", "disable", "delete", "deregister", "boot" };
    static const std::set<std::string> operate   = { "startup", "start", "shutdown", "stop", "restart" };
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

inline int xCtrl::parseArgs( int argc, char **argv )
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

inline int xCtrl::systemctl( const std::vector<std::string> &args, bool offline )
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

inline std::vector<std::string> xCtrl::registered() const
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

inline bool xCtrl::registryRead( appEntry &app, const std::string &name ) const
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

inline bool xCtrl::enabled( const std::string &name ) const
{
    unitNames       u( m_role );
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::symlink_status( unitDir() + "/" + u.target() + ".wants/" + u.instance( name ), ec ) );
}

inline std::vector<appEntry> xCtrl::namedApps()
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
                      << " and run sudo xctrl reset\n";
            ok = false;
        }
    }
    return ok ? apps : std::vector<appEntry>();
}

inline std::vector<std::string> xCtrl::activeApps()
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

inline directiveMap xCtrl::roleDirectives( std::vector<std::string> &errors )
{
    directiveMap dirs = defaultDirectives();
    confMap      gconf;
    readConf( gconf, m_top + "/" + MAGAOX_configRelPath + "/" + MAGAOX_globalConfig );
    systemdSettings( dirs, errors, gconf, MAGAOX_globalConfig );
    return dirs;
}

inline bool
xCtrl::prepareApp( const appEntry &app, directiveMap &dirs, std::vector<std::string> &errors, bool *hasConf )
{
    std::string exe = m_top + "/bin/" + app.exec;
    struct stat st;
    if( stat( exe.c_str(), &st ) < 0 || !S_ISREG( st.st_mode ) || access( exe.c_str(), X_OK ) < 0 )
    {
        errors.push_back( app.name + ": executable " + exe + " is missing or not executable" );
        return false;
    }

    confMap c;
    bool    found = readConf( c, m_top + "/" + MAGAOX_configRelPath + "/" + app.name + ".conf" );
    if( hasConf )
    {
        *hasConf = found;
    }

    size_t n = errors.size();
    dirs     = appDirectives( app, c, errors );
    return errors.size() == n;
}

inline void
xCtrl::serverAndTunnels( const std::vector<appEntry> &apps, std::string &server, std::vector<std::string> &tunnels )
{
    server.clear();
    tunnels.clear();
    for( const auto &a : apps )
    {
        if( a.isServer() && server.empty() )
        {
            server = a.name;
        }
        if( a.isTunnel() )
        {
            tunnels.push_back( a.name );
        }
    }
}

inline int xCtrl::updateProclist( const std::string &name, proclistEdit edit, const std::string &newLine )
{
    std::string text;
    {
        std::ifstream f( proclistPath() );
        if( !f )
        {
            std::cerr << m_prog << ": cannot read " << proclistPath() << "\n";
            return -1;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }

    int         changed = 0;
    std::string edited  = editProclist( text, name, edit, newLine, changed );
    if( changed == 0 )
    {
        return 0;
    }
    if( m_dryRun )
    {
        std::cout << "[dry-run] would edit " << changed << " line(s) for " << name << " in " << proclistPath() << "\n";
        return changed;
    }

    // in place, not by rename, so the file keeps its owner and permissions
    std::ofstream f( proclistPath(), std::ios::trunc );
    f << edited;
    if( !f.good() )
    {
        std::cerr << m_prog << ": writing " << proclistPath() << ": " << strerror( errno ) << "\n";
        return -1;
    }
    return changed;
}

inline bool xCtrl::findApp( appEntry &app, const std::string &name )
{
    if( registryRead( app, name ) )
    {
        return true;
    }
    std::ifstream pf( proclistPath() );
    proclist      pl = parseProclist( pf );
    for( const auto *list : { &pl.apps, &pl.disabled } )
    {
        for( const auto &a : *list )
        {
            if( a.name == name )
            {
                app = a;
                return true;
            }
        }
    }
    return false;
}

inline int xCtrl::registerApp( const appEntry &app )
{
    unitNames                u( m_role );
    directiveMap             dirs;
    std::vector<std::string> errors;
    if( !prepareApp( app, dirs, errors ) )
    {
        for( const auto &e : errors )
        {
            std::cerr << m_prog << ": error: " << e << "\n";
        }
        return -1;
    }

    std::vector<appEntry> all{ app };
    for( const auto &n : registered() )
    {
        appEntry a;
        if( n != app.name && registryRead( a, n ) )
        {
            all.push_back( a );
        }
    }
    std::string              server;
    std::vector<std::string> tunnels;
    serverAndTunnels( all, server, tunnels );

    int rc = writeFile( unitDir() + "/" + u.instance( app.name ) + ".d/override.conf",
                        dropIn( u, app, m_top + "/bin", dirs, server, tunnels ),
                        0644 ) < 0;
    rc |= writeFile( regDir() + "/" + app.name + ".conf", registryEntry( app ), 0644 ) < 0;
    return rc ? -1 : 0;
}

inline int xCtrl::deregister( const std::vector<std::string> &names )
{
    if( names.empty() )
    {
        return 0;
    }
    unitNames                u( m_role );
    std::vector<std::string> dis{ "disable" };
    for( const auto &n : names )
    {
        dis.push_back( u.instance( n ) );
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
    return rc;
}

inline int xCtrl::refreshApps( const std::vector<appEntry> &apps )
{
    if( apps.empty() )
    {
        return 0;
    }
    unitNames u( m_role );

    // the ordering in the drop-ins depends on all registered apps
    std::vector<appEntry> all;
    for( const auto &n : registered() )
    {
        appEntry a;
        if( registryRead( a, n ) )
        {
            all.push_back( a );
        }
    }
    std::string              server;
    std::vector<std::string> tunnels;
    serverAndTunnels( all, server, tunnels );

    std::vector<std::string> errors, names;
    directiveMap             role = roleDirectives( errors );
    for( const auto &a : apps )
    {
        names.push_back( a.name );
    }
    auto props = showProps( names );

    bool reload = false;
    int  rc     = 0;
    for( const auto &a : apps )
    {
        directiveMap             dirs;
        std::vector<std::string> aerr;
        if( !prepareApp( a, dirs, aerr ) )
        {
            for( const auto &e : aerr )
            {
                std::cerr << m_prog << ": warning: " << e << "\n";
            }
            std::cerr << m_prog << ": warning: " << a.name << " starts with its last good settings\n";
            continue;
        }

        int w = writeFile( unitDir() + "/" + u.instance( a.name ) + ".d/override.conf",
                           dropIn( u, a, m_top + "/bin", dirs, server, tunnels ),
                           0644 );
        rc |= w < 0;

        // the watchdog systemd has now, against the one configured
        auto  &p        = props[a.name];
        double want     = timespanSeconds( dirs.count( "WatchdogSec" ) ? dirs["WatchdogSec"] : role["WatchdogSec"] );
        double have     = p.count( "WatchdogUSec" ) ? timespanSeconds( p["WatchdogUSec"] ) : -1;
        bool   stale    = p.count( "NeedDaemonReload" ) && p["NeedDaemonReload"] == "yes";
        bool   wdChange = have >= 0 && want >= 0 && std::abs( have - want ) > 1e-3;
        if( wdChange )
        {
            std::cout << m_prog << ": " << a.name << ": watchdog " << p["WatchdogUSec"] << " -> " << want
                      << "s, from its configuration\n";
        }
        else if( w == 1 )
        {
            std::cout << m_prog << ": " << a.name << ": settings updated from its configuration\n";
        }
        reload = reload || w == 1 || stale || wdChange;
    }

    if( reload )
    {
        rc |= systemctl( { "daemon-reload" } ) != 0;
    }
    return rc;
}

inline int xCtrl::startApps( const std::vector<appEntry> &apps )
{
    unitNames                u( m_role );
    std::vector<std::string> others{ "start" }, server{ "start" }, reset{ "reset-failed" };
    for( const auto &a : apps )
    {
        ( a.isServer() ? server : others ).push_back( u.instance( a.name ) );
        reset.push_back( u.instance( a.name ) );
    }

    // with its current configuration
    refreshApps( apps );

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

inline int xCtrl::stopApps( const std::vector<std::string> &names )
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

inline int xCtrl::writeFile( const std::string &path, const std::string &content, mode_t mode )
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

inline int xCtrl::cmdReset()
{
    unitNames                u( m_role );
    std::vector<std::string> errors, warnings;

    std::ifstream pf( proclistPath() );
    if( !pf )
    {
        std::cerr << m_prog << ": cannot read proclist " << proclistPath() << "\n";
        return 1;
    }
    proclist pl = parseProclist( pf );
    errors      = pl.errors;
    warnings    = pl.warnings;

    // role-wide defaults, from magaox.conf [systemd]
    std::string  cfgDir = m_top + "/" + MAGAOX_configRelPath;
    directiveMap gdirs  = roleDirectives( errors );

    // the INDI server and the apps must agree on the control FIFO
    confMap gconf, isconf;
    readConf( gconf, cfgDir + "/" + MAGAOX_globalConfig );
    if( readConf( isconf, cfgDir + "/is" + m_role + ".conf" ) && isconf.count( "indiserver.f" ) &&
        gconf.count( "indiserver_ctrl_fifo" ) && isconf["indiserver.f"] != gconf["indiserver_ctrl_fifo"] )
    {
        std::cerr << m_prog << ": warning: [indiserver] f = " << isconf["indiserver.f"] << " in is" << m_role
                  << ".conf differs from indiserver_ctrl_fifo = " << gconf["indiserver_ctrl_fifo"]
                  << " in magaox.conf\n";
    }

    struct valid
    {
        appEntry     app;
        directiveMap dirs;
        bool         autostart;
    };
    std::vector<valid> apps;

    // names in the proclist that keep any registration they have, though they could not be (re)registered now
    std::set<std::string> keep;

    // active lines: registered and started automatically; invalid ones keep their earlier registration and state
    for( const auto &app : pl.apps )
    {
        directiveMap d;
        bool         hasConf = false;
        size_t       n       = errors.size();
        if( !prepareApp( app, d, errors, &hasConf ) )
        {
            errors.insert( errors.begin() + n,
                           "proclist line " + std::to_string( app.line ) + ": " + app.name + " not (re)registered:" );
            keep.insert( app.name );
            continue;
        }
        if( !hasConf )
        {
            std::cout << m_prog << ": note: " << app.name << " has no " << app.name << ".conf, using defaults\n";
        }
        apps.push_back( { app, d, true } );
    }

    // disabled lines: registered, to start by hand, but not started automatically; they need a .conf
    std::set<std::string> commented;
    for( const auto &app : pl.disabled )
    {
        if( access( ( cfgDir + "/" + app.name + ".conf" ).c_str(), R_OK ) != 0 )
        {
            warnings.push_back( "proclist line " + std::to_string( app.line ) + ": disabled " + app.name + " has no " +
                                app.name + ".conf, not registered; use ## for comments" );
            continue;
        }
        commented.insert( app.name );
        directiveMap             d;
        std::vector<std::string> aerr;
        if( !prepareApp( app, d, aerr ) )
        {
            warnings.push_back( "proclist line " + std::to_string( app.line ) + ": commented-out " + app.name +
                                " not (re)registered:" );
            warnings.insert( warnings.end(), aerr.begin(), aerr.end() );
            keep.insert( app.name );
            continue;
        }
        apps.push_back( { app, d, false } );
    }

    // one INDI server per role
    std::string server;
    for( auto it = apps.begin(); it != apps.end(); )
    {
        if( it->app.isServer() && !server.empty() )
        {
            errors.push_back( it->app.name + ": a second INDI server, " + server + " is the role's, not registered" );
            keep.insert( it->app.name );
            it = apps.erase( it );
            continue;
        }
        if( it->app.isServer() )
        {
            server = it->app.name;
        }
        ++it;
    }
    std::vector<appEntry> entries;
    for( const auto &v : apps )
    {
        entries.push_back( v.app );
    }
    std::vector<std::string> tunnels;
    serverAndTunnels( entries, server, tunnels );

    // registered apps no longer in the proclist at all are stopped and deregistered
    std::vector<std::string> before = registered(), removed;
    std::set<std::string>    wasRegistered( before.begin(), before.end() );
    std::set<std::string>    listed( keep.begin(), keep.end() );
    for( const auto &v : apps )
    {
        listed.insert( v.app.name );
    }
    for( const auto &n : before )
    {
        if( !listed.count( n ) )
        {
            removed.push_back( n );
        }
    }
    int rc = deregister( removed );

    // the role's template and target, then each app's drop-in and registry entry
    rc |= writeFile( unitDir() + "/" + u.templ(), templateUnit( u, gdirs, envFile() ), 0644 ) < 0;
    rc |= writeFile( unitDir() + "/" + u.target(), targetUnit( u ), 0644 ) < 0;

    std::vector<std::string> changed, toEnable, toDisable, stopNow;
    std::vector<appEntry>    newlyEnabled;
    for( const auto &v : apps )
    {
        int w = writeFile( unitDir() + "/" + u.instance( v.app.name ) + ".d/override.conf",
                           dropIn( u, v.app, m_top + "/bin", v.dirs, server, tunnels ),
                           0644 );
        rc |= w < 0;
        rc |= writeFile( regDir() + "/" + v.app.name + ".conf", registryEntry( v.app ), 0644 ) < 0;
        if( w == 1 && wasRegistered.count( v.app.name ) )
        {
            changed.push_back( v.app.name );
        }
        if( v.autostart && !enabled( v.app.name ) )
        {
            toEnable.push_back( u.instance( v.app.name ) );
            newlyEnabled.push_back( v.app );
        }
        if( !v.autostart && enabled( v.app.name ) )
        {
            toDisable.push_back( u.instance( v.app.name ) );
            stopNow.push_back( v.app.name );
        }
    }
    // commented-out apps that could not be re-registered still leave automatic startup
    for( const auto &n : commented )
    {
        if( keep.count( n ) && wasRegistered.count( n ) && enabled( n ) )
        {
            toDisable.push_back( u.instance( n ) );
            stopNow.push_back( n );
        }
    }
    if( rc )
    {
        std::cerr << m_prog << ": could not write the systemd configuration; run with sudo\n";
        return 1;
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

    // apps just commented out also stop, unless --keep-running
    if( !stopNow.empty() && !m_keepRunning )
    {
        std::vector<std::string> act = activeApps(), stop;
        for( const auto &n : stopNow )
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
    active << "# Applied by xctrl reset from " << proclistPath() << "; do not edit\n";
    size_t nauto = 0;
    for( const auto &v : apps )
    {
        active << ( v.autostart ? "" : "#" ) << v.app.prefix << v.app.name << " " << v.app.exec << "\n";
        nauto += v.autostart;
    }
    if( writeFile( activePath(), active.str(), 0664 ) < 0 )
    {
        rc = 1;
    }

    std::cout << m_prog << ": role " << m_role << ": registered " << nauto << " app(s) for automatic startup ("
              << newlyEnabled.size() << " new), " << apps.size() - nauto << " commented-out app(s) to start by hand";
    if( !removed.empty() )
    {
        std::cout << "; deregistered, no longer in the proclist:";
        for( const auto &n : removed )
        {
            std::cout << " " << n;
        }
    }
    std::cout << "\n";
    for( const auto &n : changed )
    {
        std::cout << m_prog << ": " << n << ": settings changed, applied at its next start; " << m_prog << " restart "
                  << n << " applies them now\n";
    }
    for( const auto &w : warnings )
    {
        std::cerr << m_prog << ": warning: " << w << "\n";
    }
    for( const auto &e : errors )
    {
        std::cerr << m_prog << ": error: " << e << "\n";
    }

    return ( rc || !errors.empty() ) ? 1 : 0;
}

inline int xCtrl::cmdStartup()
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
        std::cerr << m_prog << ": no apps enabled for role " << m_role << "; run sudo xctrl reset\n";
        return 1;
    }
    int rc = startApps( apps );
    return rc | systemctl( { "start", u.target() } );
}

inline int xCtrl::cmdShutdown()
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

inline int xCtrl::cmdStart()
{
    std::vector<appEntry> apps = namedApps();
    return apps.empty() ? 1 : startApps( apps );
}

inline int xCtrl::cmdStop()
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

inline int xCtrl::cmdRestart()
{

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
    refreshApps( apps );
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
xCtrl::showProps( const std::vector<std::string> &names )
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
        "ExecMainStartTimestampMonotonic,CPUUsageNSec,MemoryCurrent,ControlGroup,NeedDaemonReload" };
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

inline int xCtrl::cmdStatus()
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
        std::cout << m_prog << ": no apps registered for role " << m_role << "; run sudo xctrl reset\n";
        return 0;
    }

    auto props = showProps( names );

    // sorted by state then name, colored
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
        // name: state (pid: N)
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

inline int xCtrl::cmdPeek()
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

inline int xCtrl::cmdInspect()
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

inline int xCtrl::cmdEnable()
{
    if( m_names.empty() )
    {
        std::cerr << m_prog << ": " << m_verb << " needs app names\n";
        return 1;
    }
    unitNames                u( m_role );
    bool                     enable = m_verb == "enable", reload = false;
    std::vector<std::string> args{ m_verb };
    int                      rc = 0;

    for( const auto &n : m_names )
    {
        appEntry app;
        if( !findApp( app, n ) )
        {
            std::cerr << m_prog << ": " << n << " is neither registered nor in " << proclistPath() << "\n";
            rc = 1;
            continue;
        }

        // the proclist says what starts automatically, so it changes too
        int e = enable ? updateProclist( n, proclistEdit::uncomment, app.prefix + app.name + " " + app.exec )
                       : updateProclist( n, proclistEdit::comment );
        if( e < 0 )
        {
            rc = 1;
            continue;
        }

        // an app commented out without a .conf, or added since the last reset, is registered now
        appEntry reg;
        if( enable && !registryRead( reg, n ) )
        {
            if( registerApp( app ) < 0 )
            {
                std::cerr << m_prog << ": " << n << " is enabled in the proclist but could not be registered\n";
                rc = 1;
                continue;
            }
            reload = true;
        }
        if( registryRead( reg, n ) )
        {
            args.push_back( u.instance( n ) );
        }
        std::cout << m_prog << ": " << n << ( enable ? ": autostart on" : ": autostart off" )
                  << ( e > 0 ? ( enable ? ", uncommented in " : ", commented out in " ) + proclistPath() : "" ) << "\n";
    }

    if( reload )
    {
        rc |= systemctl( { "daemon-reload" } ) != 0;
    }
    if( args.size() > 1 )
    {
        rc |= systemctl( args, true ) != 0;
    }

    std::cout << m_prog << ": note: " << m_verb << " does not "
              << ( enable ? "start apps: " + m_prog + " start NAME, or the next startup, does"
                          : "stop apps: " + m_prog + " stop NAME does" )
              << "\n";
    return rc;
}

inline int xCtrl::cmdDelete()
{
    if( m_names.empty() )
    {
        std::cerr << m_prog << ": " << m_verb << " needs app names\n";
        return 1;
    }
    int                      rc = 0;
    std::vector<std::string> regs;
    for( const auto &n : m_names )
    {
        appEntry app;
        int      e = updateProclist( n, proclistEdit::remove );
        if( e < 0 )
        {
            rc = 1;
            continue;
        }
        bool isReg = registryRead( app, n );
        if( isReg )
        {
            regs.push_back( n );
        }
        if( e == 0 && !isReg )
        {
            std::cerr << m_prog << ": " << n << " is neither registered nor in " << proclistPath() << "\n";
            rc = 1;
            continue;
        }
        if( e > 0 )
        {
            std::cout << m_prog << ": removed " << n << " from " << proclistPath() << "\n";
        }
    }
    if( !regs.empty() )
    {
        rc |= deregister( regs );
        rc |= systemctl( { "daemon-reload" } );
    }
    return rc;
}

inline int xCtrl::cmdBoot()
{
    if( m_names.size() != 1 || ( m_names[0] != "on" && m_names[0] != "off" ) )
    {
        std::cerr << m_prog << ": boot needs on or off\n";
        return 1;
    }
    unitNames u( m_role );
    return systemctl( { m_names[0] == "on" ? "enable" : "disable", u.target() }, true );
}

inline int xCtrl::main( int argc, char **argv )
{
    // also installed as xctrl
    std::string invoked = argc > 0 ? argv[0] : "xctrl";
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

    static const std::map<std::string, int ( xCtrl::* )()> verbs = { { "reset", &xCtrl::cmdReset },
                                                                     { "startup", &xCtrl::cmdStartup },
                                                                     { "shutdown", &xCtrl::cmdShutdown },
                                                                     { "start", &xCtrl::cmdStart },
                                                                     { "stop", &xCtrl::cmdStop },
                                                                     { "restart", &xCtrl::cmdRestart },
                                                                     { "status", &xCtrl::cmdStatus },
                                                                     { "lstatus", &xCtrl::cmdStatus },
                                                                     { "peek", &xCtrl::cmdPeek },
                                                                     { "inspect", &xCtrl::cmdInspect },
                                                                     { "enable", &xCtrl::cmdEnable },
                                                                     { "disable", &xCtrl::cmdEnable },
                                                                     { "delete", &xCtrl::cmdDelete },
                                                                     { "deregister", &xCtrl::cmdDelete },
                                                                     { "boot", &xCtrl::cmdBoot } };
    auto                                                   it    = verbs.find( m_verb );
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

} // namespace xctrl

#endif // xctrl_hpp
