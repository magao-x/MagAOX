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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

#include "xctrlLog.hpp"

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

    /// The proclist prefix of the name: "" for apps that notify systemd, "-" for those that do not
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

    /// Disabled processes, `#!NAME EXECUTABLE`: registered but not started automatically
    std::vector<appEntry> disabled;

    /// Problems found, one message each; the lines concerned are not in apps
    std::vector<std::string> errors;

    /// `#!` lines that are not `#!NAME EXECUTABLE`, one message each; they are ignored
    std::vector<std::string> warnings;
};

/// Key-value settings from a configuration file, keyed by "section.key", or "key" before any section
typedef std::map<std::string, std::string> confMap;

/// systemd directives and their values, keyed by directive name
typedef std::map<std::string, std::string> directiveMap;

/// The section of each systemd directive a service can set: true for [Unit], false for [Service]
typedef std::map<std::string, bool> directiveSections;

/// Read the directives of [Unit] and [Service] from `systemd --dump-configuration-items` output
/** The output lists `[Section]` headers, each followed by `Directive=VALUE-TYPE` lines.  A directive listed in
 * both, e.g. StartLimitBurst, which [Service] still accepts for compatibility, is taken as [Unit], listed first.
 *
 * \returns the directives and their sections
 */
directiveSections parseDirectiveDump( const std::string &dump /**< [in] the output */ );

/// The directives this host's systemd accepts in [Unit] and [Service], read once from systemd itself
/**
 * \returns the directives and their sections, empty if systemd could not be asked
 */
const directiveSections &systemdDirectives();

/// Directives a `[systemd]` section may not set: xctrl sets them, or they could give an app more than xsup's
/// privileges or act on the whole host, while any member of magaox-dev may edit the configuration
/**
 * \returns true if the directive may not be set
 */
bool deniedDirective( const std::string &directive /**< [in] the directive name */ );

/// Return s in lower case
std::string lower( const std::string &s /**< [in] the string */ );

/// Return s without leading and trailing white space
std::string trim( const std::string &s /**< [in] the string */ );

/// Whether s is a valid app or executable name: letters, digits, `_`, `.` and `-`, starting with a letter or digit
bool validName( const std::string &s /**< [in] the name */ );

/// Parse a proclist
/** Each line `NAME EXECUTABLE` is a process started automatically.  A line starting with `#!`, with or without
 * spaces after it, `#!NAME EXECUTABLE`, is a disabled process: registered, but not started automatically; one that is
 * not `NAME EXECUTABLE` is reported in warnings and ignored.  Any other line starting with `#` is a comment.  NAME may
 * have a prefix: `-` for processes that do not notify systemd, so have no watchdog.
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
    comment,   ///< disable its active lines, with #!
    uncomment, ///< make its first disabled line active, or append newLine if it has no line
    remove,    ///< remove all its lines, active or disabled
    append     ///< append newLine if the app has no line, active or disabled
};

/// Edit an app's lines in proclist text, leaving every other line, and every `#` comment, as it is
/**
 * \returns the edited text
 */
std::string editProclist( const std::string &text,    /**< [in] the proclist contents */
                          const std::string &name,    /**< [in] the app name */
                          proclistEdit       edit,    /**< [in] what to do */
                          const std::string &newLine, /**< [in] the line to append for uncomment or append, if none */
                          int               &changed /**< [out] the number of lines changed */ );

/// Convert a systemd time span, as `systemctl show` prints it (e.g. "1min 30s", "500ms", "infinity"), to seconds
/**
 * \returns the seconds, a large number for infinity, or -1 if it cannot be parsed
 */
double timespanSeconds( const std::string &span /**< [in] the time span */ );

/// The systemd directives `[systemd]` sections may set, one name per line, `#` comments
typedef std::set<std::string> allowList;

/// Read a directive allow list, e.g. /etc/magaox/systemd-directives.local.txt
/**
 * \returns the directive names; names that are not plain words are skipped
 */
allowList parseAllowList( std::istream &in /**< [in] the list */ );

/// Collect systemd directives from the `[systemd]` section of a configuration
/** Each key is a systemd directive, in any case: `runtimeMaxSec = 20` and `cpuAffinity = 2` set `RuntimeMaxSec=20` and
 * `CPUAffinity=2`, spelled as this systemd's list has them.
 * A key is used only if it is in the allow list, is a directive this systemd has in [Unit] or [Service], is not
 * denied (deniedDirective()), is not set twice, and its value has no control characters.  Any other is reported in
 * warnings and ignored.  Values are checked by systemd itself, in xCtrl::dropBadValues().
 */
void systemdSettings( directiveMap             &dirs,     /**< [in.out] directives, updated with the settings found */
                      std::vector<std::string> &warnings, /**< [in.out] directives ignored, and why, appended to */
                      const confMap            &conf,     /**< [in] the configuration */
                      const std::string        &source,   /**< [in] whose configuration, for messages */
                      const allowList          &allow,    /**< [in] the directives that may be set */
                      const directiveSections  &sections = systemdDirectives() /**< [in] [optional] the directives
                                                                                  systemd accepts, for tests */ );

/// The systemd directives of one app: the role's, from magaox.conf, with the app's own .conf settings in their place
/** All are written to the app's drop-in, none to the template, so each directive is set in one file only: an app's
 * value replaces the role's, even for directives systemd merges across files, e.g. CPUAffinity.  Apps that do not
 * notify (`-` prefix) get Type=simple and no watchdog, whatever is set.
 *
 * \returns the directives
 */
directiveMap appDirectives( const appEntry     &app, /**< [in] the app */
                            const directiveMap &own, /**< [in] the app's settings, from its .conf */
                            const directiveMap &role /**< [in] the role's settings, from magaox.conf */ );

/// The `systemd-analyze verify` messages about lines of a unit file, the lines systemd ignores
/** systemd reports a value it can not use as `PATH:LINE: Failed to parse ..., ignoring: VALUE`.
 *
 * \returns line number: message, for each message about a line of path that says it is ignored
 */
std::map<int, std::string> parseVerify( const std::string &output, /**< [in] the output of systemd-analyze verify */
                                        const std::string &path /**< [in] the unit file verified */ );

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

/// Append "Directive=value" lines to os for the directives of one section, Type first, then by name
void appendDirectives( std::ostringstream      &os,          /**< [in.out] the unit being generated */
                       const directiveMap      &dirs,        /**< [in] the directives */
                       bool                     unitSection, /**< [in] true for [Unit], false for [Service] */
                       const directiveSections &sections = systemdDirectives() /**< [in] [optional] the directives'
                                                                                  sections, for tests */ );

/// Generate the service template of a role
/** It sets what xctrl needs, and no `[systemd]` settings: those are in each app's drop-in (appDirectives()), and
 * where neither magaox.conf nor the app's .conf sets a directive, systemd's own default applies.
 *
 * \returns the unit file contents
 */
std::string templateUnit( const unitNames   &units, /**< [in] the role's unit names */
                          const std::string &envFile /**< [in] optional environment file for all apps */ );

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

/// Root for one block of code, in a setuid run that has dropped to the caller
/** xctrl gives up root at startup, keeping only the saved uid 0, and takes it back with one of these around each
 * action that needs it: writing or removing generated files, opening the log, running systemctl.  The effective gid
 * goes to 0 too, so files and directories come out root:root.  Nothing happens if root is not ours to take back, or
 * we already are root (sudo).  Both ids are restored when it goes out of scope.
 *
 * \ingroup xctrl
 */
class rootScope
{
  protected:
    uid_t m_uid;             ///< the effective uid to return to
    gid_t m_gid;             ///< the effective gid to return to
    bool  m_raised{ false }; ///< whether this raised them

  public:
    /// Take root back, if it is ours to take
    rootScope();

    /// Return to the caller's effective ids
    ~rootScope();
};

/// Whether the file at path holds exactly content
bool sameContent( const std::string &path, /**< [in] the file */
                  const std::string &content /**< [in] the expected contents */ );

/// Write content to path if it differs from what is there, creating directories as needed
/**
 * \returns 1 if written, 0 if unchanged, -1 on error with a message on stderr
 */
int writeIfChanged( const std::string &path,    /**< [in] the file */
                    const std::string &content, /**< [in] its new contents */
                    mode_t             mode /**< [in] its permissions */ );

/// Run a command, without a shell
/** The command runs as the calling user, or, if root is ours to take, fully as root.
 *
 * \returns the exit status, or -1 if it could not be run
 */
int runCommand( const std::vector<std::string> &argv, /**< [in] the command and its arguments */
                std::string *output   = nullptr,      /**< [out] [optional] its standard output, else inherited */
                bool         asCaller = false,        /**< [in] [optional] run as the calling user, not as root */
                bool         stdErr   = false /**< [in] [optional] capture its standard error in output too */ );

/// How an app's settings in systemd compare with its .conf
/**
 * \ingroup xctrl
 */
struct settingsState
{
    bool                     ok{ false };        ///< the .conf and executable could be checked
    std::vector<std::string> errors;             ///< why ok is false
    std::string              path;               ///< the app's drop-in
    std::string              content;            ///< the drop-in its .conf makes now
    bool                     fileStale{ false }; ///< the drop-in differs from content: not registered yet
    bool                     loadStale{ false }; ///< systemd has not loaded the drop-in on disk
    bool                     running{ false };   ///< the app is running, starting or reloading
    bool                     runStale{ false };  ///< running since before its registered settings last changed
    std::string              wdHave;             ///< the watchdog of the run, as systemd shows it, if it differs
    double                   wdWant{ -1 };       ///< the configured watchdog, seconds
};

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
    bool        m_verbose{ false };   ///< status shows a table with systemd and resource details
    int         m_lines{ -1 };        ///< log lines for peek and inspect, -1 for their defaults
    std::string m_prog{ "xctrl" };    ///< the name we were run as
    bool        m_color{ false };     ///< color the output, when it is a terminal and NO_COLOR is not set
    bool      m_privileged{ false };  ///< may edit the proclist and sync it: root, or a user who can write the proclist
    bool      m_allowRead{ false };   ///< the directive allow list has been read
    bool      m_allowOk{ false };     ///< it was there, and safe to use
    allowList m_allow;                ///< the directive allow list
    std::string m_allowError;         ///< why it can not be used

    /// systemd's verdict on each `[systemd]` value checked: "Directive=value": "" if valid, else systemd's message
    std::map<std::string, std::string> m_valueChecks;

    bool        m_valuesUnchecked{ false }; ///< systemd-analyze is missing, so values go to systemd unchecked
    bool        m_proclistChecked{ false }; ///< warnIfProclistChanged() has run
    bool        m_rolesChecked{ false };    ///< warnOtherRoles() has run
    std::string m_cmdLine;                  ///< the command line, for the log
    std::string m_verb;                     ///< the command
    std::vector<std::string> m_names;       ///< the app names given

    ///@}

    /** \name Paths
     * @{
     */

    /// Directory of the systemd units
    std::string unitDir() const;

    /// The optional environment file for all apps of the role, as seen by systemd
    std::string envFile() const;

    /// The proclist of the role
    std::string proclistPath() const;

    /// The root-owned allow list of the systemd directives `[systemd]` sections may set
    std::string allowListPath() const;

    ///@}

    /// Print usage
    void usage( std::ostream &os /**< [in] where to print */ ) const;

    /// Whether the calling user may start and stop apps: root, xsup, or a member of group magaox
    bool operatorAllowed() const;

    /// Settle privileges for the command before running it
    /** A setuid run drops to the caller's effective ids at once, keeping the saved uid 0, so rootScope can take root
     * back for the few actions that need it; commands that never need root drop it for good.  The caller must be an
     * editor (root, or able to write the proclist) for sync, add, enable, disable, delete and boot, and an operator
     * (root, xsup, group magaox) for startup, start, shutdown, stop and restart.  Test runs never keep root.
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

    /// The names of the registered apps: those with a drop-in, magaox-<role>-app@NAME.service.d/override.conf
    std::vector<std::string> registered() const;

    /// Read a registered app back from its drop-in: the executable from ExecStart, the - prefix from Type=simple
    /**
     * \returns true if the app is registered
     */
    bool registryRead( appEntry          &app, /**< [out] the app */
                       const std::string &name /**< [in] its name */ ) const;

    /// Whether path is one of the files xctrl generates for this role, which it may write or remove as root
    bool generatedPath( const std::string &path /**< [in] the path */ ) const;

    /// Remove an app's drop-in, as root; the only removals xctrl makes as root, with removeLegacy()
    /**
     * \returns 0, or -1 with a message
     */
    int removeDropIn( const std::string &name /**< [in] the app */ );

    /// Remove what older xctrl versions kept: /etc/magaox-<role>/apps and
    /// sys/sentinel_proclist/proclist_<role>.txt.active
    void removeLegacy();

    /// Whether an app is enabled for automatic startup
    bool enabled( const std::string &name /**< [in] the app */ ) const;

    /// Check that the given names are registered apps
    /**
     * \returns the apps, empty with messages on stderr if any is not registered
     */
    std::vector<appEntry> namedApps();

    /// Replace --all by every app in the proclist, active or disabled (#!), and every registered app
    /**
     * \returns 0 if there are names to work on, -1 with a message on stderr if not
     */
    int namesOrAll();

    /// The service instances of the role that are active or activating
    std::vector<std::string> activeApps();

    /// The apps of a role on this host: registered with xctrl, and running under systemd
    struct roleApps
    {
        std::string              name;          ///< the role as set, e.g. ICC, from its template, else in lower case
        std::vector<std::string> registered;    ///< apps with a drop-in, magaox-<role>-app@NAME.service.d
        std::vector<std::string> running;       ///< apps whose magaox-<role>-app@NAME.service is active or starting
        std::vector<std::string> autostart;     ///< apps enabled to start with the role's target
        bool                     boot{ false }; ///< the role's target starts at boot (xctrl boot on)
    };

    /// Every role with apps registered or running on this host, this one included, by lower-case role
    std::map<std::string, roleApps> hostRoles();

    /// Warn, once, about apps of other roles still running on this host
    void warnOtherRoles();

    /// Warn about other roles that start apps at boot, which could conflict with this role's after a reboot
    void warnOtherRolesBoot();

    /// Start apps: ssh tunnels, then the INDI server, then the others
    /**
     * \returns 0 if all started, else non-zero
     */
    int startApps( const std::vector<appEntry> &apps /**< [in] the apps */ );

    /// Stop apps: the others, then the INDI server, then ssh tunnels
    /**
     * \returns 0 if all stopped, else non-zero
     */
    int stopApps( const std::vector<std::string> &names /**< [in] the app names */ );

    /// The directive allow list, read once: it must exist, and, outside tests, be owned by root and writable by
    /// root only, as must its directory; and this systemd's list of directives must be known
    /**
     * \returns the list, or nullptr with the reason in m_allowError
     */
    const allowList *allowedDirectives();

    /// Have systemd check the `[systemd]` values not checked yet, all at once, with `systemd-analyze verify` on a
    /// scratch unit holding them, run as the caller
    /** The results go to m_valueChecks.  Without systemd-analyze, m_valuesUnchecked is set and values pass.
     */
    void checkValues( const std::vector<directiveMap> &sets /**< [in] the settings to check */ );

    /// Check the `[systemd]` values of magaox.conf and of these apps' .conf in one systemd-analyze run
    void checkConfValues( const std::vector<std::string> &names /**< [in] the app names */ );

    /// Remove the directives whose values systemd does not accept, with a warning for each
    void dropBadValues( directiveMap             &dirs,   /**< [in.out] the settings */
                        const std::string        &source, /**< [in] whose settings, for messages */
                        std::vector<std::string> &warnings /**< [in.out] directives ignored, appended to */ );

    /// The role-wide directives: the `[systemd]` settings of magaox.conf
    directiveMap roleDirectives( std::vector<std::string> &warnings, /**< [in.out] directives ignored, appended to */
                                 const allowList          &allow /**< [in] the directives that may be set */ );

    /// Check that an app can be registered, and work out its systemd directives from magaox.conf and its .conf
    /**
     * \returns true if it can, false with the reasons appended to errors
     */
    bool prepareApp( const appEntry           &app,      /**< [in] the app */
                     directiveMap             &dirs,     /**< [out] its directives */
                     std::vector<std::string> &errors,   /**< [in.out] why it can not be registered, appended to */
                     std::vector<std::string> &warnings, /**< [in.out] directives ignored, appended to */
                     const allowList          &allow,    /**< [in] the directives that may be set */
                     const directiveMap       &role,     /**< [in] the role's directives, roleDirectives() */
                     bool                     *hasConf = nullptr /**< [out] [optional] whether it has a .conf */ );

    /// Find the role's INDI server and ssh tunnels among apps, for the drop-ins' ordering
    void serverAndTunnels( const std::vector<appEntry> &apps,   /**< [in] the apps */
                           std::string                 &server, /**< [out] the INDI server, "" if none */
                           std::vector<std::string>    &tunnels /**< [out] the ssh tunnel apps */ );

    /// Warn, once, if the proclist differs from what is registered: the drop-ins and autostart links
    /** Compares the apps, their autostart (`#!`), their watchdog (`-`) and executables, not comments or order.
     */
    void warnIfProclistChanged();

    /// Edit an app's lines in the role's proclist, as the caller, in place so it keeps its owner and permissions
    /**
     * \returns the number of lines changed, or -1 with a message on stderr
     */
    int updateProclist( const std::string &name, /**< [in] the app name */
                        proclistEdit       edit, /**< [in] what to do */
                        const std::string &newLine = "" /**< [in] the line to append for uncomment, if none */ );

    /// Find an app in the registry, else in the proclist, active or disabled (#!)
    /**
     * \returns true if found
     */
    bool findApp( appEntry          &app, /**< [out] the app */
                  const std::string &name /**< [in] its name */ );

    /// Register one app from its .conf, without reloading systemd
    /**
     * \returns 1 if its drop-in was written, 0 if it was up to date, -1 with messages if not registered
     */
    int registerApp( const appEntry &app /**< [in] the app */ );

    /// Stop, disable and unregister apps, without reloading systemd
    /**
     * \returns 0 if all went well, else non-zero
     */
    int deregister( const std::vector<std::string> &names /**< [in] the app names */ );

    /// Before apps start, warn about any whose registered settings differ from their .conf
    /** Nothing is written or reloaded: the apps start as they are registered, and sync applies changes.
     *
     * \returns 0 if all went well, else non-zero
     */
    int refreshApps( const std::vector<appEntry> &apps /**< [in] the apps about to start */ );

    /// The role's INDI server and ssh tunnels among the registered apps, which order every drop-in
    void registeredOrdering( std::string              &server, /**< [out] the INDI server, "" if none */
                             std::vector<std::string> &tunnels /**< [out] the ssh tunnel apps */ );

    /// Compare an app's drop-in, what systemd has loaded, and what it runs with, against its .conf
    /**
     * \returns the comparison; ok is false, with the reasons, if the .conf or executable has a problem
     */
    settingsState checkSettings( const appEntry                     &app,     /**< [in] the app */
                                 const directiveMap                 &role,    /**< [in] the role-wide directives */
                                 const std::string                  &server,  /**< [in] the role's INDI server */
                                 const std::vector<std::string>     &tunnels, /**< [in] the role's ssh tunnels */
                                 std::map<std::string, std::string> &props /**< [in] systemctl show of the app */ );

    /// Warn about running apps whose settings changed since they started, and how to apply the new ones
    void warnOutdatedRunning(
        const std::vector<std::string>                            &names, /**< [in] the apps to check */
        std::map<std::string, std::map<std::string, std::string>> &props /**< [in] systemctl show of the apps */ );

    /** \name Commands
     * @{
     */

    /// Bring systemd in line with the proclist and the apps' .conf: register new and changed apps, unregister
    /// removed ones that are not running, set autostart links; never start or stop apps
    /** Apps disabled (#!) in the proclist, or removed from it while running, keep running but lose their autostart
     * link, so neither startup nor a reboot starts them; the user is told which.
     *
     * \returns 0 if all went well, else non-zero
     */
    int doSync( bool summary /**< [in] list the roles on this host first */ );

    /// Whether the proclist or any app's .conf differs from what is registered
    bool syncNeeded();

    /// Warn about apps running that are no longer in the proclist
    void warnRemovedRunning();

    int cmdSync();     ///< Sync the role's apps with systemd: doSync()
    int cmdAdd();      ///< Add an app to the proclist, disabled, and register it
    int cmdStartup();  ///< Start enabled apps, or enable and start named ones
    int cmdShutdown(); ///< Stop all apps, or named ones
    int cmdStart();    ///< Start named apps
    int cmdStop();     ///< Stop named apps
    int cmdRestart();  ///< Restart named apps, or all
    int cmdStatus();   ///< Report the apps' service states, one line each, or a detailed table with -v
    int cmdPeek();     ///< Report state and the last log lines of apps
    int cmdInspect();  ///< Follow an app's console output in the journal
    int cmdEnable();   ///< Enable or disable named apps, or all, for automatic startup
    int cmdDelete();   ///< Remove named apps, or all, from the proclist, stop and unregister them
    int cmdBoot();     ///< Start the role at boot, or not

    ///@}
};

// ---------------------------------------------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------------------------------------------

inline bool appEntry::notifies() const
{
    return prefix != "-";
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
        // the <cctype> functions take an unsigned char value: a negative char, e.g. a UTF-8 byte, is undefined
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
    // 128 keeps magaox-<role>-app@NAME.service well within systemd's 256-character unit names; a first letter or
    // digit keeps a name from looking like an option, e.g. -n NAME
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

        // a blank line, or a first word followed by a # comment rather than an executable, e.g. a header
        if( toks.empty() || ( toks.size() > 1 && toks[1][0] == '#' ) )
        {
            continue;
        }

        // a disabled process, e.g. "#!camsci1 picamCtrl" or "#! -camsci1 picamCtrl"; any other "#" line is a comment
        bool commented = toks[0].compare( 0, 2, "#!" ) == 0;
        if( !commented && toks[0][0] == '#' )
        {
            continue;
        }
        if( commented )
        {
            std::string        rest = line.substr( line.find( "#!" ) + 2 ); // after the 2 characters of #!
            std::istringstream cs( rest );
            toks.clear();
            while( cs >> tok )
            {
                toks.push_back( tok );
            }
            if( toks.size() != 2 )
            {
                pl.warnings.push_back( "proclist line " + std::to_string( n ) + ": [" + trim( line ) +
                                       "] is not #!NAME EXECUTABLE, ignored; use # for comments" );
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

        if( commented )
        {
            if( validName( app.name ) && validName( app.exec ) )
            {
                pl.disabled.push_back( app );
            }
            else
            {
                pl.warnings.push_back( "proclist line " + std::to_string( n ) + ": [" + trim( line ) +
                                       "] is not a valid #!NAME EXECUTABLE, ignored; use # for comments" );
            }
            continue;
        }

        if( !validName( app.name ) )
        {
            pl.errors.push_back( "proclist line " + std::to_string( n ) + ": invalid name [" + toks[0] +
                                 "]: use letters, digits, _ . and -, with a - prefix if needed" );
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

    // an active line wins over a disabled one for the same name, and the first disabled one wins
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
    // whether line is NAME EXECUTABLE for this name, with any prefix, and whether it is disabled, #!
    auto match = [&]( const std::string &line, bool &commented ) -> bool
    {
        std::string t = trim( line );
        commented     = t.compare( 0, 2, "#!" ) == 0;
        if( !commented && !t.empty() && t[0] == '#' )
        {
            return false; // only a comment, never an app's line
        }
        t.erase( 0, commented ? 2 : 0 ); // drop the #!
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
        return n == name;
    };

    std::vector<std::string> lines;
    std::istringstream       in( text );
    std::string              line;
    // first, whether the app has an active line (then enable has nothing to do) and any line at all (add appends
    // only if not)
    bool hasActive = false, hasAny = false;
    while( std::getline( in, line ) )
    {
        bool c, m = match( line, c );
        hasActive = hasActive || ( m && !c );
        hasAny    = hasAny || m;
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
            l = "#!" + l;
            ++changed;
        }
        else if( edit == proclistEdit::uncomment && c && !hasActive ) // only the first disabled line
        {
            // drop the #! and any spaces after it, keeping the - prefix if there is one
            l.erase( 0, l.find( "#!" ) + 2 );
            l.erase( 0, l.find_first_not_of( " \t" ) );
            hasActive = true;
            ++changed;
        }
        out << l << "\n";
    }
    // an app with no line to enable, or not in the proclist at all for add, gets newLine at the end
    if( ( ( edit == proclistEdit::uncomment && !hasActive ) || ( edit == proclistEdit::append && !hasAny ) ) &&
        !newLine.empty() )
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
    // each token is a number and a unit, e.g. "1min" "30.5s": i ends at the first character of the unit
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
            section = trim( line.substr( 1, line.size() - 2 ) ); // the name between the brackets
            continue;
        }

        size_t eq = line.find( '=' );
        if( eq == std::string::npos )
        {
            continue;
        }

        std::string key = trim( line.substr( 0, eq ) );
        std::string val = trim( line.substr( eq + 1 ) );

        // TOML values: "string", 'string' or [array], kept as the text inside
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

inline directiveSections parseDirectiveDump( const std::string &dump )
{
    directiveSections  dirs;
    std::istringstream ss( dump );
    std::string        section;
    for( std::string l; std::getline( ss, l ); )
    {
        if( !l.empty() && l[0] == '[' )
        {
            section = l;
        }
        else if( ( section == "[Unit]" || section == "[Service]" ) && l.find( '=' ) != std::string::npos )
        {
            dirs.insert( { l.substr( 0, l.find( '=' ) ), section == "[Unit]" } ); // insert keeps the first
        }
    }
    return dirs;
}

inline const directiveSections &systemdDirectives()
{
    static const directiveSections dirs = []()
    {
        for( const char *bin : { "/usr/lib/systemd/systemd", "/lib/systemd/systemd" } )
        {
            std::string out;
            if( access( bin, X_OK ) == 0 && runCommand( { bin, "--dump-configuration-items" }, &out, true ) == 0 )
            {
                return parseDirectiveDump( out );
            }
        }
        return directiveSections();
    }();
    return dirs;
}

inline bool deniedDirective( const std::string &d )
{
    static const std::set<std::string> denied = {
        // set by xctrl: the command, how the app reports, and the user it runs as
        "Type",
        "NotifyAccess",
        "User",
        "Group",
        "DynamicUser",
        // more privileges than xsup's
        "SupplementaryGroups",
        "AmbientCapabilities",
        "CapabilityBoundingSet",
        "SecureBits",
        "NoNewPrivileges",
        "PAMName",
        "PermissionsStartOnly",
        "RootDirectoryStartOnly",
        // actions on the whole host, e.g. a reboot
        "FailureAction",
        "SuccessAction",
        "StartLimitAction",
        "JobTimeoutAction" };

    // ExecStart, ExecStartPre, ExecStop, ...: commands, which a + prefix would run as root
    return denied.count( d ) > 0 || d.compare( 0, 4, "Exec" ) == 0;
}

inline allowList parseAllowList( std::istream &in )
{
    allowList allow;
    for( std::string l; std::getline( in, l ); )
    {
        std::string d = trim( l.substr( 0, l.find( '#' ) ) );
        if( !d.empty() && std::all_of( d.begin(), d.end(), []( char c ) { return isalnum( (unsigned char)c ); } ) )
        {
            allow.insert( d );
        }
    }
    return allow;
}

inline void systemdSettings( directiveMap             &dirs,
                             std::vector<std::string> &warnings,
                             const confMap            &conf,
                             const std::string        &source,
                             const allowList          &allow,
                             const directiveSections  &sections )
{
    std::map<std::string, std::string> seen; // directive: the key that set it, to catch two spellings of one
    for( const auto &kv : conf )
    {
        // keys of the [systemd] section start with the 8 characters "systemd."
        if( kv.first.compare( 0, 8, "systemd." ) != 0 || kv.first.size() == 8 )
        {
            continue;
        }
        // the directive as systemd spells it, whatever the key's case: cpuAffinity is CPUAffinity; one systemd does
        // not have keeps the key, with its first letter in upper case: fooSec is FooSec
        std::string key = kv.first.substr( 8 ), d = key;
        d[0] = static_cast<char>( toupper( static_cast<unsigned char>( d[0] ) ) );
        for( const auto &sd : sections )
        {
            d = lower( sd.first ) == lower( key ) ? sd.first : d;
        }

        std::string why;
        if( seen.count( d ) )
        {
            why = " directive for " + source + " is also set as " + seen[d];
        }
        else if( deniedDirective( d ) )
        {
            why = " directive for " + source + " can not be set: xctrl sets it, or it could give the app more " +
                  "privileges than xsup has or act on the whole host";
        }
        else if( !allow.count( d ) )
        {
            why = " directive for " + source + " not in systemd-directives.local.txt allow list";
        }
        else if( !sections.count( d ) )
        {
            why = " directive for " + source + " does not exist on this version of systemd";
        }
        else if( std::any_of( kv.second.begin(),
                              kv.second.end(),
                              []( char c ) { return iscntrl( static_cast<unsigned char>( c ) ); } ) )
        {
            // a control character, e.g. a newline, could start another directive in the unit file
            why = " = " + kv.second + " for " + source + " is an invalid value";
        }

        if( why.empty() )
        {
            dirs[d] = kv.second;
            seen[d] = key;
        }
        else
        {
            warnings.push_back( "[systemd] " + key + why + "; ignoring directive" );
        }
    }
}

inline directiveMap appDirectives( const appEntry &app, const directiveMap &own, const directiveMap &role )
{
    directiveMap dirs = role;
    for( const auto &kv : own )
    {
        dirs[kv.first] = kv.second;
    }

    if( !app.notifies() )
    {
        dirs["Type"]        = "simple";
        dirs["WatchdogSec"] = "0";
    }

    return dirs;
}

inline std::map<int, std::string> parseVerify( const std::string &output, const std::string &path )
{
    std::map<int, std::string> bad;
    std::istringstream         in( output );
    for( std::string l; std::getline( in, l ); )
    {
        // PATH:LINE: message
        if( l.compare( 0, path.size() + 1, path + ":" ) != 0 )
        {
            continue;
        }
        size_t c = l.find( ':', path.size() + 1 );
        int    n = atoi( l.c_str() + path.size() + 1 );
        if( c != std::string::npos && n > 0 && l.find( "ignoring" ) != std::string::npos )
        {
            bad[n] = trim( l.substr( c + 1 ) );
        }
    }
    return bad;
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

inline void appendDirectives( std::ostringstream      &os,
                              const directiveMap      &dirs,
                              bool                     unitSection,
                              const directiveSections &sections )
{
    // Type first, it reads best there
    if( !unitSection && dirs.count( "Type" ) )
    {
        os << "Type=" << dirs.at( "Type" ) << "\n";
    }

    // then the others, in their section; [Service] if systemd's list does not have it
    for( const auto &kv : dirs )
    {
        auto it = sections.find( kv.first );
        if( kv.first != "Type" && ( it != sections.end() && it->second ) == unitSection )
        {
            os << kv.first << "=" << kv.second << "\n";
        }
    }
}

inline std::string templateUnit( const unitNames &units, const std::string &envFile )
{
    std::ostringstream os;
    os << "# Generated by xctrl sync for MAGAOX_ROLE=" << units.role << ".  Do not edit.  The [systemd] settings of\n"
       << "# magaox.conf, and of each app's .conf, are in the apps' drop-ins.\n"
       << "[Unit]\n"
       << "Description=MagAO-X app %i (role " << units.role << ")\n"
       << "PartOf=" << units.target() << "\n"
       << "Wants=network-online.target\n"
       << "After=network-online.target\n"
       << "\n[Service]\n"
       << "Type=notify\n"
       << "NotifyAccess=main\n"
       << "User=xsup\n"
       << "Environment=MAGAOX_ROLE=" << units.role << "\n"
       << "# Python apps write their output to the journal as it happens\n"
       << "Environment=PYTHONUNBUFFERED=1\n"
       << "EnvironmentFile=-" << envFile << "\n"
       << "# Each registered app's drop-in sets its ExecStart; an unregistered instance fails\n"
       << "ExecStart=/bin/false\n"
       << "KillMode=control-group\n"
       << "SyslogIdentifier=%i\n"
       << "\n[Install]\n"
       << "WantedBy=" << units.target() << "\n";
    return os.str();
}

inline std::string targetUnit( const unitNames &units )
{
    std::ostringstream os;
    os << "# Generated by xctrl sync for MAGAOX_ROLE=" << units.role << ".  Do not edit.\n"
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
    os << "# Generated by xctrl sync from proclist_" << units.role << ".txt, magaox.conf and " << app.name << ".conf.\n"
       << "# Do not edit: change those, then rerun xctrl sync.\n"
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

inline rootScope::rootScope() : m_uid( geteuid() ), m_gid( getegid() )
{
    uid_t r, e, s;
    if( m_uid != 0 && getresuid( &r, &e, &s ) == 0 && s == 0 && seteuid( 0 ) == 0 )
    {
        m_raised = true;
        if( setegid( 0 ) < 0 )
        {
            // files keep the caller's group
        }
    }
}

inline rootScope::~rootScope()
{
    // the group first, while we are still root; staying root by mistake is not an option
    if( m_raised && ( setegid( m_gid ) < 0 || seteuid( m_uid ) < 0 ) )
    {
        std::cerr << "xctrl: could not give up root again: " << strerror( errno ) << "\n";
        std::abort();
    }
}

inline bool sameContent( const std::string &path, const std::string &content )
{
    std::ifstream f( path );
    if( !f )
    {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str() == content;
}

inline int writeIfChanged( const std::string &path, const std::string &content, mode_t mode )
{
    if( sameContent( path, content ) )
    {
        return 0;
    }

    std::error_code ec;
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path(), ec );
    if( ec )
    {
        say( msgLevel::error ) << "creating directory for " << path << ": " << ec.message();
        return -1;
    }

    std::string tmp = path + ".xctrl.tmp";
    {
        std::ofstream f( tmp, std::ios::trunc );
        f << content;
        if( !f.good() )
        {
            say( msgLevel::error ) << "writing " << tmp << ": " << strerror( errno );
            return -1;
        }
    }
    if( chmod( tmp.c_str(), mode ) < 0 || rename( tmp.c_str(), path.c_str() ) < 0 )
    {
        say( msgLevel::error ) << "installing " << path << ": " << strerror( errno );
        unlink( tmp.c_str() );
        return -1;
    }
    return 1;
}

inline int runCommand( const std::vector<std::string> &argv, std::string *output, bool asCaller, bool stdErr )
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
        // in the child: its stdout (fd 1), and stderr (fd 2) if asked, go into the pipe we read
        if( output )
        {
            dup2( pfd[1], 1 );
            if( stdErr )
            {
                dup2( pfd[1], 2 );
            }
            close( pfd[0] );
            close( pfd[1] );
        }
        // become wholly the caller, or, if root is ours (saved or effective uid 0, setuid or sudo), wholly root, as
        // systemctl needs.  Dropping, the group goes first, while it may still change; raising, the uid goes first,
        // from the saved uid, as only root may set the group to 0.  126, as a shell does, for a command not run.
        uid_t r, e, s;
        getresuid( &r, &e, &s );
        if( asCaller )
        {
            gid_t g = getgid();
            if( setresgid( g, g, g ) < 0 || setresuid( r, r, r ) < 0 )
            {
                _exit( 126 );
            }
        }
        else if( e == 0 || s == 0 )
        {
            if( setresuid( 0, 0, 0 ) < 0 || setresgid( 0, 0, 0 ) < 0 )
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
        say( msgLevel::error ) << "running " << argv[0] << ": " << strerror( errno );
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
    // the exit code, or -1 if it was killed by a signal
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
    // failed, or stopped by something other than a clean stop: the result says what, e.g. watchdog or exit-code
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

inline std::string xCtrl::envFile() const
{
    return "/etc/magaox-" + lower( m_role ) + "/environment";
}

inline std::string xCtrl::proclistPath() const
{
    return m_top + "/" + MAGAOX_configRelPath + "/proclist_" + m_role + ".txt";
}

inline std::string xCtrl::allowListPath() const
{
    return m_sysroot + "/etc/magaox/systemd-directives.local.txt";
}

inline void xCtrl::usage( std::ostream &os ) const
{
    os << "usage: " << m_prog << " [-r ROLE] [options] COMMAND [NAME ...]\n"
       << "\n"
          "MagAO-X process control: run the apps of a role (MAGAOX_ROLE, or -r) as systemd services, one\n"
          "instance of magaox-<role>-app@.service per app in /opt/MagAOX/config/proclist_<role>.txt.\n"
          "\n"
          "Any user can look (inspect reads the journal: sudo, or group systemd-journal); xsup and\n"
          "members of group magaox can start and stop apps:\n"
          "  status [NAME...]       state and pid of the apps; -v adds autostart, restarts, uptime,\n"
          "                         last watchdog ping, CPU, memory and GPU memory\n"
          "  peek [NAME...]         state and last log lines of the apps (-n LINES, default 10)\n"
          "  inspect NAME           follow an app's console output (-n LINES, default 50)\n"
          "  startup [NAME...]      start the apps enabled for automatic startup, the INDI server first;\n"
          "                         with names, start those apps\n"
          "  start NAME...          start apps, enabled or not; systemd supervises them either way.\n"
          "                         For a user who can edit the proclist, start and startup sync first\n"
          "  stop NAME...           stop apps; they stay stopped until started again\n"
          "  shutdown NAME...|--all stop apps, or all running apps of the role, the INDI server last\n"
          "  restart NAME...|--all  restart apps, or stop then start everything\n"
          "\n"
          "Changing which apps are configured needs write access to the proclist (group magaox-dev):\n"
          "  sync                   bring systemd in line with the proclist and the apps' .conf: register\n"
          "                         new and changed apps, set autostart from #!, unregister apps no\n"
          "                         longer in the proclist unless they are running; starts and stops\n"
          "                         nothing.  [systemd] directives must be in the root-owned allow list\n"
          "                         /etc/magaox/systemd-directives.local.txt\n"
          "  add NAME EXECUTABLE    add the app to the proclist disabled (#!), and register it; it needs\n"
          "                         NAME.conf; add -- -NAME EXECUTABLE for an app without a watchdog\n"
          "  enable NAME...|--all   autostart on: remove the app's #! in the proclist (adding it if\n"
          "                         needed) and register it if it is not; does not start it\n"
          "  disable NAME...|--all  autostart off: disable the app with #! in the proclist; it stays\n"
          "                         registered to start by hand; does not stop it\n"
          "  delete NAME...|--all   remove the app from the proclist, stop and unregister it\n"
          "  boot on|off            start the role's enabled apps at boot, or not\n"
          "\n"
          "Options:\n"
          "  -r, --role ROLE      the MagAO-X role, else MAGAOX_ROLE or /etc/profile.d/magaox_role.sh\n"
          "  --all                all apps\n"
          "  -v, --verbose        status shows a table with more details\n"
          "  -q, --quiet          show no warnings, only the normal output and errors; they are still logged\n"
          "  --                   end of options, so a name may start with -\n"
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
    // who may do what, checked as the caller: editors change the proclist and what is registered, operators start
    // and stop; anyone looks
    static const std::set<std::string> edit      = { "sync", "add", "enable", "disable", "delete", "boot" };
    static const std::set<std::string> operate   = { "startup", "start", "shutdown", "stop", "restart" };
    bool                               setuidRun = geteuid() == 0 && getuid() != 0;
    bool                               testing = m_dryRun || !m_sysroot.empty() || m_top != std::string( MAGAOX_path );

    if( setuidRun )
    {
        // a setuid program trusts nothing from the caller's environment
        clearenv();
        setenv( "PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/MagAOX/bin", 1 );
        setenv( "SYSTEMD_PAGER", "", 1 );
        setenv( "PAGER", "cat", 1 );

        // from here on the caller's effective ids; the saved uid stays 0, so rootScope can take root back where needed
        if( setegid( getgid() ) < 0 || seteuid( getuid() ) < 0 )
        {
            std::cerr << "xctrl: dropping privileges: " << strerror( errno ) << "\n";
            return -1;
        }
    }

    // journalctl (peek, inspect) colors warnings and errors only on a terminal whose TERM it knows, and TERM was just
    // cleared; tell it directly, as decided for our own output, which also honors NO_COLOR
    setenv( "SYSTEMD_COLORS", m_color ? "1" : "0", 1 );

    // every command but looking is logged; a real run opens the log as root, so any user's commands go to the one
    // log.  A test run logs under --top as the caller, a dry run not at all.
    static const std::set<std::string> look = { "status", "lstatus", "peek", "inspect" };
    bool logged  = !look.count( m_verb ) && !m_dryRun && !( testing && m_top == std::string( MAGAOX_path ) );
    auto openLog = [&]()
    {
        if( xlog().open( m_top + "/" + MAGAOX_logRelPath ) == 0 )
        {
            struct passwd *pw = getpwuid( getuid() );
            xlog().message( msgLevel::info,
                            m_prog + m_cmdLine + " (user " + ( pw ? pw->pw_name : std::to_string( getuid() ) ) + ")",
                            false );
        }
    };
    if( logged && !testing )
    {
        rootScope root;
        openLog();
    }

    struct passwd *pw   = getpwuid( getuid() );
    std::string    user = pw ? pw->pw_name : std::to_string( getuid() );

    // an editor may write the proclist; access() checks the real user, the caller
    m_privileged = getuid() == 0 || access( proclistPath().c_str(), W_OK ) == 0;
    if( edit.count( m_verb ) && !m_privileged )
    {
        if( errno == ENOENT )
        {
            say( msgLevel::error ) << "no proclist " << proclistPath() << " for role " << m_role;
            return -1;
        }
        say( msgLevel::error ) << "cannot " << m_verb << " as user " << user << ": " << m_verb
                               << " needs write access to " << proclistPath() << " (group magaox-dev), or sudo";
        return -1;
    }
    if( operate.count( m_verb ) && !operatorAllowed() )
    {
        say( msgLevel::error ) << "cannot " << m_verb << " as user " << user << ": only root, xsup and members of "
                               << "group magaox can start and stop apps";
        return -1;
    }

    // for good, the saved uid too, for commands that never need root and for tests: root can not come back
    if( setuidRun && ( testing || !( edit.count( m_verb ) || operate.count( m_verb ) ) ) )
    {
        gid_t g = getgid();
        uid_t u = getuid();
        if( setresgid( g, g, g ) < 0 || setresuid( u, u, u ) < 0 )
        {
            say( msgLevel::error ) << "dropping privileges: " << strerror( errno );
            return -1;
        }
    }
    if( logged && testing )
    {
        openLog();
    }
    return 0;
}

inline int xCtrl::parseArgs( int argc, char **argv )
{
    for( int i = 1; i < argc; ++i )
    {
        m_cmdLine += std::string( " " ) + argv[i];
    }

    bool endOpts = false;
    for( int i = 1; i < argc; ++i )
    {
        std::string a     = argv[i];
        auto        value = [&]( std::string &dest ) -> bool
        {
            if( i + 1 >= argc )
            {
                say( msgLevel::error ) << a << " requires a value";
                return false;
            }
            dest = argv[++i];
            return true;
        };

        if( !endOpts && a == "--" )
        {
            endOpts = true; // what follows are names, e.g. add -- -NAME EXECUTABLE
        }
        else if( endOpts || a.size() < 2 || a[0] != '-' )
        {
            if( m_verb.empty() )
            {
                m_verb = a;
            }
            else
            {
                m_names.push_back( a );
            }
        }
        else if( a == "-h" || a == "--help" )
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
            m_role = a.substr( 7 ); // after the 7 characters of --role=
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
        else if( a == "-q" || a == "--quiet" )
        {
            xlog().quiet( true );
        }
        else if( a.size() > 2 && a[1] != '-' && a.find_first_not_of( "vq", 1 ) == std::string::npos )
        {
            // single-letter flags together, e.g. -vq
            m_verbose = m_verbose || a.find( 'v' ) != std::string::npos;
            if( a.find( 'q' ) != std::string::npos )
            {
                xlog().quiet( true );
            }
        }
        else
        {
            say( msgLevel::error ) << "unknown option " << a << "; try --help";
            return -1;
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
                // e.g. export MAGAOX_ROLE="ICC"; # comment: the value after the 12 characters of MAGAOX_ROLE=,
                // without quotes, up to a space or ;
                std::string v = trim( line.substr( p + 12 ) );
                v.erase( std::remove( v.begin(), v.end(), '"' ), v.end() );
                v.erase( std::remove( v.begin(), v.end(), '\'' ), v.end() );
                m_role = trim( v.substr( 0, v.find_first_of( " ;\t" ) ) );
            }
        }
    }

    if( m_verb.empty() )
    {
        say( msgLevel::error ) << "missing command; try --help";
        return -1;
    }
    if( m_top != MAGAOX_path && m_sysroot.empty() && !m_dryRun )
    {
        say( msgLevel::error ) << "--top is for testing, with --sysroot or --dry-run";
        return -1;
    }
    if( m_role.empty() || !validName( m_role ) )
    {
        say( msgLevel::error ) << "missing or invalid role [" << m_role << "]: give -r ROLE or set MAGAOX_ROLE";
        return -1;
    }
    for( const auto &n : m_names )
    {
        if( !validName( n ) && m_verb != "boot" && m_verb != "add" )
        {
            say( msgLevel::error ) << "invalid app name [" << n << "]";
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

    // with --sysroot only the commands that just edit files under it run, e.g. enable; the rest need a running systemd
    if( m_dryRun || ( !m_sysroot.empty() && !offline ) )
    {
        say( msgLevel::info ) << ( m_dryRun ? "[dry-run] " : "[sysroot, not run] " ) << line;
        return 0;
    }

    // what systemctl says goes to the console and the log too, as errors if it failed
    say( msgLevel::action ) << "+ " << line;
    std::string        out;
    int                rc = runCommand( argv, &out, false, true );
    std::istringstream ss( out );
    for( std::string l; std::getline( ss, l ); )
    {
        say( rc == 0 ? msgLevel::info : msgLevel::error ) << l;
    }
    if( rc == 0 )
    {
        xlog().message( msgLevel::success, line + ": done", false );
    }
    else
    {
        say( msgLevel::error ) << line << " failed" << ( rc > 0 ? " with exit status " + std::to_string( rc ) : "" );
    }
    return rc;
}

inline std::vector<std::string> xCtrl::registered() const
{
    unitNames                u( m_role );
    std::vector<std::string> names;
    std::error_code          ec;
    for( const auto &e : std::filesystem::directory_iterator( unitDir(), ec ) )
    {
        // magaox-<role>-app@NAME.service.d, without its .d (2 characters), names an app's instance
        std::string d = e.path().filename().string(), n;
        if( d.size() > 2 && d.compare( d.size() - 2, 2, ".d" ) == 0 )
        {
            n = u.appName( d.substr( 0, d.size() - 2 ) );
        }
        std::error_code ec2;
        if( validName( n ) && std::filesystem::exists( e.path() / "override.conf", ec2 ) )
        {
            names.push_back( n );
        }
    }
    std::sort( names.begin(), names.end() );
    return names;
}

inline bool xCtrl::registryRead( appEntry &app, const std::string &name ) const
{
    unitNames     u( m_role );
    std::ifstream f( unitDir() + "/" + u.instance( name ) + ".d/override.conf" );
    std::string   exec, type;
    for( std::string l; std::getline( f, l ); )
    {
        // ExecStart=/opt/MagAOX/bin/EXEC -n NAME, after the empty ExecStart= that clears the template's
        if( l.compare( 0, 10, "ExecStart=" ) == 0 && l.size() > 10 ) // 10 is the length of "ExecStart="
        {
            std::string cmd = l.substr( 10, l.find( ' ' ) - 10 );
            exec            = cmd.substr( cmd.find_last_of( '/' ) + 1 );
        }
        else if( l.compare( 0, 5, "Type=" ) == 0 ) // Type= is 5 characters
        {
            type = l.substr( 5 );
        }
    }
    if( exec.empty() )
    {
        return false;
    }
    app.name   = name;
    app.exec   = exec;
    app.prefix = type == "simple" ? "-" : "";
    return true;
}

inline bool xCtrl::generatedPath( const std::string &path ) const
{
    unitNames   u( m_role );
    std::string dir = unitDir() + "/";
    if( path.compare( 0, dir.size(), dir ) != 0 )
    {
        return false;
    }
    std::string rel = path.substr( dir.size() );
    if( rel == u.templ() || rel == u.target() )
    {
        return true;
    }

    // magaox-<role>-app@NAME.service.d/override.conf, with a valid NAME, so nothing like ../ gets through
    std::string tail = ".d/override.conf";
    if( rel.size() <= tail.size() || rel.compare( rel.size() - tail.size(), tail.size(), tail ) != 0 )
    {
        return false;
    }
    return validName( u.appName( rel.substr( 0, rel.size() - tail.size() ) ) );
}

inline int xCtrl::removeDropIn( const std::string &name )
{
    unitNames   u( m_role );
    std::string d = unitDir() + "/" + u.instance( name ) + ".d";
    if( !validName( name ) || !generatedPath( d + "/override.conf" ) )
    {
        say( msgLevel::error ) << "refusing to remove the drop-in of [" << name << "]";
        return -1;
    }
    if( m_dryRun )
    {
        say( msgLevel::info ) << "[dry-run] would remove " << d;
        return 0;
    }
    rootScope       root;
    std::error_code ec;
    std::filesystem::remove( d + "/override.conf", ec );
    std::filesystem::remove( d, ec ); // only if nothing else is there
    return 0;
}

inline void xCtrl::removeLegacy()
{
    std::string     apps = m_sysroot + "/etc/magaox-" + lower( m_role ) + "/apps";
    std::string     rec  = m_top + "/" + MAGAOX_sysRelPath + "/sentinel_proclist/proclist_" + m_role + ".txt.active";
    std::error_code ec;
    bool            has = std::filesystem::exists( apps, ec ) || std::filesystem::exists( rec, ec );
    if( !has || m_dryRun )
    {
        return;
    }
    rootScope root;
    for( const auto &e : std::filesystem::directory_iterator( apps, ec ) )
    {
        if( e.path().extension() == ".conf" && !e.is_symlink() )
        {
            std::filesystem::remove( e.path(), ec );
        }
    }
    std::filesystem::remove( apps, ec ); // only if empty
    std::filesystem::remove( rec, ec );
    say( msgLevel::info ) << "removed the old registry " << apps << " and record " << rec
                          << ": the drop-ins are the registry now";
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
        say( msgLevel::error ) << m_verb << " needs app names";
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
            say( msgLevel::error ) << n << " is not registered: add it to " << proclistPath() << ", then " << m_prog
                                   << " sync, by a user who can edit the proclist, registers it";
            ok = false;
        }
    }
    return ok ? apps : std::vector<appEntry>();
}

inline int xCtrl::namesOrAll()
{
    if( !m_all )
    {
        if( m_names.empty() )
        {
            say( msgLevel::error ) << m_verb << " needs app names, or --all";
            return -1;
        }
        return 0;
    }
    if( !m_names.empty() )
    {
        say( msgLevel::error ) << m_verb << " takes app names or --all, not both";
        return -1;
    }

    std::ifstream         pf( proclistPath() );
    proclist              pl = parseProclist( pf );
    std::set<std::string> all;
    for( const auto *list : { &pl.apps, &pl.disabled } )
    {
        for( const auto &a : *list )
        {
            all.insert( a.name );
        }
    }
    for( const auto &n : registered() )
    {
        all.insert( n );
    }
    if( all.empty() )
    {
        say( msgLevel::error ) << "no apps in " << proclistPath() << " or registered for role " << m_role;
        return -1;
    }
    m_names.assign( all.begin(), all.end() );
    return 0;
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
                &out,
                true );
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

inline const allowList *xCtrl::allowedDirectives()
{
    if( m_allowRead )
    {
        return m_allowOk ? &m_allow : nullptr;
    }
    m_allowRead = true;

    std::string path = allowListPath(), dir = path.substr( 0, path.rfind( '/' ) );
    struct stat fs, ds;
    if( lstat( path.c_str(), &fs ) < 0 || !S_ISREG( fs.st_mode ) )
    {
        m_allowError = "no directive allow list " + path + ": install it with sudo make install in utils/xctrl";
        return nullptr;
    }

    // it decides what root writes into units, so only root may change it; tests (--sysroot) are exempt
    bool safe = [&]( const struct stat &s ) { return s.st_uid == 0 && !( s.st_mode & ( S_IWGRP | S_IWOTH ) ); }( fs );
    if( m_sysroot.empty() &&
        ( !safe || lstat( dir.c_str(), &ds ) < 0 || ds.st_uid != 0 || ( ds.st_mode & ( S_IWGRP | S_IWOTH ) ) ) )
    {
        m_allowError = path + " and " + dir + " must be owned by root and writable only by root";
        return nullptr;
    }

    // which directives exist, and in which section, comes from systemd itself
    if( systemdDirectives().empty() )
    {
        m_allowError = "cannot read the directives of this host's systemd (systemd --dump-configuration-items)";
        return nullptr;
    }

    std::ifstream f( path );
    m_allow   = parseAllowList( f );
    m_allowOk = true;
    return &m_allow;
}

inline void xCtrl::checkValues( const std::vector<directiveMap> &sets )
{
    static const char *analyze = []() -> const char *
    {
        for( const char *p : { "/usr/bin/systemd-analyze", "/bin/systemd-analyze" } )
        {
            if( access( p, X_OK ) == 0 )
            {
                return p;
            }
        }
        return nullptr;
    }();

    // a scratch unit: each value not checked yet on a line of its own, in its section
    const directiveSections &sections = systemdDirectives();
    std::vector<std::string> unitLines{ "[Unit]" }, serviceLines{ "[Service]", "ExecStart=/bin/true" };
    std::set<std::string>    added;
    for( const auto &set : sets )
    {
        for( const auto &kv : set )
        {
            std::string line = kv.first + "=" + kv.second;
            if( !m_valueChecks.count( line ) && added.insert( line ).second )
            {
                auto it = sections.find( kv.first );
                ( it != sections.end() && it->second ? unitLines : serviceLines ).push_back( line );
            }
        }
    }
    if( added.empty() )
    {
        return;
    }
    for( const auto &l : added )
    {
        m_valueChecks[l] = "";
    }
    if( !analyze )
    {
        m_valuesUnchecked = true;
        return;
    }

    // written and verified as the caller, in a directory of its own
    char dir[] = "/tmp/xctrl.XXXXXX";
    if( !mkdtemp( dir ) )
    {
        m_valuesUnchecked = true;
        return;
    }
    std::string path = std::string( dir ) + "/xctrl-check.service", out;
    {
        std::ofstream f( path );
        for( const auto *lines : { &unitLines, &serviceLines } )
        {
            for( const auto &l : *lines )
            {
                f << l << "\n";
            }
        }
    }
    runCommand( { analyze, "verify", path }, &out, true, true );
    unlink( path.c_str() );
    rmdir( dir );

    // line n of the unit: the [Unit] lines, then the [Service] lines
    for( const auto &kv : parseVerify( out, path ) )
    {
        size_t i = kv.first - 1;
        if( i < unitLines.size() + serviceLines.size() )
        {
            const std::string &line = i < unitLines.size() ? unitLines[i] : serviceLines[i - unitLines.size()];
            if( m_valueChecks.count( line ) )
            {
                m_valueChecks[line] = kv.second;
            }
        }
    }
}

inline void xCtrl::checkConfValues( const std::vector<std::string> &names )
{
    const allowList *allow = allowedDirectives();
    if( !allow )
    {
        return;
    }
    std::vector<std::string>  ignored; // reported where the settings are used
    std::vector<directiveMap> sets( 1 );
    std::string               cfgDir = m_top + "/" + MAGAOX_configRelPath;
    confMap                   gconf;
    readConf( gconf, cfgDir + "/" + MAGAOX_globalConfig );
    systemdSettings( sets[0], ignored, gconf, MAGAOX_globalConfig, *allow );
    for( const auto &n : names )
    {
        confMap c;
        sets.emplace_back();
        readConf( c, cfgDir + "/" + n + ".conf" );
        systemdSettings( sets.back(), ignored, c, n, *allow );
    }
    checkValues( sets );
}

inline void xCtrl::dropBadValues( directiveMap &dirs, const std::string &source, std::vector<std::string> &warnings )
{
    checkValues( { dirs } );
    for( auto it = dirs.begin(); it != dirs.end(); )
    {
        const std::string &why = m_valueChecks[it->first + "=" + it->second];
        if( why.empty() )
        {
            ++it;
            continue;
        }
        warnings.push_back( "[systemd] " + it->first + " = " + it->second + " for " + source +
                            " is an invalid value (systemd: " + why + "); ignoring directive" );
        it = dirs.erase( it );
    }
}

inline directiveMap xCtrl::roleDirectives( std::vector<std::string> &warnings, const allowList &allow )
{
    directiveMap dirs;
    confMap      gconf;
    readConf( gconf, m_top + "/" + MAGAOX_configRelPath + "/" + MAGAOX_globalConfig );
    systemdSettings( dirs, warnings, gconf, MAGAOX_globalConfig, allow );
    dropBadValues( dirs, MAGAOX_globalConfig, warnings );
    return dirs;
}

inline bool xCtrl::prepareApp( const appEntry           &app,
                               directiveMap             &dirs,
                               std::vector<std::string> &errors,
                               std::vector<std::string> &warnings,
                               const allowList          &allow,
                               const directiveMap       &role,
                               bool                     *hasConf )
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
    directiveMap own;
    systemdSettings( own, warnings, c, app.name, allow );
    dropBadValues( own, app.name, warnings );
    dirs = appDirectives( app, own, role );
    return true;
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

inline std::map<std::string, xCtrl::roleApps> xCtrl::hostRoles()
{
    std::map<std::string, roleApps> roles;
    std::error_code                 ec;

    // registered: each drop-in directory, magaox-<role>-app@NAME.service.d, holding an override.conf
    for( const auto &e : std::filesystem::directory_iterator( unitDir(), ec ) )
    {
        std::string     d  = e.path().filename().string();
        size_t          at = d.find( "-app@" );
        std::error_code ec2;
        if( d.compare( 0, 7, "magaox-" ) != 0 || at == std::string::npos || d.size() < 2 ||
            d.compare( d.size() - 2, 2, ".d" ) != 0 || !std::filesystem::exists( e.path() / "override.conf", ec2 ) )
        {
            continue;
        }
        std::string role = d.substr( 7, at - 7 ); // after the 7 characters of "magaox-"
        std::string name = unitNames( role ).appName( d.substr( 0, d.size() - 2 ) );
        if( validName( name ) )
        {
            roles[role].registered.push_back( name );
        }
    }

    // running: magaox-<role>-app@NAME.service units active or on their way, whether registered or not
    if( !m_dryRun && m_sysroot.empty() )
    {
        std::string out;
        runCommand(
            { "systemctl", "list-units", "--all", "--plain", "--no-legend", "--no-pager", "magaox-*-app@*.service" },
            &out,
            true );
        std::istringstream ss( out );
        for( std::string line; std::getline( ss, line ); )
        {
            std::istringstream ls( line );
            std::string        unit, load, active;
            ls >> unit >> load >> active;
            size_t at = unit.find( "-app@" );
            if( unit.compare( 0, 7, "magaox-" ) == 0 && at != std::string::npos &&
                ( active == "active" || active == "activating" || active == "reloading" ) )
            {
                std::string role = unit.substr( 7, at - 7 );
                roles[role].running.push_back( unitNames( role ).appName( unit ) );
            }
        }
    }

    // the role as set, e.g. ICC: units and directories use it in lower case, its template keeps it as it was.  Its
    // apps start at boot if its target is enabled (a link in multi-user.target.wants, which boot on makes) and they
    // are enabled with the target (links in its own .wants)
    for( auto &kv : roles )
    {
        unitNames   u( kv.first );
        std::string target = unitDir() + "/" + u.target();
        kv.second.boot     = std::filesystem::is_symlink( unitDir() + "/multi-user.target.wants/" + u.target(), ec );
        for( const auto &w : std::filesystem::directory_iterator( target + ".wants", ec ) )
        {
            std::string a = u.appName( w.path().filename().string() );
            if( !a.empty() )
            {
                kv.second.autostart.push_back( a );
            }
        }
        kv.second.name = kv.first;
        std::ifstream t( unitDir() + "/" + unitNames( kv.first ).templ() );
        for( std::string l; std::getline( t, l ); )
        {
            if( l.compare( 0, 23, "Environment=MAGAOX_ROLE" ) == 0 ) // Environment=MAGAOX_ROLE=ICC
            {
                kv.second.name = l.substr( 24 );
            }
        }
    }
    return roles;
}

inline void xCtrl::warnOtherRolesBoot()
{
    for( const auto &kv : hostRoles() )
    {
        const roleApps &r = kv.second;
        if( kv.first == lower( m_role ) || !r.boot || r.autostart.empty() )
        {
            continue;
        }
        say( msgLevel::warning ) << "role " << r.name << " starts " << r.autostart.size()
                                 << " app(s) at boot, which may conflict with role " << m_role
                                 << " after a reboot; sudo " << m_prog << " -r " << r.name << " boot off stops that";
    }
}

inline void xCtrl::warnOtherRoles()
{
    if( m_rolesChecked )
    {
        return;
    }
    m_rolesChecked = true;

    for( const auto &kv : hostRoles() )
    {
        const roleApps &r = kv.second;
        if( kv.first == lower( m_role ) || r.running.empty() )
        {
            continue;
        }
        std::string names;
        for( const auto &n : r.running )
        {
            names += " " + n;
        }
        say( msgLevel::warning ) << r.running.size() << " app(s) of role " << r.name << " still running:" << names
                                 << "; " << m_prog << " -r " << r.name << " shutdown --all stops them";
    }
}

inline void xCtrl::warnIfProclistChanged()
{
    if( m_proclistChecked )
    {
        return;
    }
    m_proclistChecked = true;

    std::ifstream pf( proclistPath() );
    if( !pf )
    {
        return;
    }

    // NAME: [#!][-]EXEC, as the proclist says and as it is registered (#! when not linked for autostart)
    std::map<std::string, std::string> now, reg;
    proclist                           pl = parseProclist( pf );
    for( const auto &a : pl.apps )
    {
        now[a.name] = a.prefix + a.exec;
    }
    for( const auto &a : pl.disabled )
    {
        now[a.name] = "#!" + a.prefix + a.exec;
    }
    for( const auto &n : registered() )
    {
        appEntry a;
        if( registryRead( a, n ) )
        {
            reg[n] = ( enabled( n ) ? "" : "#!" ) + a.prefix + a.exec;
        }
    }
    if( now == reg )
    {
        return;
    }

    std::string added, removed, changed;
    for( const auto &kv : now )
    {
        if( !reg.count( kv.first ) )
        {
            added += " " + kv.first;
        }
        else if( reg[kv.first] != kv.second )
        {
            changed += " " + kv.first;
        }
    }
    for( const auto &kv : reg )
    {
        if( !now.count( kv.first ) )
        {
            removed += " " + kv.first;
        }
    }
    say( msgLevel::warning ) << proclistPath() << " differs from what is synced with systemd"
                             << ( added.empty() ? "" : "; not registered:" + added )
                             << ( removed.empty() ? "" : "; registered, no longer in the proclist:" + removed )
                             << ( changed.empty() ? ""
                                                  : "; autostart (#!), watchdog (-) or executable changed:" + changed );
    say( msgLevel::warning ) << m_prog << " sync applies it, by a user who can edit the proclist";
}

inline int xCtrl::updateProclist( const std::string &name, proclistEdit edit, const std::string &newLine )
{
    std::string text;
    {
        std::ifstream f( proclistPath() );
        if( !f )
        {
            say( msgLevel::error ) << "cannot read " << proclistPath();
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
        say( msgLevel::info ) << "[dry-run] would edit " << changed << " line(s) for " << name << " in "
                              << proclistPath();
        return changed;
    }

    // in place, not by rename, so the file keeps its owner and permissions
    {
        std::ofstream f( proclistPath(), std::ios::trunc );
        f << edited;
        if( !f.good() )
        {
            say( msgLevel::error ) << "writing " << proclistPath() << ": " << strerror( errno );
            return -1;
        }
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
    std::vector<std::string> errors, warnings;
    const allowList         *allow = allowedDirectives();
    if( !allow )
    {
        say( msgLevel::error ) << m_allowError << "; " << app.name << " not registered";
        return -1;
    }
    std::vector<std::string> ignored; // magaox.conf's, reported by sync
    bool                     ok = prepareApp( app, dirs, errors, warnings, *allow, roleDirectives( ignored, *allow ) );
    for( const auto &w : warnings )
    {
        say( msgLevel::warning ) << w;
    }
    if( !ok )
    {
        for( const auto &e : errors )
        {
            say( msgLevel::error ) << e;
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

    int w = writeFile( unitDir() + "/" + u.instance( app.name ) + ".d/override.conf",
                       dropIn( u, app, m_top + "/bin", dirs, server, tunnels ),
                       0644 );
    return w;
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
        if( removeDropIn( n ) == 0 && !m_dryRun )
        {
            say( msgLevel::success ) << "unregistered " << n;
        }
    }
    return rc;
}

/// The realtime clock minus the monotonic clock, seconds: converts systemd's monotonic timestamps
inline double monotonicOffset()
{
    timespec rt, mono;
    clock_gettime( CLOCK_REALTIME, &rt );
    clock_gettime( CLOCK_MONOTONIC, &mono );
    return ( rt.tv_sec + rt.tv_nsec / 1e9 ) - ( mono.tv_sec + mono.tv_nsec / 1e9 );
}

inline void xCtrl::registeredOrdering( std::string &server, std::vector<std::string> &tunnels )
{
    std::vector<appEntry> all;
    for( const auto &n : registered() )
    {
        appEntry a;
        if( registryRead( a, n ) )
        {
            all.push_back( a );
        }
    }
    serverAndTunnels( all, server, tunnels );
}

inline settingsState xCtrl::checkSettings( const appEntry                     &app,
                                           const directiveMap                 &role,
                                           const std::string                  &server,
                                           const std::vector<std::string>     &tunnels,
                                           std::map<std::string, std::string> &p )
{
    unitNames                u( m_role );
    settingsState            st;
    directiveMap             dirs;
    std::vector<std::string> ignored; // reported by sync, not again here
    const allowList         *allow = allowedDirectives();
    if( !allow )
    {
        st.errors.push_back( m_allowError );
        return st;
    }
    if( !prepareApp( app, dirs, st.errors, ignored, *allow, role ) )
    {
        return st;
    }
    st.ok      = true;
    st.path    = unitDir() + "/" + u.instance( app.name ) + ".d/override.conf";
    st.content = dropIn( u, app, m_top + "/bin", dirs, server, tunnels );

    st.fileStale = !sameContent( st.path, st.content );
    st.loadStale = p["NeedDaemonReload"] == "yes";
    st.running   = p["ActiveState"] == "active" || p["ActiveState"] == "activating" || p["ActiveState"] == "reloading";
    st.wdWant    = dirs.count( "WatchdogSec" ) ? timespanSeconds( dirs["WatchdogSec"] ) : 0; // systemd's default
    if( !st.running )
    {
        return st;
    }

    // systemd fixes the watchdog when the app starts, so a run keeps the one it started with
    double have = p.count( "WatchdogUSec" ) ? timespanSeconds( p["WatchdogUSec"] ) : -1;
    if( have >= 0 && st.wdWant >= 0 && std::abs( have - st.wdWant ) > 1e-3 )
    {
        st.wdHave   = p["WatchdogUSec"];
        st.runStale = true;
    }

    // the drop-in, rewritten only when it changes, newer than the run: e.g. registered again while running
    struct stat fs;
    double      started = p.count( "ExecMainStartTimestampMonotonic" )
                              ? strtod( p["ExecMainStartTimestampMonotonic"].c_str(), nullptr )
                              : 0;
    if( started > 0 && stat( st.path.c_str(), &fs ) == 0 )
    {
        // the start is in monotonic microseconds, the file time on the realtime clock: compare realtime seconds
        double startRt = monotonicOffset() + started / 1e6;
        st.runStale    = st.runStale || fs.st_mtim.tv_sec + fs.st_mtim.tv_nsec / 1e9 > startRt;
    }
    return st;
}

inline void xCtrl::warnOutdatedRunning( const std::vector<std::string>                            &names,
                                        std::map<std::string, std::map<std::string, std::string>> &props )
{
    std::string              server;
    std::vector<std::string> tunnels, ignored;
    const allowList         *allow = allowedDirectives();
    if( !allow )
    {
        return; // without it, what sync would register is not known
    }
    checkConfValues( names );
    directiveMap role = roleDirectives( ignored, *allow );
    registeredOrdering( server, tunnels );

    for( const auto &n : names )
    {
        appEntry a;
        if( !registryRead( a, n ) )
        {
            continue;
        }
        settingsState st = checkSettings( a, role, server, tunnels, props[n] );
        if( !st.running )
        {
            continue; // a stopped app starts with whatever is registered then
        }
        if( !st.ok )
        {
            for( const auto &e : st.errors )
            {
                say( msgLevel::warning ) << e;
            }
        }
        else if( st.fileStale || st.loadStale )
        {
            say( msgLevel::warning ) << n << ": running with magaox.conf defaults or old [systemd] settings in " << n
                                     << ".conf; " << m_prog << " sync, then " << m_prog << " restart " << n
                                     << ", applies them";
        }
        else if( st.runStale )
        {
            char wd[64] = "";
            if( !st.wdHave.empty() )
            {
                snprintf( wd, sizeof( wd ), " (watchdog %s, now %gs)", st.wdHave.c_str(), st.wdWant );
            }
            say( msgLevel::warning ) << n << ": runs with settings registered before it started" << wd << "; " << m_prog
                                     << " restart " << n << " applies the new ones";
        }
    }
}

inline int xCtrl::refreshApps( const std::vector<appEntry> &apps )
{
    const allowList *allow = allowedDirectives();
    if( apps.empty() || !allow )
    {
        return 0; // without the allow list, what sync would register is not known
    }

    // the ordering in the drop-ins depends on all registered apps
    std::string              server;
    std::vector<std::string> tunnels, ignored, names;
    registeredOrdering( server, tunnels );
    for( const auto &a : apps )
    {
        names.push_back( a.name );
    }
    checkConfValues( names );
    directiveMap role  = roleDirectives( ignored, *allow );
    auto         props = showProps( names );

    for( const auto &a : apps )
    {
        settingsState st = checkSettings( a, role, server, tunnels, props[a.name] );
        if( !st.ok )
        {
            for( const auto &e : st.errors )
            {
                say( msgLevel::warning ) << e;
            }
            say( msgLevel::warning ) << a.name << " starts with its last synced settings";
        }
        else if( st.fileStale || st.loadStale )
        {
            say( msgLevel::warning ) << a.name << ": " << a.name
                                     << ".conf or magaox.conf changed since the last sync; it "
                                     << "starts with the synced settings. " << m_prog
                                     << " sync, by a user who can edit the proclist, applies them";
        }
        else if( st.running && !st.wdHave.empty() )
        {
            say( msgLevel::warning ) << a.name << " is running with settings synced before it started; " << m_prog
                                     << " restart " << a.name << " applies them";
        }
    }
    return 0;
}

inline int xCtrl::startApps( const std::vector<appEntry> &apps )
{
    unitNames                u( m_role );
    std::vector<std::string> tunnels{ "start" }, others{ "start" }, server{ "start" }, reset{ "reset-failed" }, names;
    for( const auto &a : apps )
    {
        ( a.isServer() ? server : a.isTunnel() ? tunnels : others ).push_back( u.instance( a.name ) );
        names.push_back( a.name );
    }

    // with its current configuration
    refreshApps( apps );

    // an explicit start also clears a failure, e.g. after the start limit was hit; only failed apps have one
    for( const auto &kv : showProps( names ) )
    {
        if( kv.second.count( "ActiveState" ) && kv.second.at( "ActiveState" ) == "failed" )
        {
            reset.push_back( u.instance( kv.first ) );
        }
    }
    if( reset.size() > 1 )
    {
        systemctl( reset );
    }

    // the INDI server first, so the apps connect to it as they start, after the tunnels its remote links go through;
    // systemctl start waits for each to report READY=1
    int rc = 0;
    if( tunnels.size() > 1 )
    {
        rc |= systemctl( tunnels );
    }
    if( server.size() > 1 )
    {
        rc |= systemctl( server );
    }
    if( others.size() > 1 )
    {
        rc |= systemctl( others );
    }
    return rc;
}

inline int xCtrl::stopApps( const std::vector<std::string> &names )
{
    unitNames                u( m_role );
    std::vector<std::string> others{ "stop" }, server{ "stop" }, tunnels{ "stop" };
    for( const auto &n : names )
    {
        appEntry a;
        // an app no longer registered: INDI servers are named is<ROLE>, e.g. isICC
        bool known    = registryRead( a, n );
        bool isServer = known ? a.isServer() : n.compare( 0, 2, "is" ) == 0;
        ( isServer ? server : known && a.isTunnel() ? tunnels : others ).push_back( u.instance( n ) );
    }

    // the reverse of startApps(): the INDI server after the apps, so their last messages get through, and the
    // tunnels its remote links go through last
    int rc = 0;
    if( others.size() > 1 )
    {
        rc |= systemctl( others );
    }
    if( server.size() > 1 )
    {
        rc |= systemctl( server );
    }
    if( tunnels.size() > 1 )
    {
        rc |= systemctl( tunnels );
    }
    return rc;
}

inline int xCtrl::writeFile( const std::string &path, const std::string &content, mode_t mode )
{
    // root writes nothing but this role's template, target and drop-ins
    if( !generatedPath( path ) )
    {
        say( msgLevel::error ) << "refusing to write " << path << ": not a file xctrl generates";
        return -1;
    }
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
        say( msgLevel::info ) << "[dry-run] would write " << path << ":\n" << content;
        return 1;
    }
    rootScope root;
    return writeIfChanged( path, content, mode );
}

inline int xCtrl::doSync( bool summary )
{
    unitNames                u( m_role );
    std::vector<std::string> errors, warnings;

    // nothing without the allow list: it decides which [systemd] directives root may write
    const allowList *allow = allowedDirectives();
    if( !allow )
    {
        say( msgLevel::error ) << m_allowError << "; nothing synced";
        return 1;
    }

    std::ifstream pf( proclistPath() );
    if( !pf )
    {
        say( msgLevel::error ) << "cannot read proclist " << proclistPath();
        return 1;
    }
    proclist pl = parseProclist( pf );
    errors      = pl.errors;
    warnings    = pl.warnings;

    // every role on this host first, warning about other roles' running apps, then this one
    auto roles = hostRoles();
    if( summary )
    {
        for( const auto &kv : roles )
        {
            if( kv.first != u.lrole )
            {
                const roleApps &r = kv.second;
                say( r.running.empty() ? msgLevel::info : msgLevel::warning )
                    << "role " << r.name << ": " << r.registered.size() << " app(s) registered, " << r.running.size()
                    << " running";
            }
        }
        say( msgLevel::info ) << "Current role: " << m_role << ": " << roles[u.lrole].registered.size()
                              << " app(s) registered, " << roles[u.lrole].running.size() << " running";
    }

    // allow list entries this systemd does not have
    const directiveSections &sections = systemdDirectives();
    for( const auto &d : *allow )
    {
        if( !sections.empty() && !sections.count( d ) )
        {
            warnings.push_back( allowListPath() + " allows " + d +
                                ", which does not exist on this version of systemd" );
        }
    }

    // systemd checks the values of magaox.conf and every app's .conf at once; then the role-wide settings
    std::vector<std::string> listedNames;
    for( const auto *list : { &pl.apps, &pl.disabled } )
    {
        for( const auto &a : *list )
        {
            listedNames.push_back( a.name );
        }
    }
    checkConfValues( listedNames );
    if( m_valuesUnchecked )
    {
        warnings.push_back( "systemd-analyze not found: [systemd] values are not checked, and systemd ignores any it "
                            "can not use when it loads the units" );
    }
    std::string  cfgDir = m_top + "/" + MAGAOX_configRelPath;
    directiveMap gdirs  = roleDirectives( warnings, *allow );

    // the INDI server and the apps must agree on the control FIFO
    confMap gconf, isconf;
    readConf( gconf, cfgDir + "/" + MAGAOX_globalConfig );
    if( readConf( isconf, cfgDir + "/is" + m_role + ".conf" ) && isconf.count( "indiserver.f" ) &&
        gconf.count( "indiserver_ctrl_fifo" ) && isconf["indiserver.f"] != gconf["indiserver_ctrl_fifo"] )
    {
        warnings.push_back( "[indiserver] f = " + isconf["indiserver.f"] + " in is" + m_role +
                            ".conf differs from indiserver_ctrl_fifo = " + gconf["indiserver_ctrl_fifo"] +
                            " in magaox.conf" );
    }

    struct valid
    {
        appEntry     app;
        directiveMap dirs;
        bool         autostart;
    };
    std::vector<valid> apps;

    // names in the proclist that keep any registration they have, though they could not be (re)registered now,
    // and those of them that are disabled (#!), so lose their autostart
    std::set<std::string>    keep;
    std::vector<std::string> keepOff;

    // active lines: registered and started automatically; invalid ones keep their earlier registration
    for( const auto &app : pl.apps )
    {
        directiveMap d;
        bool         hasConf = false;
        size_t       n       = errors.size();
        if( !prepareApp( app, d, errors, warnings, *allow, gdirs, &hasConf ) )
        {
            errors.insert( errors.begin() + n,
                           "proclist line " + std::to_string( app.line ) + ": " + app.name + " not (re)registered:" );
            keep.insert( app.name );
            continue;
        }
        if( !hasConf )
        {
            say( msgLevel::note ) << app.name << " has no " << app.name << ".conf, using defaults";
        }
        apps.push_back( { app, d, true } );
    }

    // disabled lines, #!: registered, to start by hand, but not started automatically; they need a .conf
    for( const auto &app : pl.disabled )
    {
        if( access( ( cfgDir + "/" + app.name + ".conf" ).c_str(), R_OK ) != 0 )
        {
            warnings.push_back( "proclist line " + std::to_string( app.line ) + ": disabled " + app.name + " has no " +
                                app.name + ".conf, not registered; use # for comments" );
            continue;
        }
        directiveMap d;
        size_t       n = errors.size();
        if( !prepareApp( app, d, errors, warnings, *allow, gdirs ) )
        {
            errors.insert( errors.begin() + n,
                           "proclist line " + std::to_string( app.line ) + ": " + app.name + " not (re)registered:" );
            keep.insert( app.name );
            keepOff.push_back( app.name );
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

    // registered apps no longer in the proclist are unregistered, unless they are still running: those stay
    // registered, to be stopped by name, but lose their autostart, so neither startup nor a reboot starts them
    std::vector<std::string> before = registered(), removed, keptRunning, running = activeApps();
    std::set<std::string>    wasRegistered( before.begin(), before.end() ), listed( keep.begin(), keep.end() );
    std::set<std::string>    isRunning( running.begin(), running.end() );
    for( const auto &v : apps )
    {
        listed.insert( v.app.name );
    }
    for( const auto &n : before )
    {
        if( listed.count( n ) )
        {
            continue;
        }
        if( isRunning.count( n ) )
        {
            warnings.push_back( n + " removed from proclist but still running, so still registered; stop it first (" +
                                m_prog + " stop " + n + "), then " + m_prog + " sync unregisters it" );
            keptRunning.push_back( n );
            continue;
        }
        removed.push_back( n );
    }

    // write only what changed: the role's template and target, then each app's drop-in
    int  rc    = 0;
    bool wrote = false;
    for( const auto &f :
         { std::make_pair( u.templ(), templateUnit( u, envFile() ) ), std::make_pair( u.target(), targetUnit( u ) ) } )
    {
        int w = writeFile( unitDir() + "/" + f.first, f.second, 0644 );
        rc |= w < 0;
        wrote = wrote || w == 1;
    }

    // apps whose autostart is turned off, and why, to tell the user
    std::vector<std::string>                         added, changed, toEnable{ "enable" }, toDisable{ "disable" };
    std::vector<std::pair<std::string, std::string>> turnedOff;
    for( const auto &v : apps )
    {
        int w = writeFile( unitDir() + "/" + u.instance( v.app.name ) + ".d/override.conf",
                           dropIn( u, v.app, m_top + "/bin", v.dirs, server, tunnels ),
                           0644 );
        rc |= w < 0;
        wrote = wrote || w == 1;
        if( w == 1 )
        {
            ( wasRegistered.count( v.app.name ) ? changed : added ).push_back( v.app.name );
        }
        if( v.autostart != enabled( v.app.name ) )
        {
            ( v.autostart ? toEnable : toDisable ).push_back( u.instance( v.app.name ) );
            if( !v.autostart )
            {
                turnedOff.push_back( { v.app.name, "now #! disabled in proclist" } );
            }
        }
    }
    for( const auto &n : keepOff )
    {
        if( enabled( n ) )
        {
            toDisable.push_back( u.instance( n ) );
            turnedOff.push_back( { n, "now #! disabled in proclist" } );
        }
    }
    for( const auto &n : keptRunning )
    {
        if( enabled( n ) )
        {
            toDisable.push_back( u.instance( n ) );
            turnedOff.push_back( { n, "now removed from proclist" } );
        }
    }
    for( const auto &n : removed )
    {
        if( enabled( n ) )
        {
            toDisable.push_back( u.instance( n ) );
        }
    }
    if( rc )
    {
        say( msgLevel::error ) << "could not write the systemd configuration";
        return 1;
    }

    // links first, while systemd still has the removed apps' units, then the removals
    if( toDisable.size() > 1 )
    {
        rc |= systemctl( toDisable, true ) != 0;
    }
    for( const auto &n : removed )
    {
        rc |= removeDropIn( n ) != 0;
        wrote = true;
    }

    // a reload only if a unit file changed, or systemd still has older ones loaded
    bool stale = false;
    for( auto &kv : showProps( before ) )
    {
        stale = stale || kv.second["NeedDaemonReload"] == "yes";
    }
    if( wrote || stale )
    {
        rc |= systemctl( { "daemon-reload" } ) != 0;
    }
    if( toEnable.size() > 1 )
    {
        rc |= systemctl( toEnable, true ) != 0;
    }
    removeLegacy();

    auto list = []( const std::vector<std::string> &v )
    {
        std::string s;
        for( const auto &n : v )
        {
            s += " " + n;
        }
        return s;
    };
    size_t nauto = 0;
    for( const auto &v : apps )
    {
        nauto += v.autostart;
    }
    say( msgLevel::success ) << "role " << m_role << ": " << nauto << " app(s) for automatic startup, "
                             << apps.size() - nauto << " with autostart disabled"
                             << ( added.empty() ? "" : "; registered:" + list( added ) )
                             << ( changed.empty() ? "" : "; updated:" + list( changed ) )
                             << ( removed.empty() ? "" : "; unregistered:" + list( removed ) )
                             << ( wrote || stale || toEnable.size() > 1 || toDisable.size() > 1 ? ""
                                                                                                : "; nothing changed" );
    for( const auto &n : changed )
    {
        if( isRunning.count( n ) )
        {
            say( msgLevel::info ) << n << " is running with its old settings; " << m_prog << " restart " << n
                                  << " applies the new ones";
        }
    }
    for( const auto &off : turnedOff )
    {
        say( msgLevel::info ) << off.first << ": autostart turned off (" << off.second << ")"
                              << ( isRunning.count( off.first ) ? "; however the app is still running." : "" );
    }
    for( const auto &w : warnings )
    {
        say( msgLevel::warning ) << w;
    }
    for( const auto &e : errors )
    {
        say( msgLevel::error ) << e;
    }
    return ( rc || !errors.empty() ) ? 1 : 0;
}

inline int xCtrl::cmdSync()
{
    return doSync( true );
}

inline bool xCtrl::syncNeeded()
{
    // the proclist against the drop-ins and links, as warnIfProclistChanged() compares them
    std::ifstream         pf( proclistPath() );
    proclist              pl = parseProclist( pf );
    std::set<std::string> listed;
    for( const auto *list : { &pl.apps, &pl.disabled } )
    {
        for( const auto &a : *list )
        {
            appEntry reg;
            listed.insert( a.name );
            if( !registryRead( reg, a.name ) || reg.exec != a.exec || reg.prefix != a.prefix ||
                enabled( a.name ) != ( list == &pl.apps ) )
            {
                return true;
            }
        }
    }
    for( const auto &n : registered() )
    {
        if( !listed.count( n ) )
        {
            return true;
        }
    }

    // and each drop-in against what its .conf makes now
    const allowList *allow = allowedDirectives();
    if( !allow )
    {
        return true; // doSync() then says why it can not run
    }
    std::string              server;
    std::vector<std::string> tunnels, ignored;
    registeredOrdering( server, tunnels );
    checkConfValues( registered() );
    directiveMap                       role = roleDirectives( ignored, *allow );
    std::map<std::string, std::string> none;
    for( const auto &n : registered() )
    {
        appEntry a;
        if( registryRead( a, n ) && checkSettings( a, role, server, tunnels, none ).fileStale )
        {
            return true;
        }
    }
    return false;
}

inline void xCtrl::warnRemovedRunning()
{
    std::ifstream         pf( proclistPath() );
    proclist              pl = parseProclist( pf );
    std::set<std::string> listed;
    for( const auto *list : { &pl.apps, &pl.disabled } )
    {
        for( const auto &a : *list )
        {
            listed.insert( a.name );
        }
    }
    for( const auto &n : activeApps() )
    {
        if( !listed.count( n ) )
        {
            say( msgLevel::warning ) << n << " removed from proclist but still running; stop it first (" << m_prog
                                     << " stop " << n << "), then " << m_prog << " sync unregisters it";
        }
    }
}

inline int xCtrl::cmdStartup()
{
    // with names, as start
    if( !m_names.empty() )
    {
        return cmdStart();
    }

    // an editor's startup first syncs new and changed apps; without the allow list, nothing starts
    if( m_privileged && syncNeeded() && doSync( false ) != 0 && !allowedDirectives() )
    {
        return 1;
    }
    warnIfProclistChanged();
    warnOtherRoles();

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
        say( msgLevel::error ) << "no apps enabled for role " << m_role << "; " << m_prog
                               << " sync, by a user who can edit the proclist, registers them";
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
        say( msgLevel::error ) << "specify --all to stop all apps, or give app names";
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
    // an editor's start first syncs new and changed apps; without the allow list, nothing starts
    if( m_privileged && syncNeeded() && doSync( false ) != 0 && !allowedDirectives() )
    {
        return 1;
    }
    warnIfProclistChanged();
    warnOtherRoles();
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
    warnIfProclistChanged();

    if( m_all || m_names.empty() )
    {
        if( !m_all )
        {
            say( msgLevel::error ) << "restart needs app names, or --all";
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
    // the INDI server first, as in startApps()
    int rc = 0;
    if( server.size() > 1 )
    {
        rc |= systemctl( server );
    }
    if( others.size() > 1 )
    {
        rc |= systemctl( others );
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
    runCommand( argv, &out, true ); // read-only, as the caller

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
    warnIfProclistChanged();
    warnRemovedRunning();
    warnOtherRoles();
    warnOtherRolesBoot();
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
        say( msgLevel::info ) << "no apps registered for role " << m_role << "; run " << m_prog << " sync";
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
    // no watchdog: for a running app, systemd's is 0 or infinity; else its drop-in, what it starts with, sets none,
    // like systemd, whose default is none
    unitNames u( m_role );
    auto      noWatchdog = [&]( const std::string &n )
    {
        auto       &p  = props[n];
        std::string wd = "0";
        if( !pidOf( n ).empty() && p.count( "WatchdogUSec" ) )
        {
            wd = p["WatchdogUSec"];
        }
        else
        {
            std::ifstream f( unitDir() + "/" + u.instance( n ) + ".d/override.conf" );
            for( std::string l; std::getline( f, l ); )
            {
                wd = l.compare( 0, 12, "WatchdogSec=" ) == 0 ? l.substr( 12 ) : wd;
            }
        }
        double s = timespanSeconds( wd );
        return s <= 0 || s >= 1e18;
    };

    if( !m_verbose )
    {
        // name: state (pid: N)
        for( const auto &r : rows )
        {
            std::string pid = pidOf( r.name );
            std::cout << ( m_color ? r.st.color : "" ) << r.name << ": " << r.st.label
                      << ( pid.empty() ? "" : " (pid: " + pid + ")" )
                      << ( regs.count( r.name ) && noWatchdog( r.name ) ? ", no watchdog" : "" ) << reset << "\n";
        }
        warnOutdatedRunning( names, props );
        return 0;
    }

    // a number from systemctl show, or -1 if it is not set.  systemd shows an unset number, e.g. MemoryCurrent when
    // memory accounting is off, as the largest 64-bit unsigned value, 18446744073709551615; and words like "[not set]"
    // or "infinity" do not start with a digit.
    static const std::string unset = std::to_string( std::numeric_limits<uint64_t>::max() );
    auto                     num   = []( std::map<std::string, std::string> &p, const char *key ) -> double
    {
        if( !p.count( key ) || p[key].empty() || !isdigit( static_cast<unsigned char>( p[key][0] ) ) ||
            p[key] == unset )
        {
            return -1;
        }
        return strtod( p[key].c_str(), nullptr );
    };
    // now on the monotonic clock, in microseconds like systemd's ...Monotonic timestamps
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
                cpuPct[n] = 100 * ( c1 - c0 ) / 1e3 / dt; // CPU nanoseconds / 1e3 per elapsed microsecond, %
            }
        }
    }
    double now = monoUsec();

    // GPU memory, if this host has NVIDIA GPUs, summed over each app's processes
    std::map<std::string, double> gpuMiB;
    std::string                   smi;
    bool                          haveSmi = false;
    std::string                   path    = getenv( "PATH" ) ? getenv( "PATH" ) : "/usr/bin:/bin";
    // look for nvidia-smi in each directory of PATH, from b to the next ':' at e, as a shell would
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

    // NAME, STATE and PID are as wide as their longest entry, to fit small terminals
    int wName = 4, wState = 5, wPid = 3;
    for( const auto &r : rows )
    {
        wName  = std::max( wName, static_cast<int>( r.name.size() ) );
        wState = std::max( wState, static_cast<int>( r.st.label.size() ) );
        wPid   = std::max( wPid, static_cast<int>( pidOf( r.name ).size() ) );
    }

    char buf[512];
    snprintf( buf,
              sizeof( buf ),
              "%-*s %-*s %*s %-9s %8s %10s %9s %8s %7s %9s %9s\n",
              wName,
              "NAME",
              wState,
              "STATE",
              wPid,
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
        if( regs.count( r.name ) && noWatchdog( r.name ) )
        {
            wd = "none";
        }
        else if( p.count( "WatchdogUSec" ) )
        {
            wd = p["WatchdogUSec"];
        }
        char v[64];
        if( !pid.empty() )
        {
            double start = num( p, "ExecMainStartTimestampMonotonic" );
            if( start > 0 )
            {
                snprintf( v, sizeof( v ), "%.0f", ( now - start ) / 1e6 ); // microseconds to seconds
                up = v;
            }
            double last = num( p, "WatchdogTimestampMonotonic" );
            if( last > 0 && wd != "-" && wd != "none" )
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
                snprintf( v, sizeof( v ), "%.1f", m / 1048576 ); // bytes to MiB, 1024 * 1024
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
                  "%-*s %-*s %*s %-9s %8s %10s %9s %8s %7s %9s %9s",
                  wName,
                  r.name.c_str(),
                  wState,
                  r.st.label.c_str(),
                  wPid,
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
    warnOutdatedRunning( names, props );
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
            // as the caller: the journal is theirs to read with sudo or in group systemd-journal
            runCommand( { "journalctl", "-u", u.instance( n ), "-n", std::to_string( lines ), "--no-pager", "-q" },
                        nullptr,
                        true );
        }
        std::cout << "End " << n << " logs\n\n";
    }
    return 0;
}

inline int xCtrl::cmdInspect()
{
    if( m_names.size() != 1 )
    {
        say( msgLevel::error ) << "inspect needs one app name";
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
                         "-f" },
                       nullptr,
                       true ); // as the caller: sudo, or group systemd-journal, reads the journal
}

inline int xCtrl::cmdEnable()
{
    if( namesOrAll() < 0 )
    {
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
            say( msgLevel::error ) << n << " is neither registered nor in " << proclistPath();
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

        // an app disabled without a .conf, or added since the last sync, is registered now
        appEntry reg;
        if( enable && !registryRead( reg, n ) )
        {
            int w = registerApp( app );
            if( w < 0 )
            {
                say( msgLevel::error ) << n << " is enabled in the proclist but could not be registered";
                rc = 1;
                continue;
            }
            reload = reload || w == 1;
        }
        if( registryRead( reg, n ) )
        {
            args.push_back( u.instance( n ) );
        }
        say( msgLevel::success ) << n << ( enable ? ": autostart on" : ": autostart off" )
                                 << ( e > 0
                                          ? ( enable ? ", #! removed in " : ", disabled with #! in " ) + proclistPath()
                                          : "" );
    }

    if( reload )
    {
        rc |= systemctl( { "daemon-reload" } ) != 0;
    }
    if( args.size() > 1 )
    {
        rc |= systemctl( args, true ) != 0;
    }

    say( msgLevel::note ) << m_verb << " does not "
                          << ( enable ? "start apps: " + m_prog + " start NAME, or the next startup, does"
                                      : "stop apps: " + m_prog + " stop NAME does" );
    return rc;
}

inline int xCtrl::cmdDelete()
{
    if( namesOrAll() < 0 )
    {
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
            say( msgLevel::error ) << n << " is neither registered nor in " << proclistPath();
            rc = 1;
            continue;
        }
        if( e > 0 )
        {
            say( msgLevel::success ) << "removed " << n << " from " << proclistPath();
        }
    }
    if( !regs.empty() )
    {
        rc |= deregister( regs );
        rc |= systemctl( { "daemon-reload" } );
    }
    return rc;
}

inline int xCtrl::cmdAdd()
{
    if( m_names.size() != 2 )
    {
        say( msgLevel::error ) << "add needs NAME EXECUTABLE, or -- -NAME EXECUTABLE for an app without a watchdog";
        return 1;
    }
    unitNames u( m_role );
    appEntry  app;
    app.name = m_names[0];
    app.exec = m_names[1];
    if( app.name[0] == '-' )
    {
        app.prefix = "-";
        app.name.erase( 0, 1 );
    }
    if( !validName( app.name ) || !validName( app.exec ) )
    {
        say( msgLevel::error ) << "invalid app name or executable [" << m_names[0] << " " << m_names[1] << "]";
        return 1;
    }

    std::ifstream pf( proclistPath() );
    if( !pf )
    {
        say( msgLevel::error ) << "cannot read proclist " << proclistPath();
        return 1;
    }
    proclist pl = parseProclist( pf );
    for( const auto *list : { &pl.apps, &pl.disabled } )
    {
        for( const auto &a : *list )
        {
            if( a.name == app.name )
            {
                say( msgLevel::error ) << app.name << " is already in " << proclistPath() << ", line " << a.line
                                       << "; use enable or disable";
                return 1;
            }
        }
    }

    // an added app has its own configuration
    std::string conf = m_top + "/" + MAGAOX_configRelPath + "/" + app.name + ".conf";
    if( access( conf.c_str(), R_OK ) != 0 )
    {
        say( msgLevel::error ) << conf << " not found; " << app.name << " not added";
        return 1;
    }

    // one INDI server per role
    for( const auto &n : registered() )
    {
        appEntry a;
        if( app.isServer() && n != app.name && registryRead( a, n ) && a.isServer() )
        {
            say( msgLevel::error ) << n << " is the role's INDI server; " << app.name << " not added";
            return 1;
        }
    }

    // check the allow list, the executable and the settings before changing the proclist
    directiveMap             dirs;
    std::vector<std::string> errors, ignored; // registerApp() reports the ignored directives
    const allowList         *allow = allowedDirectives();
    if( !allow )
    {
        say( msgLevel::error ) << m_allowError << "; " << app.name << " not added";
        return 1;
    }
    if( !prepareApp( app, dirs, errors, ignored, *allow, roleDirectives( ignored, *allow ) ) )
    {
        for( const auto &e : errors )
        {
            say( msgLevel::error ) << e;
        }
        say( msgLevel::error ) << app.name << " not added";
        return 1;
    }

    int w = -1;
    if( updateProclist( app.name, proclistEdit::append, "#!" + app.prefix + app.name + " " + app.exec ) < 0 ||
        ( w = registerApp( app ) ) < 0 )
    {
        return 1;
    }
    int rc = w == 1 ? systemctl( { "daemon-reload" } ) != 0 : 0;
    if( enabled( app.name ) )
    {
        rc |= systemctl( { "disable", u.instance( app.name ) }, true ) != 0;
    }
    say( msgLevel::success ) << "added #!" << app.prefix << app.name << " " << app.exec << " to " << proclistPath()
                             << " and registered it, autostart off: " << m_prog << " start " << app.name << " runs it, "
                             << m_prog << " enable " << app.name << " starts it automatically";
    return rc;
}

inline int xCtrl::cmdBoot()
{
    if( m_names.size() != 1 || ( m_names[0] != "on" && m_names[0] != "off" ) )
    {
        say( msgLevel::error ) << "boot needs on or off";
        return 1;
    }
    unitNames u( m_role );
    return systemctl( { m_names[0] == "on" ? "enable" : "disable", u.target() }, true );
}

inline int xCtrl::main( int argc, char **argv )
{
    // also installed as xctrl
    std::string invoked = argc > 0 ? argv[0] : "xctrl";
    m_prog              = invoked.substr( invoked.find_last_of( '/' ) + 1 ); // without its directory

    if( parseArgs( argc, argv ) < 0 )
    {
        return 2;
    }
    if( m_verb == "help" )
    {
        usage( std::cout );
        return 0;
    }

    static const std::map<std::string, int ( xCtrl::* )()> verbs = { { "sync", &xCtrl::cmdSync },
                                                                     { "add", &xCtrl::cmdAdd },
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
                                                                     { "boot", &xCtrl::cmdBoot } };
    auto                                                   it    = verbs.find( m_verb );
    if( m_verb == "register" || m_verb == "deregister" )
    {
        say( msgLevel::error ) << m_verb << " is now sync: " << m_prog << " sync registers new and changed apps and "
                               << "unregisters removed ones";
        return 2;
    }
    if( it == verbs.end() )
    {
        say( msgLevel::error ) << "unknown command " << m_verb << "; try --help";
        return 2;
    }

    // what we create under /etc stays readable by everyone, whatever the caller's umask
    umask( 022 );

    // decided before settlePrivileges() clears the environment
    xlog().colors( isatty( STDOUT_FILENO ) && getenv( "NO_COLOR" ) == nullptr,
                   isatty( STDERR_FILENO ) && getenv( "NO_COLOR" ) == nullptr );
    m_color = isatty( STDOUT_FILENO ) && getenv( "NO_COLOR" ) == nullptr;

    if( settlePrivileges() < 0 )
    {
        return 1;
    }

    return ( this->*( it->second ) )();
}

} // namespace xctrl

#endif // xctrl_hpp
