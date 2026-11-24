/** \file xctrl_test.cpp
 * \brief Catch2 tests for xctrl, the systemd process control utility
 *
 * \ingroup xctrl_unit_test
 */

#include "../../../tests/catch2/catch.hpp"

#include "../xctrl.hpp"

/** \defgroup xctrl_unit_test xctrl Unit Tests
 * \brief Unit tests for the xctrl utility.
 *
 * \ingroup application_unit_test
 */

namespace libXWCTest
{

/// Namespace for `xctrl` unit tests.
/** \ingroup xctrl_unit_test
 */
namespace xctrlTest
{

using namespace xctrl;

/// Proclist lines become apps with their prefixes; comments are skipped and bad lines reported.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl parses proclists", "[xctrl]" )
{
    std::istringstream in( "# comment\n"
                           "isICC  xindiserver\n"
                           "\n"
                           "  camsci1-dark   shmimIntegrator\n"
                           "-quiet  someExec\n"
                           "-camsci-1  someExec\n"
                           "#!off  someExec\n"
                           "#commented  someExec\n"
                           "name exec extra\n"
                           "x:bad  someExec\n"
                           "camsci1-dark  shmimIntegrator\n"
                           "ok  ../evil\n" );
    proclist           pl = parseProclist( in );

    REQUIRE( pl.apps.size() == 4 );
    REQUIRE( pl.apps[0].name == "isICC" );
    REQUIRE( pl.apps[0].isServer() );
    REQUIRE( pl.apps[1].name == "camsci1-dark" );
    REQUIRE( pl.apps[1].line == 4 );
    REQUIRE( pl.apps[1].notifies() );
    REQUIRE( pl.apps[2].name == "quiet" );
    REQUIRE( !pl.apps[2].notifies() );
    REQUIRE( pl.apps[3].name == "camsci-1" );
    REQUIRE( pl.apps[3].prefix == "-" );
    REQUIRE( !pl.apps[3].notifies() );

    // three tokens, unknown prefix, duplicate, path in the executable
    REQUIRE( pl.errors.size() == 4 );

    // the disabled process; a # line is a comment
    REQUIRE( pl.disabled.size() == 1 );
    REQUIRE( pl.disabled[0].name == "off" );
}

/// #! makes a disabled process, any other # line a comment; #! lines that are not processes are reported.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl keeps disabled proclist processes", "[xctrl]" )
{
    std::istringstream in( "##process-ID   Executable\n"
                           "# Processes for the MagAO-X ICC\n"
                           "##### camflowfs\n"
                           "#!camsci1  picamCtrl\n"
                           "#!  camsci2  picamCtrl\n"
                           "  #! camsci4 picamCtrl\n"
                           "#!-quietapp  someExec\n"
                           "#camsci_old  camsci\n"
                           "#-camsci_old2  camsci\n"
                           "# camsci_old3  camsci\n"
                           "#!Processes for the MagAO-X ICC\n"
                           "#!==========  ==========\n"
                           "camsci3  picamCtrl\n"
                           "#!camsci3  picamCtrl\n"
                           "#!camsci1  otherExec\n" );
    proclist           pl = parseProclist( in );

    REQUIRE( pl.errors.empty() );
    REQUIRE( pl.apps.size() == 1 );
    REQUIRE( pl.apps[0].name == "camsci3" );

    // #!name and #! name are the same; an active line and the first disabled line win; # lines are comments
    std::vector<std::string> names;
    for( const auto &d : pl.disabled )
    {
        names.push_back( d.name );
    }
    REQUIRE( names == std::vector<std::string>{ "camsci1", "camsci2", "camsci4", "quietapp" } );
    REQUIRE( pl.disabled[1].exec == "picamCtrl" );
    REQUIRE( pl.disabled[3].prefix == "-" );

    // #! lines that are not processes
    REQUIRE( pl.warnings.size() == 2 );
}

/// systemd time spans, as systemctl show prints them, convert to seconds.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl reads systemd time spans", "[xctrl]" )
{
    REQUIRE( timespanSeconds( "10s" ) == 10 );
    REQUIRE( timespanSeconds( "20" ) == 20 );
    REQUIRE( timespanSeconds( "2.5" ) == 2.5 );
    REQUIRE( timespanSeconds( "1min 30s" ) == 90 );
    REQUIRE( timespanSeconds( "500ms" ) == 0.5 );
    REQUIRE( timespanSeconds( "2h" ) == 7200 );
    REQUIRE( timespanSeconds( "0" ) == 0 );
    REQUIRE( timespanSeconds( "infinity" ) > 1e17 );
    REQUIRE( timespanSeconds( "" ) == -1 );
    REQUIRE( timespanSeconds( "ten" ) == -1 );
}

/// Configuration files parse into section.key values with quotes and arrays unwrapped.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl reads MagAO-X and TOML configuration", "[xctrl]" )
{
    std::istringstream in( "top = 1\n"
                           "[systemd]\n"
                           "timeouts = [20, 30]  # comment\n"
                           "killSignal = \"SIGINT\"\n"
                           "restart='on-failure'\n" );
    confMap            c = parseConf( in );

    REQUIRE( c["top"] == "1" );
    REQUIRE( c["systemd.timeouts"] == "20, 30" );
    REQUIRE( c["systemd.killSignal"] == "SIGINT" );
    REQUIRE( c["systemd.restart"] == "on-failure" );
}

/// App [systemd] settings become systemd directives, if allowed; any other is ignored with a warning.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl maps app configuration to systemd directives", "[xctrl]" )
{
    appEntry app;
    app.name = "cam";
    app.exec = "camCtrl";
    std::vector<std::string> warnings;

    // an allow list as /etc/magaox/systemd-directives.txt has it: comments, and a name that is not one, skipped
    std::istringstream al( "# the checked keys\nWatchdogSec\nTimeoutStartSec\nTimeoutStopSec\nRestartSec\nRestart\n"
                           "StartLimitBurst\nKillSignal\n\nRuntimeMaxSec   # stop after\nOnFailure\nFooSec\n"
                           "ExecStartPre\nUser\nbad-name\n" );
    allowList          allow = parseAllowList( al );
    REQUIRE( allow.size() == 12 );
    REQUIRE( allow.count( "RuntimeMaxSec" ) );
    REQUIRE( !allow.count( "bad-name" ) );

    SECTION( "[systemd] settings become directives" )
    {
        directiveMap d = appDirectives( app,
                                        { { "systemd.watchdogSec", "20" },
                                          { "systemd.timeoutStartSec", "90" },
                                          { "systemd.timeoutStopSec", "15" },
                                          { "systemd.restartSec", "2" },
                                          { "systemd.restart", "on-failure" },
                                          { "systemd.startLimitBurst", "3" } },
                                        warnings,
                                        allow );
        REQUIRE( warnings.empty() );
        REQUIRE( d["WatchdogSec"] == "20" );
        REQUIRE( d["TimeoutStartSec"] == "90" );
        REQUIRE( d["TimeoutStopSec"] == "15" );
        REQUIRE( d["RestartSec"] == "2" );
        REQUIRE( d["Restart"] == "on-failure" );
        REQUIRE( d["StartLimitBurst"] == "3" );
    }

    SECTION( "bad values and keys not allowed are ignored, with warnings" )
    {
        directiveMap d = appDirectives( app,
                                        { { "systemd.restart", "sometimes" },
                                          { "systemd.killSignal", "SIGSEGV" },
                                          { "systemd.timeoutStopSec", "5\nExecStartPre=/bin/evil" },
                                          { "systemd.nonsense", "1" } },
                                        warnings,
                                        allow );
        REQUIRE( d.empty() );
        REQUIRE( warnings.size() == 4 );
        for( const auto &w : warnings )
        {
            REQUIRE( w.find( "; ignoring directive" ) != std::string::npos );
        }
    }

    SECTION( "allowed directives systemd has, except denied ones" )
    {
        // a few lines of systemd --dump-configuration-items
        directiveSections sections = parseDirectiveDump( "[Unit]\nOnFailure=UNIT [...]\nStartLimitBurst=UNSIGNED\n"
                                                         "[Service]\nRuntimeMaxSec=SECONDS\nStartLimitBurst=UNSIGNED\n"
                                                         "ExecStartPre=PATH [ARGUMENT [...]]\nUser=STRING\nNice=NICE\n"
                                                         "[Socket]\nListenStream=SOCKET [...]\n" );
        REQUIRE( sections.size() == 6 );
        REQUIRE( sections["OnFailure"] );
        REQUIRE( sections["StartLimitBurst"] ); // [Unit], listed first
        REQUIRE( !sections["RuntimeMaxSec"] );
        REQUIRE( !sections.count( "ListenStream" ) );

        directiveMap d;
        systemdSettings( d,
                         warnings,
                         { { "systemd.runtimeMaxSec", "20" },
                           { "systemd.OnFailure", "notify@%n.service" },
                           { "systemd.WatchdogSec", "30" },
                           { "systemd.runtimemaxsec", "20" },
                           { "systemd.execStartPre", "+/bin/sh" },
                           { "systemd.user", "root" },
                           { "systemd.fooSec", "5" },
                           { "systemd.nice", "5" },
                           { "systemd.watchdogSec", "x" } },
                         "cam",
                         allow,
                         sections );
        REQUIRE( d["RuntimeMaxSec"] == "20" );
        REQUIRE( d["OnFailure"] == "notify@%n.service" );
        REQUIRE( d["WatchdogSec"] == "30" );
        REQUIRE( d.size() == 3 );

        // a typo and Nice not in the allow list, FooSec not in systemd, two denied, WatchdogSec spelled twice
        REQUIRE( warnings.size() == 6 );
        auto has = [&]( const std::string &text )
        {
            return std::any_of( warnings.begin(),
                                warnings.end(),
                                [&]( const std::string &w ) { return w.find( text ) != std::string::npos; } );
        };
        REQUIRE(
            has( "[systemd] nice directive for cam not in systemd-directives.txt allow list; ignoring directive" ) );
        REQUIRE(
            has( "[systemd] fooSec directive for cam does not exist on this version of systemd; ignoring directive" ) );
        REQUIRE( has( "[systemd] user directive for cam can not be set" ) );

        // written in their sections, after the checked settings
        std::ostringstream unit, service;
        appendDirectives( unit, d, true, sections );
        appendDirectives(
            service, directiveMap{ { "WatchdogSec", "30" }, { "RuntimeMaxSec", "20" } }, false, sections );
        REQUIRE( unit.str() == "OnFailure=notify@%n.service\n" );
        REQUIRE( service.str() == "WatchdogSec=30\nRuntimeMaxSec=20\n" );
    }

    SECTION( "apps that do not notify are simple services without a watchdog" )
    {
        app.prefix     = "-";
        directiveMap d = appDirectives( app, { { "systemd.watchdogSec", "20" } }, warnings, allow );
        REQUIRE( d["Type"] == "simple" );
        REQUIRE( d["WatchdogSec"] == "0" );
    }
}

/// Values are validated by kind.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl validates setting values", "[xctrl]" )
{
    REQUIRE( validValue( valueKind::seconds, "10" ) );
    REQUIRE( validValue( valueKind::seconds, "2.5" ) );
    REQUIRE( validValue( valueKind::seconds, "infinity" ) );
    REQUIRE( !validValue( valueKind::seconds, "" ) );
    REQUIRE( !validValue( valueKind::seconds, ".5" ) );
    REQUIRE( !validValue( valueKind::seconds, "10s" ) );
    REQUIRE( validValue( valueKind::count, "10" ) );
    REQUIRE( !validValue( valueKind::count, "-1" ) );
    REQUIRE( validValue( valueKind::restart, "on-watchdog" ) );
    REQUIRE( validValue( valueKind::signal, "SIGINT" ) );
    REQUIRE( !validValue( valueKind::signal, "SIGSEGV" ) );
}

/// Unit names use the role in lower case and map back to app names.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl names units by role", "[xctrl]" )
{
    unitNames u( "ICC" );
    REQUIRE( u.templ() == "magaox-icc-app@.service" );
    REQUIRE( u.target() == "magaox-icc.target" );
    REQUIRE( u.instance( "camsci1-dark" ) == "magaox-icc-app@camsci1-dark.service" );
    REQUIRE( u.appName( "magaox-icc-app@camsci1-dark.service" ) == "camsci1-dark" );
    REQUIRE( u.appName( "magaox-rtc-app@camwfs.service" ) == "" );
    REQUIRE( u.appName( "magaox-icc-app@.service" ) == "" );
}

/// Generated units carry the defaults, the app command and the INDI server ordering.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl generates the template and drop-ins", "[xctrl]" )
{
    unitNames    u( "ICC" );
    directiveMap defs  = defaultDirectives();
    defs["RestartSec"] = "3";

    std::string t = templateUnit( u, defs, "/etc/magaox-icc/environment" );
    REQUIRE( t.find( "Type=notify\n" ) != std::string::npos );
    REQUIRE( t.find( "RestartSec=3\n" ) != std::string::npos );
    REQUIRE( t.find( "User=xsup\n" ) != std::string::npos );
    REQUIRE( t.find( "WantedBy=magaox-icc.target\n" ) != std::string::npos );
    REQUIRE( t.find( "Environment=MAGAOX_ROLE=ICC\n" ) != std::string::npos );
    REQUIRE( t.find( "Environment=PYTHONUNBUFFERED=1\n" ) != std::string::npos );
    // StartLimit* belong in [Unit]
    REQUIRE( t.find( "StartLimitBurst" ) < t.find( "[Service]" ) );

    appEntry                 server{ "isICC", "xindiserver", "", 1 };
    appEntry                 tunnel{ "ICC_RTC", "sshDigger", "", 2 };
    appEntry                 cam{ "camsci1", "picamCtrl", "", 3 };
    std::vector<std::string> tunnels{ "ICC_RTC" };

    std::string ds = dropIn( u, server, "/opt/MagAOX/bin", {}, "isICC", tunnels );
    REQUIRE( ds.find( "After=magaox-icc-app@ICC_RTC.service\n" ) != std::string::npos );
    REQUIRE( ds.find( "ExecStart=/opt/MagAOX/bin/xindiserver -n isICC\n" ) != std::string::npos );

    std::string dc = dropIn( u, cam, "/opt/MagAOX/bin", { { "WatchdogSec", "20" } }, "isICC", tunnels );
    REQUIRE( dc.find( "After=magaox-icc-app@isICC.service\n" ) != std::string::npos );
    REQUIRE( dc.find( "ExecStart=\nExecStart=/opt/MagAOX/bin/picamCtrl -n camsci1\n" ) != std::string::npos );
    REQUIRE( dc.find( "WatchdogSec=20\n" ) != std::string::npos );

    std::string dt = dropIn( u, tunnel, "/opt/MagAOX/bin", {}, "isICC", tunnels );
    REQUIRE( dt.find( "After=" ) == std::string::npos );
}

/// Service states are reported with ranks and colors, and apps run by hand stand out.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl classifies app states", "[xctrl]" )
{
    typedef std::map<std::string, std::string> props;
    props                                      running{ { "ActiveState", "active" }, { "SubState", "running" } };
    props stopped{ { "ActiveState", "inactive" }, { "SubState", "dead" }, { "Result", "success" } };

    appState autostarted = classify( running, true, true );
    REQUIRE( autostarted.rank == 0 );
    REQUIRE( autostarted.label == "running" );
    REQUIRE( std::string( autostarted.color ) == "\033[32m" ); // green

    appState manual = classify( running, true, false );
    REQUIRE( manual.rank == 1 );
    REQUIRE( manual.label == "manually started" );
    REQUIRE( std::string( manual.color ) == "\033[34m" ); // blue

    appState disabled = classify( stopped, true, false );
    REQUIRE( disabled.rank == 2 );
    REQUIRE( disabled.label == "disabled" );
    REQUIRE( std::string( disabled.color ) == "\033[33m" ); // yellow

    appState notStarted = classify( stopped, true, true );
    REQUIRE( notStarted.rank == 3 );
    REQUIRE( notStarted.label == "not started" );

    REQUIRE( classify( props{ { "ActiveState", "activating" } }, true, true ).label == "starting" );

    appState dead = classify( props{ { "ActiveState", "failed" }, { "Result", "watchdog" } }, true, false );
    REQUIRE( dead.rank == 5 );
    REQUIRE( dead.label == "dead (watchdog)" );
    REQUIRE( std::string( dead.color ) == "\033[31m" ); // red

    REQUIRE( classify( props{ { "ActiveState", "inactive" }, { "Result", "exit-code" } }, true, true ).rank == 5 );
    REQUIRE( classify( running, false, false ).label == "running, not registered" );
    REQUIRE( classify( props{}, true, true ).label == "unknown" );
}

/// Adding, enabling, disabling and deleting edit only the app's proclist lines.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl edits an app's proclist lines", "[xctrl]" )
{
    std::string text    = "# comment camsci1\n"
                          "isICC        xindiserver\n"
                          "camsci1      picamCtrl\n"
                          "#!camsci2    picamCtrl\n"
                          "#! -quietapp someExec\n"
                          "#camsci5     picamCtrl\n"
                          "camsci10     picamCtrl\n";
    int         changed = 0;

    SECTION( "disable marks the active line #!" )
    {
        std::string t = editProclist( text, "camsci1", proclistEdit::comment, "", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t.find( "#!camsci1      picamCtrl\n" ) != std::string::npos );
        REQUIRE( t.find( "\ncamsci10     picamCtrl\n" ) != std::string::npos );
        REQUIRE( t.find( "# comment camsci1\n" ) == 0 );
    }

    SECTION( "enable removes #!, with or without a space after it, keeping the prefix" )
    {
        std::string t = editProclist( text, "camsci2", proclistEdit::uncomment, "camsci2 picamCtrl", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t.find( "\ncamsci2    picamCtrl\n" ) != std::string::npos );

        t = editProclist( text, "quietapp", proclistEdit::uncomment, "-quietapp someExec", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t.find( "\n-quietapp someExec\n" ) != std::string::npos );
    }

    SECTION( "enable of an active app changes nothing, enable of an unlisted app appends it" )
    {
        REQUIRE( editProclist( text, "camsci1", proclistEdit::uncomment, "camsci1 picamCtrl", changed ) == text );
        REQUIRE( changed == 0 );

        std::string t = editProclist( text, "newapp", proclistEdit::uncomment, "newapp someExec", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t == text + "newapp someExec\n" );
    }

    SECTION( "# lines are only comments: never enabled, disabled or deleted" )
    {
        std::string withOld = text + "#camsci_old  camsci\n# camsci1  picamCtrl\n";

        REQUIRE( editProclist( withOld, "camsci5", proclistEdit::append, "#!camsci5 picamCtrl", changed ) ==
                 withOld + "#!camsci5 picamCtrl\n" );

        std::string t = editProclist( withOld, "camsci_old", proclistEdit::uncomment, "camsci_old camsci", changed );
        REQUIRE( changed == 1 ); // appended as a new line, the # comment stays
        REQUIRE( t.find( "#camsci_old  camsci\n" ) != std::string::npos );
        REQUIRE( t.find( "\ncamsci_old camsci\n" ) != std::string::npos );

        t = editProclist( withOld, "camsci1", proclistEdit::remove, "", changed );
        REQUIRE( changed == 1 ); // only the active line
        REQUIRE( t.find( "# camsci1  picamCtrl\n" ) != std::string::npos );

        std::istringstream in( withOld );
        proclist           pl = parseProclist( in );
        for( const auto &d : pl.disabled )
        {
            REQUIRE( d.name != "camsci_old" );
        }
    }

    SECTION( "delete removes active and disabled lines of that app only" )
    {
        std::string t = editProclist( text + "#!camsci1 picamCtrl\n", "camsci1", proclistEdit::remove, "", changed );
        REQUIRE( changed == 2 );
        REQUIRE( t.find( "camsci1 " ) == std::string::npos );
        REQUIRE( t.find( "camsci10" ) != std::string::npos );
        REQUIRE( t.find( "# comment camsci1" ) != std::string::npos );
    }

    SECTION( "add appends a disabled line only for an app with no line" )
    {
        std::string t = editProclist( text, "maths_CO2", proclistEdit::append, "#!maths_CO2 magAOXMaths", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t == text + "#!maths_CO2 magAOXMaths\n" );

        REQUIRE( editProclist( text, "camsci2", proclistEdit::append, "#!camsci2 picamCtrl", changed ) == text );
        REQUIRE( changed == 0 );
        REQUIRE( editProclist( text, "camsci1", proclistEdit::append, "#!camsci1 picamCtrl", changed ) == text );
        REQUIRE( changed == 0 );
    }
}

/// Files are only rewritten when their contents change.
/**
 * \ingroup xctrl_unit_test
 */
TEST_CASE( "xctrl writes files only when they change", "[xctrl]" )
{
    std::string dir  = std::filesystem::temp_directory_path() / ( "xctrl_test_" + std::to_string( getpid() ) );
    std::string path = dir + "/sub/file.conf";

    REQUIRE( writeIfChanged( path, "a\n", 0644 ) == 1 );
    REQUIRE( writeIfChanged( path, "a\n", 0644 ) == 0 );
    REQUIRE( writeIfChanged( path, "b\n", 0644 ) == 1 );

    std::filesystem::remove_all( dir );
}

} // namespace xctrlTest
} // namespace libXWCTest
