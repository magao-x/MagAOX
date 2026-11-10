/** \file resuctrl_test.cpp
 * \brief Catch2 tests for resuctrl, the systemd process control utility
 *
 * \ingroup resuctrl_unit_test
 */

#include "../../../tests/catch2/catch.hpp"

#include "../resuctrl.hpp"

/** \defgroup resuctrl_unit_test resuctrl Unit Tests
 * \brief Unit tests for the resuctrl utility.
 *
 * \ingroup application_unit_test
 */

namespace libXWCTest
{

/// Namespace for `resuctrl` unit tests.
/** \ingroup resuctrl_unit_test
 */
namespace resuctrlTest
{

using namespace resuctrl;

/// Proclist lines become apps with their prefixes; comments are skipped and bad lines reported.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl parses proclists like resurrector_indi", "[resuctrl]" )
{
    std::istringstream in( "# comment\n"
                           "isICC  xindiserver\n"
                           "\n"
                           "  camsci1-dark   shmimIntegrator\n"
                           "py:pyapp  pyExec\n"
                           "-quiet  someExec\n"
                           "nhb:quiet2  someExec\n"
                           "#off  someExec\n"
                           "name exec extra\n"
                           "x:bad  someExec\n"
                           "camsci1-dark  shmimIntegrator\n"
                           "ok  ../evil\n" );
    proclist           pl = parseProclist( in );

    REQUIRE( pl.apps.size() == 5 );
    REQUIRE( pl.apps[0].name == "isICC" );
    REQUIRE( pl.apps[0].isServer() );
    REQUIRE( pl.apps[1].name == "camsci1-dark" );
    REQUIRE( pl.apps[1].line == 4 );
    REQUIRE( pl.apps[2].name == "pyapp" );
    REQUIRE( pl.apps[2].prefix == "py:" );
    REQUIRE( pl.apps[2].notifies() );
    REQUIRE( pl.apps[3].name == "quiet" );
    REQUIRE( !pl.apps[3].notifies() );
    REQUIRE( pl.apps[4].prefix == "nhb:" );
    REQUIRE( !pl.apps[4].notifies() );

    // three tokens, unknown prefix, duplicate, path in the executable
    REQUIRE( pl.errors.size() == 4 );

    // the commented-out process
    REQUIRE( pl.disabled.size() == 1 );
    REQUIRE( pl.disabled[0].name == "off" );
}

/// Commented-out processes are kept apart, ordinary comments are not, and an active line wins.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl keeps commented-out proclist processes", "[resuctrl]" )
{
    std::istringstream in( "#process-ID   Executable\n"
                           "#Processes for the MagAO-X ICC\n"
                           "##### camflowfs\n"
                           "#camsci1  picamCtrl\n"
                           "#  camsci2  picamCtrl\n"
                           "#py:pyapp  pyExec\n"
                           "##camsci_old  camsci\n"
                           "## camsci_old2  camsci\n"
                           "#==========  ==========\n"
                           "camsci3  picamCtrl\n"
                           "#camsci3  picamCtrl\n"
                           "#camsci1  otherExec\n" );
    proclist           pl = parseProclist( in );

    REQUIRE( pl.errors.empty() );
    REQUIRE( pl.apps.size() == 1 );
    REQUIRE( pl.apps[0].name == "camsci3" );

    // the column header looks like a process; reset skips it as it has no .conf
    std::vector<std::string> names;
    for( const auto &d : pl.disabled )
    {
        names.push_back( d.name );
    }
    REQUIRE( names == std::vector<std::string>{ "process-ID", "camsci1", "camsci2", "pyapp" } );
    REQUIRE( pl.disabled[1].exec == "picamCtrl" );
    REQUIRE( pl.disabled[3].prefix == "py:" );
}

/// systemd time spans, as systemctl show prints them, convert to seconds.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl reads systemd time spans", "[resuctrl]" )
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
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl reads MagAO-X and TOML configuration", "[resuctrl]" )
{
    std::istringstream in( "top = 1\n"
                           "[resurrectee]\n"
                           "timeout = [20, 30]  # comment\n"
                           "[systemd]\n"
                           "killSignal = \"SIGINT\"\n"
                           "restart='on-failure'\n" );
    confMap            c = parseConf( in );

    REQUIRE( c["top"] == "1" );
    REQUIRE( c["resurrectee.timeout"] == "20, 30" );
    REQUIRE( c["systemd.killSignal"] == "SIGINT" );
    REQUIRE( c["systemd.restart"] == "on-failure" );
}

/// App settings map the resurrector timeouts and [systemd] keys onto systemd directives, and reject bad values.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl maps app configuration to systemd directives", "[resuctrl]" )
{
    appEntry app;
    app.name = "cam";
    app.exec = "camCtrl";
    std::vector<std::string> errors;

    SECTION( "resurrectee timeouts become WatchdogSec and TimeoutStartSec" )
    {
        directiveMap d = appDirectives( app, { { "resurrectee.timeout", "20,90" } }, errors );
        REQUIRE( errors.empty() );
        REQUIRE( d["WatchdogSec"] == "20" );
        REQUIRE( d["TimeoutStartSec"] == "90" );
    }

    SECTION( "a negative running timeout disables the watchdog, zero means the default" )
    {
        REQUIRE( appDirectives( app, { { "resurrectee.timeout", "-1" } }, errors )["WatchdogSec"] == "0" );
        REQUIRE( appDirectives( app, { { "resurrectee.timeout", "0" } }, errors )["WatchdogSec"] == "10" );
    }

    SECTION( "[systemd] settings override the timeouts" )
    {
        directiveMap d = appDirectives(
            app,
            { { "resurrectee.timeout", "20" }, { "systemd.watchdogSec", "45" }, { "systemd.startLimitBurst", "3" } },
            errors );
        REQUIRE( errors.empty() );
        REQUIRE( d["WatchdogSec"] == "45" );
        REQUIRE( d["StartLimitBurst"] == "3" );
    }

    SECTION( "bad values, unknown keys and global-only keys are errors" )
    {
        appDirectives( app,
                       { { "resurrectee.timeout", "x" },
                         { "systemd.restart", "sometimes" },
                         { "systemd.killSignal", "SIGSEGV" },
                         { "systemd.timeoutStopSec", "5\nExecStartPre=/bin/evil" },
                         { "systemd.nonsense", "1" } },
                       errors );
        REQUIRE( errors.size() == 5 );
    }

    SECTION( "apps without heartbeats are simple services without a watchdog" )
    {
        app.prefix     = "nhb:";
        directiveMap d = appDirectives( app, { { "resurrectee.timeout", "20" } }, errors );
        REQUIRE( d["Type"] == "simple" );
        REQUIRE( d["WatchdogSec"] == "0" );
    }
}

/// Values are validated by kind.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl validates setting values", "[resuctrl]" )
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
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl names units by role", "[resuctrl]" )
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
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl generates the template and drop-ins", "[resuctrl]" )
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

/// Service states are reported with the old resuctrl's ranks and colors, and apps run by hand stand out.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl classifies app states like the old resuctrl", "[resuctrl]" )
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

/// Enabling, disabling and deleting edit only the app's proclist lines.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl edits an app's proclist lines", "[resuctrl]" )
{
    std::string text    = "# comment camsci1\n"
                          "isICC        xindiserver\n"
                          "camsci1      picamCtrl\n"
                          "#camsci2     picamCtrl\n"
                          "# py:pyapp   pyExec\n"
                          "camsci10     picamCtrl\n";
    int         changed = 0;

    SECTION( "disable comments out the active line" )
    {
        std::string t = editProclist( text, "camsci1", proclistEdit::comment, "", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t.find( "#camsci1      picamCtrl\n" ) != std::string::npos );
        REQUIRE( t.find( "\ncamsci10     picamCtrl\n" ) != std::string::npos );
        REQUIRE( t.find( "# comment camsci1\n" ) == 0 );
    }

    SECTION( "enable uncomments, with or without a space after #, keeping the prefix" )
    {
        std::string t = editProclist( text, "camsci2", proclistEdit::uncomment, "camsci2 picamCtrl", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t.find( "\ncamsci2     picamCtrl\n" ) != std::string::npos );

        t = editProclist( text, "pyapp", proclistEdit::uncomment, "py:pyapp pyExec", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t.find( "\npy:pyapp   pyExec\n" ) != std::string::npos );
    }

    SECTION( "enable of an active app changes nothing, enable of an unlisted app appends it" )
    {
        REQUIRE( editProclist( text, "camsci1", proclistEdit::uncomment, "camsci1 picamCtrl", changed ) == text );
        REQUIRE( changed == 0 );

        std::string t = editProclist( text, "newapp", proclistEdit::uncomment, "newapp someExec", changed );
        REQUIRE( changed == 1 );
        REQUIRE( t == text + "newapp someExec\n" );
    }

    SECTION( "## lines are only comments: never enabled, disabled or deleted" )
    {
        std::string withOld = text + "##camsci_old  camsci\n## camsci1  picamCtrl\n";

        std::string t = editProclist( withOld, "camsci_old", proclistEdit::uncomment, "camsci_old camsci", changed );
        REQUIRE( changed == 1 ); // appended as a new line, the ## comment stays
        REQUIRE( t.find( "##camsci_old  camsci\n" ) != std::string::npos );
        REQUIRE( t.find( "\ncamsci_old camsci\n" ) != std::string::npos );

        t = editProclist( withOld, "camsci1", proclistEdit::remove, "", changed );
        REQUIRE( changed == 1 ); // only the active line
        REQUIRE( t.find( "## camsci1  picamCtrl\n" ) != std::string::npos );

        std::istringstream in( withOld );
        proclist           pl = parseProclist( in );
        for( const auto &d : pl.disabled )
        {
            REQUIRE( d.name != "camsci_old" );
        }
    }

    SECTION( "delete removes active and commented-out lines of that app only" )
    {
        std::string t = editProclist( text + "#camsci1 picamCtrl\n", "camsci1", proclistEdit::remove, "", changed );
        REQUIRE( changed == 2 );
        REQUIRE( t.find( "camsci1 " ) == std::string::npos );
        REQUIRE( t.find( "camsci10" ) != std::string::npos );
        REQUIRE( t.find( "# comment camsci1" ) != std::string::npos );
    }
}

/// Files are only rewritten when their contents change.
/**
 * \ingroup resuctrl_unit_test
 */
TEST_CASE( "resuctrl writes files only when they change", "[resuctrl]" )
{
    std::string dir  = std::filesystem::temp_directory_path() / ( "resuctrl_test_" + std::to_string( getpid() ) );
    std::string path = dir + "/sub/file.conf";

    REQUIRE( writeIfChanged( path, "a\n", 0644 ) == 1 );
    REQUIRE( writeIfChanged( path, "a\n", 0644 ) == 0 );
    REQUIRE( writeIfChanged( path, "b\n", 0644 ) == 1 );

    std::filesystem::remove_all( dir );
}

} // namespace resuctrlTest
} // namespace libXWCTest
