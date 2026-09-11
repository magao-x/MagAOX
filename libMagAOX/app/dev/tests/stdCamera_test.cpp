/** \file stdCamera_test.cpp
 * \brief Catch2 tests for the MagAOX::app::dev::stdCamera device mixin.
 * \author Claude
 *
 * These tests cover the free helper functions of stdCamera and the stdCamera<derivedT> mixin itself.
 * They use two MagAOXApp<false> harnesses declared in stdCamera_test.hpp. stdCameraFullHarness turns
 * every capability flag on. stdCameraReportOnlyHarness only reports temperature and frame rate and
 * turns everything else off. The harnesses count calls to the derived-class interface and let a test
 * force any setter or property registration to fail.
 *
 * INDI callbacks are driven directly with hand-built pcf::IndiProperty objects. No INDI server is
 * needed. A few tests install a FIFO-less INDI driver so that the property update code runs.
 *
 * The tests write small config files under /tmp and the telemetry test writes logs under
 * /tmp/stdcam_test_telems.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#include "stdCamera_test.hpp"

using namespace MagAOX::app;
using namespace stdCamera_tests;

/** \defgroup stdCamera_tests libXWC::app::dev::stdCamera Unit Tests
 * \ingroup app_dev_unit_tests
 */

/// Verify that the free function stripQuotedWhitespace() trims whitespace and removes one matched pair of wrapping quotes.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera stripQuotedWhitespace", "[dev::stdCamera]" )
{
    SECTION( "empty string is untouched" )
    {
        std::string s = "";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "" );
    }

    SECTION( "all whitespace becomes empty" )
    {
        std::string s = "   \t\r\n  ";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "" );
    }

    SECTION( "leading and trailing whitespace is trimmed" )
    {
        std::string s = "  hello world  ";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "hello world" );
    }

    SECTION( "wrapping quotes are stripped after trimming" )
    {
        std::string s = "  \"{}-{}\"  ";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "{}-{}" );
    }

    SECTION( "a single quote character is left alone" )
    {
        std::string s = "\"";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "\"" );
    }

    SECTION( "unmatched quotes are left alone" )
    {
        std::string s = "\"unterminated";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "\"unterminated" );
    }

    SECTION( "no quotes, no whitespace is untouched" )
    {
        std::string s = "plain";
        dev::stripQuotedWhitespace( s );
        REQUIRE( s == "plain" );
    }
}

/// Verify that the free function loadCameraConfig() builds a cameraConfigMap from the unused sections of a config file.
/** Each section that has a configFile key becomes one camera mode. Config files are written under /tmp.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera loadCameraConfig", "[dev::stdCamera]" )
{
    SECTION( "no unused sections returns CAMCTRL_E_NOCONFIGS" )
    {
        // An empty config file has no sections at all, so unusedSections() reports none.
        mx::app::writeConfigFile( "/tmp/stdCamera_loadCameraConfig_none.conf", {}, {}, {} );

        mx::app::appConfigurator config;
        config.readConfig( "/tmp/stdCamera_loadCameraConfig_none.conf" );

        dev::cameraConfigMap ccmap;
        int                  rv = dev::loadCameraConfig( ccmap, config );
        REQUIRE( rv == CAMCTRL_E_NOCONFIGS );
        REQUIRE( ccmap.size() == 0 );
    }

    SECTION( "sections without configFile are skipped" )
    {
        mx::app::writeConfigFile(
            "/tmp/stdCamera_loadCameraConfig_skip.conf", { "modeA" }, { "serialCommand" }, { "cmd" } ); // any value, the section has no configFile so it is skipped

        mx::app::appConfigurator config;
        config.readConfig( "/tmp/stdCamera_loadCameraConfig_skip.conf" );

        dev::cameraConfigMap ccmap;
        int                  rv = dev::loadCameraConfig( ccmap, config );
        REQUIRE( rv == 0 );
        REQUIRE( ccmap.size() == 0 );
    }

    SECTION( "sections with configFile are parsed" )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_loadCameraConfig_full.conf",
                                  { "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA",
                                    "modeA" },
                                  { "configFile",
                                    "serialCommand",
                                    "centerX",
                                    "centerY",
                                    "sizeX",
                                    "sizeY",
                                    "binningX",
                                    "binningY",
                                    "digital_binningX",
                                    "digital_binningY" },
                                  { "modeA.cfg", "cmd", "512", "513", "1024", "1025", "2", "3", "4", "5" } ); // arbitrary distinct values

        mx::app::appConfigurator config;
        config.readConfig( "/tmp/stdCamera_loadCameraConfig_full.conf" );

        dev::cameraConfigMap ccmap;
        int                  rv = dev::loadCameraConfig( ccmap, config );
        REQUIRE( rv == 0 );
        REQUIRE( ccmap.size() == 1 );
        REQUIRE( ccmap.count( "modeA" ) == 1 );

        const dev::cameraConfig &cc = ccmap["modeA"];
        REQUIRE( cc.m_configFile == "modeA.cfg" );
        REQUIRE( cc.m_serialCommand == "cmd" );
        REQUIRE( cc.m_centerX == 512 );
        REQUIRE( cc.m_centerY == 513 );
        REQUIRE( cc.m_sizeX == 1024 );
        REQUIRE( cc.m_sizeY == 1025 );
        REQUIRE( cc.m_binningX == 2 );
        REQUIRE( cc.m_binningY == 3 );
        REQUIRE( cc.m_digitalBinX == 4 );
        REQUIRE( cc.m_digitalBinY == 5 );
    }

    SECTION( "multiple mode sections are all parsed" )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_loadCameraConfig_multi.conf",
                                  { "modeA", "modeB" },
                                  { "configFile", "configFile" },
                                  { "a.cfg", "b.cfg" } );

        mx::app::appConfigurator config;
        config.readConfig( "/tmp/stdCamera_loadCameraConfig_multi.conf" );

        dev::cameraConfigMap ccmap;
        int                  rv = dev::loadCameraConfig( ccmap, config );
        REQUIRE( rv == 0 );
        REQUIRE( ccmap.size() == 2 );
    }
}

namespace
{

/// Write a valid stdCamera config file to the given path.
/** The file exercises every feature of stdCameraFullHarness. It sets the full and default ROI, two
 * camera modes, the focus-state helper, and a two-switch goto-focus helper.
 */
void writeFullValidConfig( const std::string &path )
{
    std::vector<std::string> s, k, v;

    auto add = [&]( const std::string &sec, const std::string &key, const std::string &val )
    {
        s.push_back( sec );
        k.push_back( key );
        v.push_back( val );
    };

    add( "camera", "startupTemp", "-40" );
    add( "camera", "defaultReadoutSpeed", "slow" );
    add( "camera", "defaultVShiftSpeed", "std" );
    add( "camera", "fanSpeedControl", "true" );
    add( "camera", "defaultFanSpeed", "low" );
    add( "camera", "startupLED", "true" );
    add( "camera", "maxEMGain", "200" );
    add( "camera", "startupMode", "modeA" );
    add( "camera", "full_x", "511.5" );
    add( "camera", "full_y", "511.5" );
    add( "camera", "full_w", "1024" );
    add( "camera", "full_h", "1024" );
    add( "camera", "full_bin_x", "1" );
    add( "camera", "full_bin_y", "1" );
    add( "camera", "default_x", "255.5" );
    add( "camera", "default_y", "255.5" );
    add( "camera", "default_w", "512" );
    add( "camera", "default_h", "512" );
    add( "camera", "default_bin_x", "1" );
    add( "camera", "default_bin_y", "1" );

    add( "modeA", "configFile", "modeA.cfg" );
    add( "modeB", "configFile", "modeB.cfg" );

    add( "focus", "stateProperty", "sre.caution" );
    add( "focus", "stateElement", "focus-mismatch" );
    add( "focus", "stateElementOnMeansInFocus", "false" );

    add( "focus.gotoFocus", "numSwitches", "2" );
    add( "focus.gotoFocus", "property1", "stagebs.presetName" );
    add( "focus.gotoFocus", "property2", "fwfpm.filterName" );
    add( "focus.gotoFocus", "format", "{}-{}" );
    add( "focus.gotoFocus", "targetProperty", "stagesci1.presetName" );

    mx::app::writeConfigFile( path, s, k, v );
}

} // namespace

/// Verify that stdCamera::setupConfig() succeeds on both harnesses, with and without configured fan-speed names.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera setupConfig", "[dev::stdCamera]" )
{
    SECTION( "full-featured harness, with configured fan-speed names" )
    {
        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
    }

    SECTION( "full-featured harness, with no fan-speed names configured yet" )
    {
        stdCameraFullHarness app;
        app.m_fanSpeedNames.clear();
        app.m_fanSpeedNameLabels.clear();
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
    }

    SECTION( "report-only harness (most features disabled)" )
    {
        stdCameraReportOnlyHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
    }
}

/// Verify that stdCamera::loadConfig() reads a valid config file and rejects invalid fan speeds and missing full ROI values.
/** It also checks that the default ROI falls back to the full ROI when it is not configured.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera loadConfig", "[dev::stdCamera]" )
{
    SECTION( "a fully valid config file loads without error" )
    {
        writeFullValidConfig( "/tmp/stdCamera_loadConfig_valid.conf" );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_loadConfig_valid.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );

        REQUIRE( app.m_startupTemp == -40 );
        REQUIRE( app.m_defaultReadoutSpeed == "slow" );
        REQUIRE( app.m_defaultVShiftSpeed == "std" );
        REQUIRE( app.m_fanSpeedControlEnabled == true );
        REQUIRE( app.m_defaultFanSpeed == "low" );
        REQUIRE( app.m_defaultLEDState == true );
        REQUIRE( app.m_maxEMGain == 200 );
        REQUIRE( app.m_cameraModes.size() == 2 );
        REQUIRE( app.m_startupMode == "modeA" );
        REQUIRE( app.m_currentROI.x == 255.5 );
        REQUIRE( app.m_nextROI.w == 512 );
        REQUIRE( app.m_focusStateHelperConfigured == true );
        REQUIRE( app.m_focusGotoHelperConfigured == true );
        REQUIRE( app.m_hasFocus == true );
        REQUIRE( app.m_focusGotoFormat == "{}-{}" );
        REQUIRE( app.m_focusGotoTargetDevice == "stagesci1" );
        REQUIRE( app.m_focusGotoTargetName == "presetName" );
    }

    SECTION( "an invalid default fan speed is rejected" )
    {
        std::vector<std::string> s, k, v;
        s.push_back( "camera" );
        k.push_back( "defaultFanSpeed" );
        v.push_back( "not-a-speed" );
        mx::app::writeConfigFile( "/tmp/stdCamera_loadConfig_badfan.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_loadConfig_badfan.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "an invalid default fan speed is rejected even with no configured names" )
    {
        std::vector<std::string> s, k, v;
        s.push_back( "camera" );
        k.push_back( "defaultFanSpeed" );
        v.push_back( "not-a-speed" );
        mx::app::writeConfigFile( "/tmp/stdCamera_loadConfig_badfan2.conf", s, k, v );

        stdCameraFullHarness app;
        app.m_fanSpeedNames.clear();
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_loadConfig_badfan2.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "missing full ROI values are individually rejected" )
    {
        // Set the target key to zero. A zero value fails the equals-zero check for x, y, w, and h.
        // It also fails the less-than-one check for bin_x and bin_y.
        auto testMissing = [&]( const std::string &zeroKey )
        {
            std::vector<std::string> s, k, v;
            auto                     add = [&]( const std::string &key, const std::string &val )
            {
                s.push_back( "camera" );
                k.push_back( key );
                v.push_back( key == zeroKey ? "0" : val );
            };
            add( "full_x", "511.5" );
            add( "full_y", "511.5" );
            add( "full_w", "1024" );
            add( "full_h", "1024" );
            add( "full_bin_x", "1" );
            add( "full_bin_y", "1" );

            mx::app::writeConfigFile( "/tmp/stdCamera_loadConfig_roi_" + zeroKey + ".conf", s, k, v );

            stdCameraFullHarness app;
            mx::app::appConfigurator config;
            REQUIRE( app.setupConfig( config ) == 0 );
            config.readConfig( "/tmp/stdCamera_loadConfig_roi_" + zeroKey + ".conf" );
            REQUIRE( app.loadConfig( config ) == -1 );
        };

        testMissing( "full_x" );
        testMissing( "full_y" );
        testMissing( "full_w" );
        testMissing( "full_h" );
        testMissing( "full_bin_x" );
        testMissing( "full_bin_y" );
    }

    SECTION( "default ROI values fall back to the full ROI values when unset" )
    {
        std::vector<std::string> s, k, v;
        auto                     add = [&]( const std::string &key, const std::string &val )
        {
            s.push_back( "camera" );
            k.push_back( key );
            v.push_back( val );
        };
        add( "full_x", "600" );
        add( "full_y", "601" );
        add( "full_w", "800" );
        add( "full_h", "801" );
        add( "full_bin_x", "2" );
        add( "full_bin_y", "3" );

        mx::app::writeConfigFile( "/tmp/stdCamera_loadConfig_roidefaults.conf", s, k, v );

        stdCameraFullHarness app;
        // Clear the defaults set by the harness constructor so that the fallback logic in loadConfig()
        // runs. A nonzero default_x, default_y, default_w, or default_h would skip the fallback.
        // A binning value of at least one would also skip it.
        app.m_default_x     = 0;
        app.m_default_y     = 0;
        app.m_default_w     = 0;
        app.m_default_h     = 0;
        app.m_default_bin_x = 0;
        app.m_default_bin_y = 0;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_loadConfig_roidefaults.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );

        REQUIRE( app.m_default_x == 600 );
        REQUIRE( app.m_default_y == 601 );
        REQUIRE( app.m_default_w == 800 );
        REQUIRE( app.m_default_h == 801 );
        REQUIRE( app.m_default_bin_x == 2 );
        REQUIRE( app.m_default_bin_y == 3 );
        REQUIRE( app.m_currentROI.x == 600 );
        REQUIRE( app.m_nextROI.bin_y == 3 );
    }

    SECTION( "no camera mode sections logs but does not fail" )
    {
        std::vector<std::string> s, k, v;
        auto                     add = [&]( const std::string &key, const std::string &val )
        {
            s.push_back( "camera" );
            k.push_back( key );
            v.push_back( val );
        };
        add( "full_x", "511.5" );
        add( "full_y", "511.5" );
        add( "full_w", "1024" );
        add( "full_h", "1024" );
        add( "full_bin_x", "1" );
        add( "full_bin_y", "1" );

        mx::app::writeConfigFile( "/tmp/stdCamera_loadConfig_nomodes.conf", s, k, v );

        stdCameraFullHarness app;
        app.m_defaultFanSpeed = "low"; // Satisfy the fan-speed validation. It is not under test here.
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_loadConfig_nomodes.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.m_cameraModes.size() == 0 );
    }

    SECTION( "report-only harness with a trivial config file" )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_loadConfig_reportonly.conf", { "none" }, { "nada" }, { "0" } ); // the placeholder entry every test uses so the config file is not empty

        stdCameraReportOnlyHarness app;
        mx::app::appConfigurator   config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_loadConfig_reportonly.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
    }
}

/// Verify how stdCamera::loadConfig() parses the focus.stateProperty and focus.gotoFocus config sections.
/** Each section writes a small config file under /tmp with the full ROI plus one focus configuration.
 * Malformed goto-focus formats, missing keys, and invalid INDI keys must be rejected.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera loadConfig focus helper parsing", "[dev::stdCamera]" )
{
    auto baseROI = []( std::vector<std::string> &s, std::vector<std::string> &k, std::vector<std::string> &v )
    {
        auto add = [&]( const std::string &sec, const std::string &key, const std::string &val )
        {
            s.push_back( sec );
            k.push_back( key );
            v.push_back( val );
        };
        add( "camera", "full_x", "511.5" );
        add( "camera", "full_y", "511.5" );
        add( "camera", "full_w", "1024" );
        add( "camera", "full_h", "1024" );
        add( "camera", "full_bin_x", "1" );
        add( "camera", "full_bin_y", "1" );
    };

    SECTION( "no focus configuration at all leaves helpers unconfigured" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_none.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_none.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.m_focusStateHelperConfigured == false );
        REQUIRE( app.m_focusGotoHelperConfigured == false );
        REQUIRE( app.m_focusStateElement == "toggle" );
    }

    SECTION( "state helper only, goto helper unconfigured, does not enable m_hasFocus" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus" );
        k.push_back( "stateProperty" );
        v.push_back( "sre.caution" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_stateonly.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_stateonly.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.m_focusStateHelperConfigured == true );
        REQUIRE( app.m_focusGotoHelperConfigured == false );
        REQUIRE( app.m_hasFocus == false );
    }

    SECTION( "invalid focus.stateProperty is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus" );
        k.push_back( "stateProperty" );
        v.push_back( "nodothere" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_badstateprop.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_badstateprop.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with numSwitches < 1 is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotobadnum.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotobadnum.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with no format is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotonoformat.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotonoformat.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with no targetProperty is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotonotarget.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotonotarget.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with unbalanced braces is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{x}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotobadbraces.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotobadbraces.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with a stray closing brace is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "}{" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotostraybrace.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotostraybrace.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with mismatched placeholder count is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "2" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property2" );
        v.push_back( "fwfpm.filterName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotomismatch.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotomismatch.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with an invalid targetProperty key is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "nodothere" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotobadtarget.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotobadtarget.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with a missing property is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotomissingprop.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotomissingprop.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper with an invalid property key is rejected" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "nodothere" );
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotobadprop.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotobadprop.conf" );
        REQUIRE( app.loadConfig( config ) == -1 );
    }

    SECTION( "goto helper reusing the state-property key hits the already-monitored branch" )
    {
        std::vector<std::string> s, k, v;
        baseROI( s, k, v );
        s.push_back( "focus" );
        k.push_back( "stateProperty" );
        v.push_back( "stagebs.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "numSwitches" );
        v.push_back( "1" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "format" );
        v.push_back( "{}" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "targetProperty" );
        v.push_back( "stagesci1.presetName" );
        s.push_back( "focus.gotoFocus" );
        k.push_back( "property1" );
        v.push_back( "stagebs.presetName" ); // This is the same key as focus.stateProperty.
        mx::app::writeConfigFile( "/tmp/stdCamera_focus_gotoreuse.conf", s, k, v );

        stdCameraFullHarness app;
        mx::app::appConfigurator config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_focus_gotoreuse.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.m_focusMonitoredPropertyKeys.size() == 1 );
        REQUIRE( app.m_focusStateSourceIndex == app.m_focusGotoSourceIndices[0] );
        REQUIRE( app.m_hasFocus == true );
    }
}

namespace
{

/// Configure a stdCameraFullHarness so that appStartup() will succeed.
/** It loads the valid config file written by writeFullValidConfig() and points the telemetry log
 * at /tmp/stdcam_test_telems.
 */
void configureFullHarness( stdCameraFullHarness &app )
{
    writeFullValidConfig( "/tmp/stdCamera_appStartup_valid.conf" );

    mx::app::appConfigurator config;
    REQUIRE( app.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/stdCamera_appStartup_valid.conf" );
    REQUIRE( app.loadConfig( config ) == 0 );
    app.m_tel.logPath( "/tmp/stdcam_test_telems" );
}

} // namespace

/// Verify stdCamera::appStartup() on the full harness, including every property registration failure branch.
/** The harness counts registration calls and can force the Nth call to fail.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera appStartup (full harness)", "[dev::stdCamera]" )
{
    SECTION( "a successful startup registers every property" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );

        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.m_regCallCount > 0 );

        REQUIRE( app.m_indiP_temp.getName() == "temp_ccd" );
        REQUIRE( app.m_indiP_tempcont.getName() == "temp_controller" );
        REQUIRE( app.m_indiP_readoutSpeed.getName() == "readout_speed" );
        REQUIRE( app.m_indiP_vShiftSpeed.getName() == "vshift_speed" );
        REQUIRE( app.m_indiP_emGain.getName() == "emgain" );
        REQUIRE( app.m_indiP_exptime.getName() == "exptime" );
        REQUIRE( app.m_indiP_fps.getName() == "fps" );
        REQUIRE( app.m_indiP_fanSpeed.getName() == "fan_speed" );
        REQUIRE( app.m_indiP_analogGain.getName() == "analog_gain" );
        REQUIRE( app.m_indiP_led.getName() == "led" );
        REQUIRE( app.m_indiP_synchro.getName() == "synchro" );
        REQUIRE( app.m_indiP_mode.getName() == "mode" );
        REQUIRE( app.m_indiP_mode.find( "modeA" ) );
        REQUIRE( app.m_indiP_mode.find( "modeB" ) );
        REQUIRE( app.m_indiP_reconfig.getName() == "reconfigure" );
        REQUIRE( app.m_indiP_roi_x.getName() == "roi_region_x" );
        REQUIRE( app.m_indiP_cropMode.getName() == "roi_crop_mode" );
        REQUIRE( app.m_indiP_shutterStatus.getName() == "shutter_status" );
        REQUIRE( app.m_indiP_shutter.getName() == "shutter" );
        REQUIRE( app.m_indiP_focus.getName() == "focus" );
        REQUIRE( app.m_indiP_gotoFocus.getName() == "goto_focus" );
        REQUIRE( app.m_indiP_stateString.getName() == "state_string" );
        REQUIRE( app.m_indiP_focusMonitoredProperties.size() == app.m_focusMonitoredPropertyKeys.size() );
    }

    SECTION( "every registration failure is propagated" )
    {
        // First, discover exactly how many registerIndiProperty* calls a successful run makes.
        int totalCalls = 0;
        {
            stdCameraFullHarness app;
            configureFullHarness( app );
            REQUIRE( app.appStartup() == 0 );
            totalCalls = app.m_regCallCount;
        }
        REQUIRE( totalCalls > 0 );

        // createReadoutSpeed() and createVShiftSpeed() ignore the return value of their internal
        // registerIndiPropertyNew() call. Those are registration calls four and five. They come right
        // after the three temperature-control registrations. Forcing them to fail does not make
        // appStartup() fail. Every other call site checks its return value directly.
        for( int failAt = 1; failAt <= totalCalls; ++failAt )
        {
            stdCameraFullHarness app;
            configureFullHarness( app );
            app.m_regFailAt = failAt;

            int rv = app.appStartup();

            bool isIgnoredFailure = ( failAt == 4 || failAt == 5 );
            if( isIgnoredFailure )
            {
                REQUIRE( rv == 0 );
            }
            else
            {
                REQUIRE( rv == -1 );
            }
        }
    }

    SECTION( "an empty fan-speed name list is rejected" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        app.m_fanSpeedNames.clear();
        app.m_fanSpeedNameLabels.clear();
        REQUIRE( app.appStartup() == -1 );
    }

    SECTION( "mismatched fan-speed label count uses the unlabeled selection switch" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        app.m_fanSpeedNameLabels.resize( 1 ); // One label does not match the two entries in m_fanSpeedNames.
        REQUIRE( app.appStartup() == 0 );
    }

    SECTION( "an empty analog-gain name list is rejected" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        app.m_analogGainNames.clear();
        app.m_analogGainNameLabels.clear();
        REQUIRE( app.appStartup() == -1 );
    }

    SECTION( "mismatched analog-gain label count uses the unlabeled selection switch" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        app.m_analogGainNameLabels.resize( 1 ); // One label does not match the two entries in m_analogGainNames.
        REQUIRE( app.appStartup() == 0 );
    }

    SECTION( "no configured camera modes fails to create the mode selection switch" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        app.m_cameraModes.clear();
        REQUIRE( app.appStartup() == -1 );
    }

    SECTION( "an invalid cached monitored-focus-property key fails at startup" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.m_focusMonitoredPropertyKeys.size() > 0 );
        app.m_focusMonitoredPropertyKeys[0] = "no-dot-here"; // This bypasses the validation done by loadConfig().
        REQUIRE( app.appStartup() == -1 );
    }
}

/// Verify stdCamera::appStartup() on the report-only harness. Only the read-only temp_ccd and fps properties are created.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera appStartup (report-only harness)", "[dev::stdCamera]" )
{
    mx::app::writeConfigFile( "/tmp/stdCamera_appStartup_reportonly.conf", { "none" }, { "nada" }, { "0" } );

    stdCameraReportOnlyHarness app;
    mx::app::appConfigurator   config;
    REQUIRE( app.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/stdCamera_appStartup_reportonly.conf" );
    REQUIRE( app.loadConfig( config ) == 0 );

    REQUIRE( app.appStartup() == 0 );

    // Only the read-only temp_ccd and fps properties should have been created.
    REQUIRE( app.m_indiP_temp.getName() == "temp_ccd" );
    REQUIRE( app.m_indiP_fps.getName() == "fps" );
}

/// Verify that a failed registration of the read-only temp_ccd or fps property makes appStartup() fail on the report-only harness.
/** These are the report-only branches. They only run when tempControl and fpsCtrl are false while
 * temp and fps are true. The report-only harness is configured that way.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera appStartup (report-only harness) registration failures", "[dev::stdCamera]" )
{
    for( int failAt = 1; failAt <= 2; ++failAt )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_appStartup_reportonly_fail.conf", { "none" }, { "nada" }, { "0" } );

        stdCameraReportOnlyHarness app;
        mx::app::appConfigurator   config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_appStartup_reportonly_fail.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );

        app.m_regFailAt = failAt;
        REQUIRE( app.appStartup() == -1 );
    }
}

/// Verify stdCamera::appLogic() on the full harness. The POWERON state must apply power-on defaults and move to NOTCONNECTED.
/** The READY and OPERATING states must refresh the fan, analog gain, LED, and ROI selections.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera appLogic (full harness)", "[dev::stdCamera]" )
{
    SECTION( "the POWERON to NOTCONNECTED transition applies power-on defaults" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );

        app.m_shutterStatus = "OPERATING";
        app.m_shutterState   = 1;

        app.state( MagAOX::app::stateCodes::POWERON );
        REQUIRE( app.appLogic() == 0 );

        REQUIRE( app.m_powerOnDefaultsCalls == 1 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NOTCONNECTED );
        REQUIRE( app.m_ccdTempSetpt == app.m_startupTemp );
    }

    SECTION( "the POWERON to NOTCONNECTED transition with a shut/ready shutter" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );

        app.m_shutterStatus = "READY";
        app.m_shutterState   = 0;

        app.state( MagAOX::app::stateCodes::POWERON );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NOTCONNECTED );
    }

    SECTION( "the POWERON to NOTCONNECTED transition with an unknown shutter status" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );

        app.m_shutterStatus = "UNKNOWN";
        app.m_shutterState   = -1;

        app.state( MagAOX::app::stateCodes::POWERON );
        REQUIRE( app.appLogic() == 0 );
    }

    SECTION( "POWERON with the power-on wait not yet elapsed does nothing" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );

        app.m_powerMgtEnabled = true;
        app.m_powerOnWait     = 3600; // one hour
        app.m_powerOnCounter  = 0;

        app.state( MagAOX::app::stateCodes::POWERON );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_powerOnDefaultsCalls == 0 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::POWERON );
    }

    SECTION( "READY/OPERATING updates fan, analog gain, LED and ROI selections" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );

        app.m_fanSpeedValid   = true;
        app.m_analogGainValid = true;
        app.m_ledStateValid   = true;
        app.m_ledState        = true;

        app.state( MagAOX::app::stateCodes::READY );
        REQUIRE( app.appLogic() == 0 );

        app.m_ledState = false;
        app.state( MagAOX::app::stateCodes::OPERATING );
        REQUIRE( app.appLogic() == 0 );
    }
}

/// Verify stdCamera::appLogic() on the report-only harness. This covers the branches for disabled features.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera appLogic (report-only harness)", "[dev::stdCamera]" )
{
    mx::app::writeConfigFile( "/tmp/stdCamera_appLogic_reportonly.conf", { "none" }, { "nada" }, { "0" } );

    stdCameraReportOnlyHarness app;
    mx::app::appConfigurator   config;
    REQUIRE( app.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/stdCamera_appLogic_reportonly.conf" );
    REQUIRE( app.loadConfig( config ) == 0 );
    REQUIRE( app.appStartup() == 0 );

    app.state( MagAOX::app::stateCodes::POWERON );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == MagAOX::app::stateCodes::NOTCONNECTED );

    app.state( MagAOX::app::stateCodes::READY );
    REQUIRE( app.appLogic() == 0 );
}

/// Verify stdCamera::onPowerOff() and stdCamera::whilePowerOff() with and without an INDI driver.
/** Without a driver both functions return early. With a FIFO-less driver every property update
 * branch runs for each shutter status and each valid flag.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera onPowerOff and whilePowerOff", "[dev::stdCamera]" )
{
    SECTION( "full harness, with no INDI driver, is a no-op success" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );

        REQUIRE( app.onPowerOff() == 0 );

        app.m_shutterStatus = "OPERATING";
        REQUIRE( app.whilePowerOff() == 0 );

        app.m_shutterStatus = "READY";
        REQUIRE( app.whilePowerOff() == 0 );

        app.m_shutterStatus = "UNKNOWN";
        REQUIRE( app.whilePowerOff() == 0 );

        app.m_shutterState = 0;
        REQUIRE( app.whilePowerOff() == 0 );
    }

    SECTION( "full harness, with a real driver, exercises every property-update branch" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );
        app.setupRealDriver();

        app.m_shutterStatus = "OPERATING";
        app.m_shutterState  = 0;
        REQUIRE( app.onPowerOff() == 0 );

        app.m_shutterStatus = "READY";
        app.m_shutterState  = 1;
        REQUIRE( app.onPowerOff() == 0 );

        app.m_shutterStatus = "POWERON";
        REQUIRE( app.onPowerOff() == 0 );

        app.m_shutterStatus = "UNKNOWN";
        REQUIRE( app.onPowerOff() == 0 );

        app.m_fanSpeedValid   = true;
        app.m_analogGainValid = true;
        app.m_ledStateValid   = true;
        app.m_ledState        = true;
        REQUIRE( app.onPowerOff() == 0 );

        app.m_ledState = false;
        REQUIRE( app.onPowerOff() == 0 );

        app.m_shutterStatus = "OPERATING";
        REQUIRE( app.whilePowerOff() == 0 );

        app.m_shutterStatus = "READY";
        REQUIRE( app.whilePowerOff() == 0 );

        app.m_shutterStatus = "UNKNOWN";
        app.m_ledState      = true;
        REQUIRE( app.whilePowerOff() == 0 ); // m_fanSpeedValid and m_analogGainValid are still true here.

        app.m_ledState = false;
        REQUIRE( app.whilePowerOff() == 0 );
    }

    SECTION( "report-only harness onPowerOff/whilePowerOff" )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_onPowerOff_reportonly.conf", { "none" }, { "nada" }, { "0" } );

        stdCameraReportOnlyHarness app;
        mx::app::appConfigurator   config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_onPowerOff_reportonly.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.appStartup() == 0 );

        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.whilePowerOff() == 0 );
    }
}

/// Verify that stdCamera::appShutdown() returns success after a normal startup.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera appShutdown is a trivial no-op", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    configureFullHarness( app );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.appShutdown() == 0 );
}

/// Verify stdCamera::updateINDI() with a FIFO-less INDI driver installed so that every property update branch runs.
/** The test flips each piece of camera state and calls updateINDI() after each change.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera updateINDI", "[dev::stdCamera]" )
{
    SECTION( "full harness exercises every property update path" )
    {
        stdCameraFullHarness app;
        configureFullHarness( app );
        REQUIRE( app.appStartup() == 0 );
        app.setupRealDriver();

        // Temperature control is on but not yet on target.
        app.m_tempControlStatus   = true;
        app.m_tempControlOnTarget = false;
        REQUIRE( app.updateINDI() == 0 );

        // Temperature control is on and on target.
        app.m_tempControlOnTarget = true;
        REQUIRE( app.updateINDI() == 0 );

        // Temperature control is off.
        app.m_tempControlStatus = false;
        REQUIRE( app.updateINDI() == 0 );

        app.m_synchro = true;
        REQUIRE( app.updateINDI() == 0 );
        app.m_synchro = false;
        REQUIRE( app.updateINDI() == 0 );

        app.m_cropMode = true;
        REQUIRE( app.updateINDI() == 0 );
        app.m_cropMode = false;
        REQUIRE( app.updateINDI() == 0 );

        app.m_modeName = "modeA";
        app.m_nextMode = "modeB";
        REQUIRE( app.updateINDI() == 0 );
        app.m_nextMode = "";
        REQUIRE( app.updateINDI() == 0 );

        app.m_shutterStatus = "OPERATING";
        app.m_shutterState  = 1;
        REQUIRE( app.updateINDI() == 0 );

        app.m_shutterStatus = "READY";
        app.m_shutterState  = 0;
        REQUIRE( app.updateINDI() == 0 );

        app.m_fanSpeedValid   = true;
        app.m_analogGainValid = true;
        app.m_ledStateValid   = true;
        app.m_ledState        = true;
        REQUIRE( app.updateINDI() == 0 );
        app.m_ledState = false;
        REQUIRE( app.updateINDI() == 0 );

        // m_stateStringValidResult defaults to true, which takes the valid branch.
        // Flip it to take the invalid branch too.
        app.m_stateStringValidResult = false;
        REQUIRE( app.updateINDI() == 0 );
    }

    SECTION( "with no INDI driver set, returns immediately" )
    {
        stdCameraFullHarness app;
        REQUIRE( app.m_indiDriver == nullptr );
        REQUIRE( app.updateINDI() == 0 );
    }

    SECTION( "report-only harness" )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_updateINDI_reportonly.conf", { "none" }, { "nada" }, { "0" } );

        stdCameraReportOnlyHarness app;
        mx::app::appConfigurator   config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_updateINDI_reportonly.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.updateINDI() == 0 );
    }

    SECTION( "report-only harness with a real driver exercises the fps-report-only branch" )
    {
        mx::app::writeConfigFile( "/tmp/stdCamera_updateINDI_reportonly2.conf", { "none" }, { "nada" }, { "0" } );

        stdCameraReportOnlyHarness app;
        mx::app::appConfigurator   config;
        REQUIRE( app.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/stdCamera_updateINDI_reportonly2.conf" );
        REQUIRE( app.loadConfig( config ) == 0 );
        REQUIRE( app.appStartup() == 0 );
        app.setupRealDriver();

        // c_stdCamera_temp is true and c_stdCamera_tempControl is false on this harness.
        // This call therefore also runs the temperature report-only branch. That branch is
        // separate from the temperature-control branches covered by the full harness above.
        REQUIRE( app.updateINDI() == 0 );
    }
}

/// Verify that stdCamera::recordCamera() writes telemetry with and without the force flag and through the telemeter interface.
/** This test starts the real telemeter log thread and writes under /tmp/stdcam_test_telems.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera recordCamera", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    configureFullHarness( app );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.startTelemetry() == 0 );

    // The first call always records. recordTelem() uses force equal to true. The unforced path is
    // checked here as well.
    REQUIRE( app.recordCamera( false ) == 0 );
    REQUIRE( app.recordCamera( false ) == 0 ); // Nothing changed, so this does not record again. It must still succeed.

    app.m_expTime = 1.234f;
    REQUIRE( app.recordCamera( false ) == 0 );

    REQUIRE( app.recordCamera( true ) == 0 );

    // Drive it through the telemeter interface too.
    REQUIRE( app.checkRecordTimes() == 0 );
}

namespace
{

/// Build a switch-type INDI property with the given device and name. Each named element gets the given On or Off state.
pcf::IndiProperty makeSwitchProp( const std::string                                                       &device,
                                  const std::string                                                       &name,
                                  const std::vector<std::pair<std::string, pcf::IndiElement::SwitchStateType>> &elements )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( device );
    ip.setName( name );
    for( auto &el : elements )
    {
        ip.add( pcf::IndiElement( el.first ) );
        ip[el.first].setSwitchState( el.second );
    }
    return ip;
}

/// Build a number-type INDI property with the given device and name and one "target" element set to the given value.
pcf::IndiProperty makeNumberProp( const std::string &device, const std::string &name, double target )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( device );
    ip.setName( name );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"].set( target );
    return ip;
}

/// Configure and start a stdCameraFullHarness so that it can be driven through INDI callbacks.
/** The harness is passed by reference and not returned by value. MagAOXApp holds a std::mutex, so
 * the harness cannot be moved or copied.
 */
void startFullHarness( stdCameraFullHarness &app )
{
    configureFullHarness( app );
    REQUIRE( app.appStartup() == 0 );

    // registerIndiPropertySet() does nothing when there is no INDI driver, which is the case for
    // MagAOXApp<false>. It therefore never sets the device and name on the cached focus-monitored
    // properties the way a real INDI subscription would. Fill those in here from the parsed
    // monitored-property keys. Then setCallBack_focusMonitored() can match incoming properties by
    // device and name, just as it would in a running INDI-connected app.
    for( size_t n = 0; n < app.m_focusMonitoredPropertyKeys.size(); ++n )
    {
        std::string devName, propName;
        REQUIRE( MagAOX::app::indi::parseIndiKey( devName, propName, app.m_focusMonitoredPropertyKeys[n] ) == 0 );
        app.m_indiP_focusMonitoredProperties[n].setDevice( devName );
        app.m_indiP_focusMonitoredProperties[n].setName( propName );
    }
}

} // namespace

/// Verify that stdCamera::newCallBack_stdCamera() rejects the wrong device or an unknown property and dispatches every known property name.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_stdCamera dispatch", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "wrong device is rejected" )
    {
        pcf::IndiProperty ip = makeNumberProp( "not-the-device", "temp_ccd", 10 );
        REQUIRE( app.newCallBack_stdCamera( ip ) == -1 );
    }

    SECTION( "unknown property name is rejected" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "not_a_property", 10 );
        REQUIRE( app.newCallBack_stdCamera( ip ) == -1 );
    }

    // Dispatch to every known property and confirm that it reaches the right handler. Each call also
    // runs the dispatch branch of newCallBack_stdCamera() for that property name.
    SECTION( "dispatches to every known property" )
    {
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "reconfigure", { { "request", pcf::IndiElement::Off } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "temp_ccd", -30 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "temp_controller", { { "toggle", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "readout_speed", { { "slow", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "vshift_speed", { { "std", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "emgain", 50 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "exptime", 1 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "fps", 10 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "fan_speed", { { "low", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "analog_gain", { { "low", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "led", { { "toggle", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "synchro", { { "toggle", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "mode", { { "modeA", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_crop_mode", { { "toggle", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "roi_region_x", 100 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "roi_region_y", 100 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "roi_region_w", 100 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "roi_region_h", 100 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "roi_region_bin_x", 1 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeNumberProp( app.configName(), "roi_region_bin_y", 1 ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_region_check", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_set", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_set_full", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_set_full_bin", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_load_last", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_set_last", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "roi_set_default", { { "request", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "shutter", { { "toggle", pcf::IndiElement::On } } ) ) == 0 );
        REQUIRE( app.newCallBack_stdCamera( makeSwitchProp( app.configName(), "goto_focus", { { "request", pcf::IndiElement::On } } ) ) == 0 );
    }
}

/// Verify that stdCamera::newCallBack_temp() applies the target or current value through setTempSetPt() and propagates failures.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_temp", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "a target value is applied" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "temp_ccd", -35 );
        REQUIRE( app.newCallBack_temp( ip ) == 0 );
        REQUIRE( app.m_ccdTempSetpt == -35 );
        REQUIRE( app.m_setTempSetPtCalls == 1 );
    }

    SECTION( "a current-only value is applied" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( app.configName() );
        ip.setName( "temp_ccd" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].set( -20 );
        REQUIRE( app.newCallBack_temp( ip ) == 0 );
        REQUIRE( app.m_ccdTempSetpt == -20 );
    }

    SECTION( "a mismatched property is rejected" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "not_temp_ccd", -35 );
        REQUIRE( app.newCallBack_temp( ip ) == -1 );
    }

    SECTION( "setTempSetPt failure is propagated" )
    {
        app.m_setTempSetPtResult = -1;
        pcf::IndiProperty ip     = makeNumberProp( app.configName(), "temp_ccd", -35 );
        REQUIRE( app.newCallBack_temp( ip ) == -1 );
    }
}

/// Verify that the ROI and temperature-control callbacks do nothing when their feature is disabled at compile time.
/** The report-only harness has c_stdCamera_tempControl and c_stdCamera_usesROI set to false.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera callbacks are no-ops when disabled (report-only harness)", "[dev::stdCamera]" )
{
    stdCameraReportOnlyHarness app;

    pcf::IndiProperty numberProp( pcf::IndiProperty::Number );
    numberProp.setDevice( app.configName() );
    numberProp.setName( "temp_ccd" );
    REQUIRE( app.newCallBack_temp( numberProp ) == 0 );

    pcf::IndiProperty requestOn( pcf::IndiProperty::Switch );
    requestOn.setDevice( app.configName() );
    requestOn.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

    requestOn.setName( "roi_region_check" );
    REQUIRE( app.newCallBack_roi_check( requestOn ) == 0 );

    requestOn.setName( "roi_set" );
    REQUIRE( app.newCallBack_roi_set( requestOn ) == 0 );

    requestOn.setName( "roi_set_full" );
    REQUIRE( app.newCallBack_roi_full( requestOn ) == 0 );

    requestOn.setName( "roi_set_full_bin" );
    REQUIRE( app.newCallBack_roi_fullbin( requestOn ) == 0 );

    requestOn.setName( "roi_load_last" );
    REQUIRE( app.newCallBack_roi_loadlast( requestOn ) == 0 );

    requestOn.setName( "roi_set_last" );
    REQUIRE( app.newCallBack_roi_last( requestOn ) == 0 );

    requestOn.setName( "roi_set_default" );
    REQUIRE( app.newCallBack_roi_default( requestOn ) == 0 );
}

/// Verify that stdCamera::newCallBack_temp_controller() turns temperature control on or off through setTempControl().
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_temp_controller", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "toggling on enables temp control" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "temp_controller", { { "toggle", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_temp_controller( ip ) == 0 );
        REQUIRE( app.m_tempControlStatusSet == true );
        REQUIRE( app.m_setTempControlCalls == 1 );
    }

    SECTION( "toggling off disables temp control" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "temp_controller", { { "toggle", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_temp_controller( ip ) == 0 );
        REQUIRE( app.m_tempControlStatusSet == false );
    }

    SECTION( "a property with no toggle element is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "temp_controller" );
        REQUIRE( app.newCallBack_temp_controller( ip ) == 0 );
        REQUIRE( app.m_setTempControlCalls == 0 );
    }
}

/// Verify that stdCamera::newCallBack_readoutSpeed() applies one selected speed, keeps the current speed when none is selected, and rejects more than one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_readoutSpeed", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "selecting a valid speed applies it" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "readout_speed", { { "fast", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_readoutSpeed( ip ) == 0 );
        REQUIRE( app.m_readoutSpeedNameSet == "fast" );
    }

    SECTION( "selecting nothing resets to the current speed" )
    {
        app.m_readoutSpeedName = "slow";
        pcf::IndiProperty ip   = makeSwitchProp(
            app.configName(), "readout_speed", { { "slow", pcf::IndiElement::Off }, { "fast", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_readoutSpeed( ip ) == 0 );
        REQUIRE( app.m_readoutSpeedNameSet == "slow" );
    }

    SECTION( "selecting more than one speed is an error" )
    {
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "readout_speed", { { "slow", pcf::IndiElement::On }, { "fast", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_readoutSpeed( ip ) == -1 );
    }
}

/// Verify that stdCamera::newCallBack_vShiftSpeed() applies one selected speed, keeps the current speed when none is selected, and rejects more than one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_vShiftSpeed", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "selecting a valid speed applies it" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "vshift_speed", { { "fast", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_vShiftSpeed( ip ) == 0 );
        REQUIRE( app.m_vShiftSpeedNameSet == "fast" );
    }

    SECTION( "selecting nothing resets to the current speed" )
    {
        app.m_vShiftSpeedName = "std";
        pcf::IndiProperty ip  = makeSwitchProp(
            app.configName(), "vshift_speed", { { "std", pcf::IndiElement::Off }, { "fast", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_vShiftSpeed( ip ) == 0 );
        REQUIRE( app.m_vShiftSpeedNameSet == "std" );
    }

    SECTION( "selecting more than one speed is an error" )
    {
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "vshift_speed", { { "std", pcf::IndiElement::On }, { "fast", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_vShiftSpeed( ip ) == -1 );
    }
}

/// Verify that newCallBack_emgain(), newCallBack_exptime(), and newCallBack_fps() apply the target value and reject a mismatched property name.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_emgain, exptime, fps", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "emgain applies the target and can fail" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "emgain", 42 );
        REQUIRE( app.newCallBack_emgain( ip ) == 0 );
        REQUIRE( app.m_emGainSet == 42 );

        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_emgain", 42 );
        REQUIRE( app.newCallBack_emgain( bad ) == -1 );
    }

    SECTION( "exptime applies the target and can fail" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "exptime", 2.5 );
        REQUIRE( app.newCallBack_exptime( ip ) == 0 );
        REQUIRE( app.m_expTimeSet == Approx( 2.5 ) );

        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_exptime", 2.5 );
        REQUIRE( app.newCallBack_exptime( bad ) == -1 );
    }

    SECTION( "fps applies the target and can fail" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "fps", 30 );
        REQUIRE( app.newCallBack_fps( ip ) == 0 );
        REQUIRE( app.m_fpsSet == 30 );

        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_fps", 30 );
        REQUIRE( app.newCallBack_fps( bad ) == -1 );
    }
}

/// Verify that stdCamera::newCallBack_fanSpeed() applies one selected speed, keeps the current speed when none is selected, and rejects more than one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_fanSpeed", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "selecting a valid speed applies it" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "fan_speed", { { "high", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_fanSpeed( ip ) == 0 );
        REQUIRE( app.m_fanSpeedNameSet == "high" );
    }

    SECTION( "selecting nothing resets to the current speed" )
    {
        app.m_fanSpeedName   = "low";
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "fan_speed", { { "low", pcf::IndiElement::Off }, { "high", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_fanSpeed( ip ) == 0 );
        REQUIRE( app.m_fanSpeedNameSet == "low" );
    }

    SECTION( "selecting more than one speed is an error" )
    {
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "fan_speed", { { "low", pcf::IndiElement::On }, { "high", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_fanSpeed( ip ) == -1 );
    }

    SECTION( "an element the property doesn't have is skipped" )
    {
        // Only "high" is present. The callback therefore takes the continue branch for the missing "low" element.
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "fan_speed", { { "high", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_fanSpeed( ip ) == 0 );
        REQUIRE( app.m_fanSpeedNameSet == "high" );
    }
}

/// Verify that stdCamera::newCallBack_analogGain() applies one selected gain, keeps the current gain when none is selected, and rejects more than one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_analogGain", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "selecting a valid gain applies it" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "analog_gain", { { "high", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_analogGain( ip ) == 0 );
        REQUIRE( app.m_analogGainNameSet == "high" );
    }

    SECTION( "selecting nothing resets to the current gain" )
    {
        app.m_analogGainName = "low";
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "analog_gain", { { "low", pcf::IndiElement::Off }, { "high", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_analogGain( ip ) == 0 );
        REQUIRE( app.m_analogGainNameSet == "low" );
    }

    SECTION( "selecting more than one gain is an error" )
    {
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "analog_gain", { { "low", pcf::IndiElement::On }, { "high", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_analogGain( ip ) == -1 );
    }
}

/// Verify that stdCamera::newCallBack_led() sets the LED state from the toggle element and ignores a property without one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_led", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "toggling on" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "led", { { "toggle", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_led( ip ) == 0 );
        REQUIRE( app.m_ledStateSet == true );
    }

    SECTION( "toggling off" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "led", { { "toggle", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_led( ip ) == 0 );
        REQUIRE( app.m_ledStateSet == false );
    }

    SECTION( "a property with no toggle element is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "led" );
        REQUIRE( app.newCallBack_led( ip ) == 0 );
        REQUIRE( app.m_setLEDCalls == 0 );
    }
}

/// Verify that stdCamera::newCallBack_synchro() sets the synchro state from the toggle element and ignores a property without one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_synchro", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "toggling on" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "synchro", { { "toggle", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_synchro( ip ) == 0 );
        REQUIRE( app.m_synchroSet == true );
    }

    SECTION( "toggling off" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "synchro", { { "toggle", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_synchro( ip ) == 0 );
        REQUIRE( app.m_synchroSet == false );
    }

    SECTION( "a property with no toggle element is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "synchro" );
        REQUIRE( app.newCallBack_synchro( ip ) == 0 );
        REQUIRE( app.m_setSynchroCalls == 0 );
    }
}

/// Verify that stdCamera::newCallBack_mode() records the selected mode and requests a reconfigure, and rejects bad input.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_mode", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "selecting a valid mode requests a reconfigure" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "mode", { { "modeB", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_mode( ip ) == 0 );
        REQUIRE( app.m_nextMode == "modeB" );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "selecting no mode does nothing" )
    {
        app.m_nextMode       = "";
        app.m_reconfig       = false;
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "mode", { { "modeA", pcf::IndiElement::Off }, { "modeB", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_mode( ip ) == 0 );
        REQUIRE( app.m_nextMode == "" );
        REQUIRE( app.m_reconfig == false );
    }

    SECTION( "selecting more than one mode is an error" )
    {
        pcf::IndiProperty ip = makeSwitchProp(
            app.configName(), "mode", { { "modeA", pcf::IndiElement::On }, { "modeB", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_mode( ip ) == -1 );
    }

    SECTION( "a mismatched property name is rejected" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "not_mode", { { "modeA", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_mode( ip ) == -1 );
    }
}

/// Verify that stdCamera::newCallBack_reconfigure() requests a reconfigure of the current mode only when the request element is On.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_reconfigure", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );
    app.m_modeName            = "modeA";

    SECTION( "requesting a reconfigure sets m_nextMode to the current mode" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "reconfigure", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_reconfigure( ip ) == 0 );
        REQUIRE( app.m_nextMode == "modeA" );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "a request of Off is ignored" )
    {
        app.m_reconfig        = false;
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "reconfigure", { { "request", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_reconfigure( ip ) == 0 );
        REQUIRE( app.m_reconfig == false );
    }

    SECTION( "a property with no request element is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "reconfigure" );
        REQUIRE( app.newCallBack_reconfigure( ip ) == 0 );
    }
}

/// Verify that stdCamera::newCallBack_cropMode() sets crop mode from the toggle element and ignores a property without one.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_cropMode", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "toggling on" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_crop_mode", { { "toggle", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_cropMode( ip ) == 0 );
        REQUIRE( app.m_cropModeSet == true );
    }

    SECTION( "toggling off" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_crop_mode", { { "toggle", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_cropMode( ip ) == 0 );
        REQUIRE( app.m_cropModeSet == false );
    }

    SECTION( "a property with no toggle element is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "roi_crop_mode" );
        REQUIRE( app.newCallBack_cropMode( ip ) == 0 );
        REQUIRE( app.m_setCropModeCalls == 0 );
    }
}

/// Verify that each ROI element callback stores the target in m_nextROI and resets m_nextROI from m_currentROI on a mismatched property name.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_roi_x/y/w/h/bin_x/bin_y", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "roi_x applies the target, and rejects a mismatch" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "roi_region_x", 100 );
        REQUIRE( app.newCallBack_roi_x( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == 100 );

        app.m_currentROI.x     = 42;
        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_roi_region_x", 100 );
        REQUIRE( app.newCallBack_roi_x( bad ) == -1 );
        REQUIRE( app.m_nextROI.x == 42 );
    }

    SECTION( "roi_y applies the target, and rejects a mismatch" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "roi_region_y", 100 );
        REQUIRE( app.newCallBack_roi_y( ip ) == 0 );
        REQUIRE( app.m_nextROI.y == 100 );

        app.m_currentROI.y     = 42;
        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_roi_region_y", 100 );
        REQUIRE( app.newCallBack_roi_y( bad ) == -1 );
        REQUIRE( app.m_nextROI.y == 42 );
    }

    SECTION( "roi_w applies the target, and rejects a mismatch" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "roi_region_w", 200 );
        REQUIRE( app.newCallBack_roi_w( ip ) == 0 );
        REQUIRE( app.m_nextROI.w == 200 );

        app.m_currentROI.w     = 42;
        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_roi_region_w", 200 );
        REQUIRE( app.newCallBack_roi_w( bad ) == -1 );
        REQUIRE( app.m_nextROI.w == 42 );
    }

    SECTION( "roi_h applies the target, and rejects a mismatch" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "roi_region_h", 200 );
        REQUIRE( app.newCallBack_roi_h( ip ) == 0 );
        REQUIRE( app.m_nextROI.h == 200 );

        app.m_currentROI.h     = 42;
        pcf::IndiProperty bad = makeNumberProp( app.configName(), "not_roi_region_h", 200 );
        REQUIRE( app.newCallBack_roi_h( bad ) == -1 );
        REQUIRE( app.m_nextROI.h == 42 );
    }

    SECTION( "roi_bin_x applies the target, and rejects a mismatch" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "roi_region_bin_x", 2 );
        REQUIRE( app.newCallBack_roi_bin_x( ip ) == 0 );
        REQUIRE( app.m_nextROI.bin_x == 2 );

        app.m_currentROI.bin_x = 42;
        pcf::IndiProperty bad  = makeNumberProp( app.configName(), "not_roi_region_bin_x", 2 );
        REQUIRE( app.newCallBack_roi_bin_x( bad ) == -1 );
        REQUIRE( app.m_nextROI.bin_x == 42 );
    }

    SECTION( "roi_bin_y applies the target, and rejects a mismatch" )
    {
        pcf::IndiProperty ip = makeNumberProp( app.configName(), "roi_region_bin_y", 2 );
        REQUIRE( app.newCallBack_roi_bin_y( ip ) == 0 );
        REQUIRE( app.m_nextROI.bin_y == 2 );

        app.m_currentROI.bin_y = 42;
        pcf::IndiProperty bad  = makeNumberProp( app.configName(), "not_roi_region_bin_y", 2 );
        REQUIRE( app.newCallBack_roi_bin_y( bad ) == -1 );
        REQUIRE( app.m_nextROI.bin_y == 42 );
    }
}

/// Verify the ROI action callbacks roi_check, roi_set, roi_full, roi_fullbin, roi_loadlast, roi_last, and roi_default.
/** Each callback must dispatch to checkNextROI() or setNextROI() only when the request element is On.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera ROI action callbacks", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "roi_check triggers checkNextROI on request=On" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_region_check", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_check( ip ) == 0 );
        REQUIRE( app.m_checkNextROICalls == 1 );

        pcf::IndiProperty off = makeSwitchProp( app.configName(), "roi_region_check", { { "request", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_roi_check( off ) == 0 );
        REQUIRE( app.m_checkNextROICalls == 1 );

        pcf::IndiProperty noReq( pcf::IndiProperty::Switch );
        noReq.setDevice( app.configName() );
        noReq.setName( "roi_region_check" );
        REQUIRE( app.newCallBack_roi_check( noReq ) == 0 );
        REQUIRE( app.m_checkNextROICalls == 1 );
    }

    SECTION( "roi_set triggers setNextROI and records the last ROI" )
    {
        app.m_currentROI.x    = 7;
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_set", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_set( ip ) == 0 );
        REQUIRE( app.m_setNextROICalls == 1 );
        REQUIRE( app.m_lastROI.x == 7 );
    }

    SECTION( "roi_full sets the next ROI to the full ROI values" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_set_full", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_full( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == app.m_full_x );
        REQUIRE( app.m_nextROI.w == app.m_full_w );
        REQUIRE( app.m_setNextROICalls == 1 );
    }

    SECTION( "roi_fullbin resets when the current-binning full ROI isn't implemented" )
    {
        REQUIRE( app.m_full_currbin_x == 0 );
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_set_full_bin", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_fullbin( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == app.m_full_x );
        REQUIRE( app.m_nextROI.bin_x == app.m_full_bin_x );
        REQUIRE( app.m_full_currbin_x == 0 ); // The value is reset to the not-implemented sentinel.
        REQUIRE( app.m_setNextROICalls == 1 );
    }

    SECTION( "roi_fullbin uses the current-binning full ROI when implemented" )
    {
        app.m_full_currbin_x = 100;
        app.m_full_currbin_y = 100;
        app.m_full_currbin_w = 400;
        app.m_full_currbin_h = 400;
        app.m_currentROI.bin_x = 2;
        app.m_currentROI.bin_y = 2;

        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_set_full_bin", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_fullbin( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == 100 );
        REQUIRE( app.m_nextROI.bin_x == 2 );
        REQUIRE( app.m_setNextROICalls == 1 );
    }

    SECTION( "roi_loadlast loads the last ROI without calling setNextROI" )
    {
        app.m_lastROI.x       = 55;
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_load_last", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_loadlast( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == 55 );
        REQUIRE( app.m_setNextROICalls == 0 );
    }

    SECTION( "roi_last swaps in the last ROI and calls setNextROI" )
    {
        app.m_lastROI.x        = 55;
        app.m_currentROI.x     = 9;
        pcf::IndiProperty ip  = makeSwitchProp( app.configName(), "roi_set_last", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_last( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == 55 );
        REQUIRE( app.m_lastROI.x == 9 );
        REQUIRE( app.m_setNextROICalls == 1 );
    }

    SECTION( "roi_default sets the next ROI to the configured defaults" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "roi_set_default", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_roi_default( ip ) == 0 );
        REQUIRE( app.m_nextROI.x == app.m_default_x );
        REQUIRE( app.m_setNextROICalls == 1 );
    }

    SECTION( "request=Off and a missing request element are no-ops for every ROI action callback" )
    {
        auto testNoOp = [&]( const std::string &propName, auto callback )
        {
            pcf::IndiProperty off = makeSwitchProp( app.configName(), propName, { { "request", pcf::IndiElement::Off } } );
            REQUIRE( ( app.*callback )( off ) == 0 );

            pcf::IndiProperty noReq( pcf::IndiProperty::Switch );
            noReq.setDevice( app.configName() );
            noReq.setName( propName );
            REQUIRE( ( app.*callback )( noReq ) == 0 );
        };

        testNoOp( "roi_set", &stdCameraFullHarness::newCallBack_roi_set );
        testNoOp( "roi_set_full", &stdCameraFullHarness::newCallBack_roi_full );
        testNoOp( "roi_set_full_bin", &stdCameraFullHarness::newCallBack_roi_fullbin );
        testNoOp( "roi_load_last", &stdCameraFullHarness::newCallBack_roi_loadlast );
        testNoOp( "roi_set_last", &stdCameraFullHarness::newCallBack_roi_last );
        testNoOp( "roi_set_default", &stdCameraFullHarness::newCallBack_roi_default );

        REQUIRE( app.m_setNextROICalls == 0 );
    }
}

/// Verify that stdCamera::newCallBack_shutter() opens the shutter on toggle On and shuts it on toggle Off through setShutter().
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_shutter", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "toggling on opens the shutter (ss=0)" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "shutter", { { "toggle", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_shutter( ip ) == 0 );
        REQUIRE( app.m_lastShutterSS == 0 );
    }

    SECTION( "toggling off shuts the shutter (ss=1)" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "shutter", { { "toggle", pcf::IndiElement::Off } } );
        REQUIRE( app.newCallBack_shutter( ip ) == 0 );
        REQUIRE( app.m_lastShutterSS == 1 );
    }

    SECTION( "a property with no toggle element is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "shutter" );
        REQUIRE( app.newCallBack_shutter( ip ) == 0 );
        REQUIRE( app.m_setShutterCalls == 0 );
    }
}

/// Verify that stdCamera::newCallBack_gotoFocus() calls gotoFocus() only on a request of On and only when focus is enabled.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera newCallBack_gotoFocus", "[dev::stdCamera]" )
{
    SECTION( "a request of On dispatches to gotoFocus" )
    {
        stdCameraFullHarness app;
        startFullHarness( app );

        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "goto_focus", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_gotoFocus( ip ) == 0 );
        REQUIRE( app.m_gotoFocusCalls == 1 );
    }

    SECTION( "a property with no request element is ignored" )
    {
        stdCameraFullHarness app;
        startFullHarness( app );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( app.configName() );
        ip.setName( "goto_focus" );
        REQUIRE( app.newCallBack_gotoFocus( ip ) == 0 );
        REQUIRE( app.m_gotoFocusCalls == 0 );
    }

    SECTION( "when focus is not enabled at runtime, the callback is a no-op" )
    {
        stdCameraFullHarness app;
        REQUIRE( app.m_hasFocus == false ); // This is a fresh harness. loadConfig() has not run yet.

        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "goto_focus", { { "request", pcf::IndiElement::On } } );
        REQUIRE( app.newCallBack_gotoFocus( ip ) == 0 );
        REQUIRE( app.m_gotoFocusCalls == 0 );
    }
}

/// Verify that stdCamera::setCallBack_focusMonitored() caches the monitored property and that checkFocusSwitchState() reads the focus state from it.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera focus-state helper", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    // writeFullValidConfig() sets focus.stateProperty to sre.caution with element focus-mismatch and
    // onMeansInFocus set to false. That property is monitored property index 0.
    REQUIRE( app.m_focusStateSourceIndex == 0 );

    SECTION( "an unrecognized monitored property is ignored" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "unrelated" );
        ip.setName( "property" );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
    }

    SECTION( "On means out of focus by default (onMeansInFocus=false)" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "sre" );
        ip.setName( "caution" );
        ip.add( pcf::IndiElement( "focus-mismatch", pcf::IndiElement::On ) );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
        REQUIRE( app.checkFocusSwitchState() == false );
        REQUIRE( app.m_indiP_focus["state"].getSwitchState() == pcf::IndiElement::Off );

        ip["focus-mismatch"].setSwitchState( pcf::IndiElement::Off );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
        REQUIRE( app.checkFocusSwitchState() == true );
        REQUIRE( app.m_indiP_focus["state"].getSwitchState() == pcf::IndiElement::On );
    }

    SECTION( "checkFocusSwitchState returns false when the element is missing" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "sre" );
        ip.setName( "caution" );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
        REQUIRE( app.checkFocusSwitchState() == false );
    }

    SECTION( "checkFocusSwitchState returns false when the helper is not configured" )
    {
        app.m_focusStateHelperConfigured = false;
        REQUIRE( app.checkFocusSwitchState() == false );
    }

    SECTION( "checkFocusSwitchState returns false when the source index is out of range" )
    {
        app.m_focusStateSourceIndex = 99;
        REQUIRE( app.checkFocusSwitchState() == false );
    }

    SECTION( "updateFocusStateProperty is a no-op when the focus state element is missing" )
    {
        // Drop the state element that appStartup() added. The monitored-property callback then
        // reaches updateFocusStateProperty(), which must return early and leave m_indiP_focus alone.
        app.m_indiP_focus.remove( "state" );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "sre" );
        ip.setName( "caution" );
        ip.add( pcf::IndiElement( "focus-mismatch", pcf::IndiElement::Off ) );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
        REQUIRE( app.m_indiP_focus.find( "state" ) == false );
    }

    SECTION( "On means in focus when onMeansInFocus=true" )
    {
        app.m_focusStateOnMeansInFocus = true;

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "sre" );
        ip.setName( "caution" );
        ip.add( pcf::IndiElement( "focus-mismatch", pcf::IndiElement::On ) );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
        REQUIRE( app.checkFocusSwitchState() == true );

        ip["focus-mismatch"].setSwitchState( pcf::IndiElement::Off );
        REQUIRE( app.setCallBack_focusMonitored( ip ) == 0 );
        REQUIRE( app.checkFocusSwitchState() == false );
    }
}

/// Verify that loadConfig() falls back to the "toggle" default when focus.stateElement is configured as an empty string.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera loadConfig defaults an empty focus.stateElement to toggle", "[dev::stdCamera]" )
{
    std::vector<std::string> s, k, v;
    auto                     add = [&]( const std::string &sec, const std::string &key, const std::string &val )
    {
        s.push_back( sec );
        k.push_back( key );
        v.push_back( val );
    };
    add( "camera", "full_x", "511.5" );
    add( "camera", "full_y", "511.5" );
    add( "camera", "full_w", "1024" );
    add( "camera", "full_h", "1024" );
    add( "camera", "full_bin_x", "1" );
    add( "camera", "full_bin_y", "1" );
    add( "focus", "stateElement", "" ); // The value is present but empty.

    mx::app::writeConfigFile( "/tmp/stdCamera_focus_emptyStateElement.conf", s, k, v );

    stdCameraFullHarness app;
    mx::app::appConfigurator config;
    REQUIRE( app.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/stdCamera_focus_emptyStateElement.conf" );
    REQUIRE( app.loadConfig( config ) == 0 );
    REQUIRE( app.m_focusStateElement == "toggle" );
}

/// Verify that stdCamera::sendGotoFocusCommand() formats the active switch names into a target preset and sends it.
/** The harness captures the outgoing property in m_lastSentProperty instead of sending it to INDI.
 * Every rejection branch is checked as well.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera sendGotoFocusCommand", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    // writeFullValidConfig() configures a goto-focus helper with two switches. property1 is
    // stagebs.presetName and property2 is fwfpm.filterName. The format is "{}-{}" and the
    // targetProperty is stagesci1.presetName.
    REQUIRE( app.m_focusGotoHelperConfigured == true );

    pcf::IndiProperty prop1( pcf::IndiProperty::Switch );
    prop1.setDevice( "stagebs" );
    prop1.setName( "presetName" );
    prop1.add( pcf::IndiElement( "65-35", pcf::IndiElement::On ) );
    prop1.add( pcf::IndiElement( "ha-ir", pcf::IndiElement::Off ) );

    pcf::IndiProperty prop2( pcf::IndiProperty::Switch );
    prop2.setDevice( "fwfpm" );
    prop2.setName( "filterName" );
    prop2.add( pcf::IndiElement( "open", pcf::IndiElement::On ) );
    prop2.add( pcf::IndiElement( "lyotsm", pcf::IndiElement::Off ) );

    SECTION( "a successful dispatch sends the formatted preset selection" )
    {
        REQUIRE( app.setCallBack_focusMonitored( prop1 ) == 0 );
        REQUIRE( app.setCallBack_focusMonitored( prop2 ) == 0 );

        REQUIRE( app.sendGotoFocusCommand() == 0 );
        REQUIRE( app.m_lastSentProperty.getDevice() == "stagesci1" );
        REQUIRE( app.m_lastSentProperty.getName() == "presetName" );
        REQUIRE( app.m_lastSentProperty.find( "65-35-open" ) );
        REQUIRE( app.m_lastSentProperty["65-35-open"].getSwitchState() == pcf::IndiElement::On );
    }

    SECTION( "dispatch failures are propagated" )
    {
        REQUIRE( app.setCallBack_focusMonitored( prop1 ) == 0 );
        REQUIRE( app.setCallBack_focusMonitored( prop2 ) == 0 );
        app.m_sendNewPropertyResult = -1;
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }

    SECTION( "not configured is rejected" )
    {
        app.m_focusGotoHelperConfigured = false;
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }

    SECTION( "an out-of-range source index is rejected" )
    {
        app.m_focusGotoSourceIndices[0] = 99;
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }

    SECTION( "no active switch element is rejected" )
    {
        pcf::IndiProperty noneActive( pcf::IndiProperty::Switch );
        noneActive.setDevice( "stagebs" );
        noneActive.setName( "presetName" );
        noneActive.add( pcf::IndiElement( "65-35", pcf::IndiElement::Off ) );
        REQUIRE( app.setCallBack_focusMonitored( noneActive ) == 0 );
        REQUIRE( app.setCallBack_focusMonitored( prop2 ) == 0 );
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }

    SECTION( "multiple active switch elements are rejected" )
    {
        pcf::IndiProperty twoActive( pcf::IndiProperty::Switch );
        twoActive.setDevice( "stagebs" );
        twoActive.setName( "presetName" );
        twoActive.add( pcf::IndiElement( "65-35", pcf::IndiElement::On ) );
        twoActive.add( pcf::IndiElement( "ha-ir", pcf::IndiElement::On ) );
        REQUIRE( app.setCallBack_focusMonitored( twoActive ) == 0 );
        REQUIRE( app.setCallBack_focusMonitored( prop2 ) == 0 );
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }

    SECTION( "an empty formatted target element is rejected" )
    {
        REQUIRE( app.setCallBack_focusMonitored( prop1 ) == 0 );
        REQUIRE( app.setCallBack_focusMonitored( prop2 ) == 0 );
        app.m_focusGotoFormat = "";
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }
}

/// Call the trueFalseT tag-dispatch overloads and the protected creator helpers directly.
/** stdCamera selects these overloads from the compile-time capability flags of the derived class.
 * Any single harness can therefore only reach one overload of each pair through the normal call
 * sites. The other overload is dead code for that harness. Calling both directly covers them all.
 *
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera tag-dispatch and creator helpers", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    mx::meta::trueFalseT<true>  t;
    mx::meta::trueFalseT<false> f;

    SECTION( "the false overloads of every setter are no-ops returning 0/false" )
    {
        REQUIRE( app.setTempSetPt( f ) == 0 );
        REQUIRE( app.setTempControl( f ) == 0 );
        REQUIRE( app.setReadoutSpeed( f ) == 0 );
        REQUIRE( app.setVShiftSpeed( f ) == 0 );
        REQUIRE( app.setEMGain( f ) == 0 );
        REQUIRE( app.setExpTime( f ) == 0 );
        REQUIRE( app.setFPS( f ) == 0 );
        REQUIRE( app.setFanSpeed( f ) == 0 );
        REQUIRE( app.setAnalogGain( f ) == 0 );
        REQUIRE( app.setLED( f ) == 0 );
        REQUIRE( app.setSynchro( f ) == 0 );
        REQUIRE( app.setCropMode( f ) == 0 );
        REQUIRE( app.checkNextROI( f ) == 0 );
        REQUIRE( app.setNextROI( f ) == 0 );
        REQUIRE( app.setShutter( 0, f ) == 0 );
        REQUIRE( app.checkFocus( f ) == false );
        REQUIRE( app.gotoFocus( f ) == 0 );
        REQUIRE( app.stateString( f ) == "" );
        REQUIRE( app.stateStringValid( f ) == false );
    }

    SECTION( "the true overloads dispatch to the derived class implementation" )
    {
        REQUIRE( app.setTempSetPt( t ) == 0 );
        REQUIRE( app.m_setTempSetPtCalls == 1 );

        REQUIRE( app.setTempControl( t ) == 0 );
        REQUIRE( app.m_setTempControlCalls == 1 );

        REQUIRE( app.setReadoutSpeed( t ) == 0 );
        REQUIRE( app.m_setReadoutSpeedCalls == 1 );

        REQUIRE( app.setVShiftSpeed( t ) == 0 );
        REQUIRE( app.m_setVShiftSpeedCalls == 1 );

        REQUIRE( app.setEMGain( t ) == 0 );
        REQUIRE( app.m_setEMGainCalls == 1 );

        REQUIRE( app.setExpTime( t ) == 0 );
        REQUIRE( app.m_setExpTimeCalls == 1 );

        REQUIRE( app.setFPS( t ) == 0 );
        REQUIRE( app.m_setFPSCalls == 1 );

        REQUIRE( app.setFanSpeed( t ) == 0 );
        REQUIRE( app.m_setFanSpeedCalls == 1 );

        REQUIRE( app.setAnalogGain( t ) == 0 );
        REQUIRE( app.m_setAnalogGainCalls == 1 );

        REQUIRE( app.setLED( t ) == 0 );
        REQUIRE( app.m_setLEDCalls == 1 );

        REQUIRE( app.setSynchro( t ) == 0 );
        REQUIRE( app.m_setSynchroCalls == 1 );

        REQUIRE( app.setCropMode( t ) == 0 );
        REQUIRE( app.m_setCropModeCalls == 1 );

        REQUIRE( app.checkNextROI( t ) == 0 );
        REQUIRE( app.m_checkNextROICalls == 1 );

        REQUIRE( app.setNextROI( t ) == 0 );
        REQUIRE( app.m_setNextROICalls == 1 );

        REQUIRE( app.setShutter( 0, t ) == 0 );
        REQUIRE( app.m_setShutterCalls == 1 );

        REQUIRE( app.checkFocus( t ) == false );

        REQUIRE( app.gotoFocus( t ) == 0 );
        REQUIRE( app.m_gotoFocusCalls == 1 );

        REQUIRE( app.stateString( t ) == app.m_stateStringResult );
        REQUIRE( app.stateStringValid( t ) == app.m_stateStringValidResult );
    }

    SECTION( "the creator overloads that appStartup() never calls can be invoked directly" )
    {
        // The false overloads sit behind the capability guards in appStartup(), so no lifecycle path
        // runs them. The true overloads of createReadoutSpeed() and createVShiftSpeed() already ran
        // in startFullHarness().
        REQUIRE( app.exposeCreateReadoutSpeedFalse() == 0 );
        REQUIRE( app.exposeCreateVShiftSpeedFalse() == 0 );
        REQUIRE( app.exposeCreateFanSpeedFalse() == 0 );

        // The true overload of createFanSpeed() is never reached in stdCamera. appStartup() builds the
        // fan-speed selection switch inline instead of through this helper. Call it directly so that
        // the fan_speed selection switch is also registered through this code path.
        REQUIRE( app.exposeCreateFanSpeedTrue() == 0 );
    }
}

/// Verify that the static wrappers st_newCallBack_stdCamera() and st_setCallBack_focusMonitored() forward to the instance methods.
/**
 * \ingroup stdCamera_tests
 */
TEST_CASE( "stdCamera static callback wrappers", "[dev::stdCamera]" )
{
    stdCameraFullHarness app;
    startFullHarness( app );

    SECTION( "st_newCallBack_stdCamera forwards to the instance method" )
    {
        pcf::IndiProperty ip = makeSwitchProp( app.configName(), "reconfigure", { { "request", pcf::IndiElement::On } } );
        REQUIRE( stdCameraFullHarness::st_newCallBack_stdCamera( &app, ip ) == 0 );
        REQUIRE( app.m_reconfig == true );
    }

    SECTION( "st_setCallBack_focusMonitored forwards to the instance method" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "sre" );
        ip.setName( "caution" );
        ip.add( pcf::IndiElement( "focus-mismatch", pcf::IndiElement::On ) );
        REQUIRE( stdCameraFullHarness::st_setCallBack_focusMonitored( &app, ip ) == 0 );
    }
}

