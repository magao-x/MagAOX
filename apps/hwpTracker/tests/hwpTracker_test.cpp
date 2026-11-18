/** \file hwpTracker_test.cpp
 * \brief Catch2 tests for the hwpTracker app.
 *
 * \ingroup hwpTracker_files
 */

#include "../../../tests/catch2/catch.hpp"
#include "../hwpTracker.hpp"
#include "../../../tests/testMacrosINDI.hpp"
#include "../../../tests/testXWC.hpp"

#include <chrono>
#include <filesystem>
#include <limits>
#include <thread>

/** \defgroup hwpTracker_unit_test hwpTracker Unit Tests
 * \brief Unit tests for the hwpTracker application.
 *
 * \ingroup application_unit_test
 */

namespace libXWCTest
{

/// Namespace for `hwpTracker` unit tests.
/** \ingroup hwpTracker_unit_test
 */
namespace hwpTrackerTest
{

using MagAOX::app::stateCodes;

namespace
{
/// Build a per-test scratch directory under /tmp.
std::string testRoot( const std::string &name )
{
    std::string root = "/tmp/hwpTracker_test/" + name;
    std::filesystem::create_directories( root );
    return root;
}

} // namespace

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test harness recording stage commands instead of sending them.
class hwpTracker_test : public MagAOX::app::hwpTracker
{
  public:
    std::vector<float> m_sent; ///< Stage targets sent, in order.

    using MagAOX::app::hwpTracker::m_altitude;
    using MagAOX::app::hwpTracker::m_devName;
    using MagAOX::app::hwpTracker::m_haveStageState;
    using MagAOX::app::hwpTracker::m_haveTelData;
    using MagAOX::app::hwpTracker::m_hwpActualPos;
    using MagAOX::app::hwpTracker::m_hwpCurPos;
    using MagAOX::app::hwpTracker::m_hwpPosName;
    using MagAOX::app::hwpTracker::m_hwpSetPos;
    using MagAOX::app::hwpTracker::m_hwpTrackingOffset;
    using MagAOX::app::hwpTracker::m_lastUpdate;
    using MagAOX::app::hwpTracker::m_movePending;
    using MagAOX::app::hwpTracker::m_parang;
    using MagAOX::app::hwpTracker::m_pupilOffset;
    using MagAOX::app::hwpTracker::m_shutdown;
    using MagAOX::app::hwpTracker::m_sign;
    using MagAOX::app::hwpTracker::m_stageState;
    using MagAOX::app::hwpTracker::m_tcsDevName;
    using MagAOX::app::hwpTracker::m_telemCurPos;
    using MagAOX::app::hwpTracker::m_telemTracking;
    using MagAOX::app::hwpTracker::m_tracking;
    using MagAOX::app::hwpTracker::m_trackOffsetSign;
    using MagAOX::app::hwpTracker::m_updateInterval;
    using MagAOX::app::hwpTracker::m_zero;

    using MagAOX::app::hwpTracker::recordPolTrack;
    using MagAOX::app::hwpTracker::requestMove;
    using MagAOX::app::hwpTracker::trackerState;

    /// Construct a testable hwpTracker with the given device name.
    explicit hwpTracker_test( const std::string &device )
    {
        m_configName = device;

        std::string root = testRoot( "telem" );

        m_tel.logPath( root );
        m_tel.logExt( "bintel" );
        m_tel.logName( m_configName );
        m_tel.m_logLevel = logPrio::LOG_TELEM;
        m_maxInterval    = 3600.0;

        XWCTEST_SETUP_INDI_NEW_PROP( tracking );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_hwpSetPos, hwp_position );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_teldata, tcsi, teldata );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stagePos, stagepolrot, position );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stageFsm, stagepolrot, fsm );
    }

    /// Run setupConfig.
    void setupTestConfig()
    {
        setupConfig();
    }

    /// Read a config file into the app's configurator.
    void readConfigFile( const std::string &path )
    {
        config.readConfig( path );
    }

    /// Run loadConfigImpl on the app's configurator.
    int loadOwnConfig()
    {
        return loadConfigImpl( config );
    }

    /// Run loadConfig on the app's configurator.
    void loadTestConfig()
    {
        loadConfig();
    }

    /// Run appLogic, starting the telemetry thread it requires.
    int logic()
    {
        if( !m_tel.logThreadRunning() )
        {
            m_tel.m_writePause = 1000000;
            m_tel.logThreadStart();

            for( unsigned attempt = 0; attempt < 1000 && !m_tel.logThreadRunning(); ++attempt )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
        }

        return appLogic();
    }

    /// Report a stage state through the fsm callback.
    int stageState( const std::string &state )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "stagepolrot" );
        ip.setName( "fsm" );
        ip.add( pcf::IndiElement( "state" ) );
        ip["state"].set( state );

        return setCallBack_m_indiP_stageFsm( ip );
    }

    /// Report a stage position through the position callback.
    int stagePos( float pos )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "stagepolrot" );
        ip.setName( "position" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].set( pos );

        return setCallBack_m_indiP_stagePos( ip );
    }

    /// Report telescope data through the teldata callback.
    int telData( float zd, float pa )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "tcsi" );
        ip.setName( "teldata" );
        ip.add( pcf::IndiElement( "zd" ) );
        ip["zd"].set( zd );
        ip.add( pcf::IndiElement( "pa" ) );
        ip["pa"].set( pa );

        return setCallBack_m_indiP_teldata( ip );
    }

    /// Request a new HWP angle through the hwp_position callback.
    int setHwp( float target )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_configName );
        ip.setName( "hwp_position" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( target );

        return newCallBack_m_indiP_hwpSetPos( ip );
    }

    /// Switch tracking through the tracking callback.
    int setTracking( bool on )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( m_configName );
        ip.setName( "tracking" );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );

        return newCallBack_m_indiP_tracking( ip );
    }

  protected:
    int sendStageTarget( float target ) override
    {
        m_sent.push_back( target );
        return 0;
    }
};
/// \endcond

/// Write a config file and load it into a fresh app, returning loadConfigImpl's result.
int loadFromConfig( const std::string              &name,
                    const std::vector<std::string> &sections,
                    const std::vector<std::string> &keywords,
                    const std::vector<std::string> &values,
                    hwpTracker_test                &app )
{
    std::string root = testRoot( name );
    mx::app::writeConfigFile( root + "/hwptrack.conf", sections, keywords, values );

    app.setupTestConfig();
    app.readConfigFile( root + "/hwptrack.conf" );

    return app.loadOwnConfig();
}

/// Validate the hwpTracker INDI callbacks reject wrong devices and names.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker INDI callbacks validate their source properties", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::newCallBack_m_indiP_hwpSetPos( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::setCallBack_m_indiP_teldata( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::setCallBack_m_indiP_stagePos( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::setCallBack_m_indiP_stageFsm( pcf::IndiProperty() );
#endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( hwpTracker, tracking );
    XWCTEST_INDI_ARBNEW_CALLBACK( hwpTracker, newCallBack_m_indiP_hwpSetPos, hwp_position );
    XWCTEST_INDI_SET_CALLBACK( hwpTracker, m_indiP_teldata, tcsi, teldata );
    XWCTEST_INDI_SET_CALLBACK( hwpTracker, m_indiP_stagePos, stagepolrot, position );
    XWCTEST_INDI_SET_CALLBACK( hwpTracker, m_indiP_stageFsm, stagepolrot, fsm );
}

/// Validate hwpTracker configuration defaults, overrides and validation.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker configuration loading", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::setupConfig();
    MagAOX::app::hwpTracker::loadConfigImpl( mx::app::appConfigurator() );
    MagAOX::app::hwpTracker::loadConfig();
#endif
    // clang-format on

    SECTION( "defaults" )
    {
        hwpTracker_test app( "hwptrack" );

        REQUIRE( loadFromConfig( "config_defaults", {}, {}, {}, app ) == 0 );
        REQUIRE( app.m_zero == Approx( 360 ) );
        REQUIRE( app.m_sign == -1 );
        REQUIRE( app.m_devName == "stagepolrot" );
        REQUIRE( app.m_tcsDevName == "tcsi" );
        REQUIRE( app.m_updateInterval == Approx( 10 ) );
        REQUIRE( app.m_trackOffsetSign == -1 );
        REQUIRE( app.m_pupilOffset == Approx( 0 ) );
    }

    SECTION( "overrides, including tracking.sign" )
    {
        hwpTracker_test app( "hwptrack" );

        REQUIRE( loadFromConfig( "config_overrides",
                                 { "hwp", "hwp", "hwp", "tcs", "tracking", "tracking", "tracking" },
                                 { "zero", "sign", "devName", "devName", "updateInterval", "sign", "pupilOffset" },
                                 { "10", "1", "stagehwp", "tcs2", "5", "1", "3.5" },
                                 app ) == 0 );
        REQUIRE( app.m_zero == Approx( 10 ) );
        REQUIRE( app.m_sign == 1 );
        REQUIRE( app.m_devName == "stagehwp" );
        REQUIRE( app.m_tcsDevName == "tcs2" );
        REQUIRE( app.m_updateInterval == Approx( 5 ) );
        REQUIRE( app.m_trackOffsetSign == 1 );
        REQUIRE( app.m_pupilOffset == Approx( 3.5 ) );
    }

    SECTION( "invalid values are rejected" )
    {
        std::vector<std::vector<std::string>> bad = { { "hwp", "sign", "2" },
                                                      { "hwp", "sign", "0" },
                                                      { "tracking", "sign", "0" },
                                                      { "tracking", "sign", "-2" },
                                                      { "tracking", "updateInterval", "0" },
                                                      { "tracking", "updateInterval", "-1" } };

        for( size_t n = 0; n < bad.size(); ++n )
        {
            INFO( bad[n][0] + "." + bad[n][1] + "=" + bad[n][2] );

            hwpTracker_test app( "hwptrack" );

            REQUIRE( loadFromConfig(
                         "config_bad" + std::to_string( n ), { bad[n][0] }, { bad[n][1] }, { bad[n][2] }, app ) == -1 );
        }
    }

    SECTION( "non-finite angles and empty device names are rejected" )
    {
        // Set directly, since these values may not survive the config file parser.
        hwpTracker_test zero( "hwptrack" );
        zero.m_zero = std::numeric_limits<float>::quiet_NaN();
        REQUIRE( loadFromConfig( "config_zero", {}, {}, {}, zero ) == -1 );

        hwpTracker_test pupil( "hwptrack" );
        pupil.m_pupilOffset = std::numeric_limits<float>::infinity();
        REQUIRE( loadFromConfig( "config_pupil", {}, {}, {}, pupil ) == -1 );

        hwpTracker_test interval( "hwptrack" );
        interval.m_updateInterval = std::numeric_limits<float>::quiet_NaN();
        REQUIRE( loadFromConfig( "config_interval", {}, {}, {}, interval ) == -1 );

        hwpTracker_test dev( "hwptrack" );
        dev.m_devName = "";
        REQUIRE( loadFromConfig( "config_dev", {}, {}, {}, dev ) == -1 );

        hwpTracker_test tcs( "hwptrack" );
        tcs.m_tcsDevName = "";
        REQUIRE( loadFromConfig( "config_tcs", {}, {}, {}, tcs ) == -1 );
    }

    SECTION( "loadConfig shuts down on an invalid configuration" )
    {
        hwpTracker_test app( "hwptrack" );

        std::string root = testRoot( "config_shutdown" );
        mx::app::writeConfigFile( root + "/hwptrack.conf", { "hwp" }, { "sign" }, { "3" } );

        app.setupTestConfig();
        app.readConfigFile( root + "/hwptrack.conf" );
        app.loadTestConfig();

        REQUIRE( app.m_shutdown == 1 );
    }
}

/// Validate the tracking offset, stage target and angle helpers.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker angle calculations", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::trackingOffset( 0, 0, 0, 0 );
    MagAOX::app::hwpTracker::stageTarget( 0, 0, 0 );
    MagAOX::app::hwpTracker::nearestAngle( 0, 0 );
    MagAOX::app::hwpTracker::positionName( 0 );
    MagAOX::app::hwpTracker::isFinite( 0 );
#endif
    // clang-format on

    using MagAOX::app::hwpTracker;

    SECTION( "tracking offset for both signs and a pupil offset" )
    {
        REQUIRE( hwpTracker::trackingOffset( 60, 40, -1, 0 ) == Approx( 40 ) );
        REQUIRE( hwpTracker::trackingOffset( 60, 40, 1, 0 ) == Approx( 80 ) );
        REQUIRE( hwpTracker::trackingOffset( 60, 40, -1, 5 ) == Approx( 45 ) );
    }

    SECTION( "stage target applies zero and sign and wraps into [0, 360)" )
    {
        REQUIRE( hwpTracker::stageTarget( 22.5, 360, -1 ) == Approx( 337.5 ) );
        REQUIRE( hwpTracker::stageTarget( -22.5, 360, -1 ) == Approx( 22.5 ) );
        REQUIRE( hwpTracker::stageTarget( 0, 360, -1 ) == Approx( 0 ) );
        REQUIRE( hwpTracker::stageTarget( 400, 360, -1 ) == Approx( 320 ) );
        REQUIRE( hwpTracker::stageTarget( 50, 10, 1 ) == Approx( 60 ) );
        REQUIRE( hwpTracker::stageTarget( -50, 10, 1 ) == Approx( 320 ) );
        REQUIRE( hwpTracker::stageTarget( 700, 10, 1 ) == Approx( 350 ) );

        for( float pos = -720; pos <= 720; pos += 0.1 )
        {
            float target = hwpTracker::stageTarget( pos, 360, -1 );

            REQUIRE( target >= 0 );
            REQUIRE( target < 360 );
        }
    }

    SECTION( "nearest angle shifts by multiples of 360" )
    {
        REQUIRE( hwpTracker::nearestAngle( 1, 359 ) == Approx( 361 ) );
        REQUIRE( hwpTracker::nearestAngle( 359, 1 ) == Approx( -1 ) );
        REQUIRE( hwpTracker::nearestAngle( 330, -30 ) == Approx( -30 ) );
        REQUIRE( hwpTracker::nearestAngle( 10, 20 ) == Approx( 10 ) );
        REQUIRE( hwpTracker::nearestAngle( 5, 725 ) == Approx( 725 ) );
    }

    SECTION( "finiteness check works under -ffast-math" )
    {
        REQUIRE( hwpTracker::isFinite( 0 ) );
        REQUIRE( hwpTracker::isFinite( -360.5 ) );
        REQUIRE( hwpTracker::isFinite( std::numeric_limits<float>::max() ) );
        REQUIRE_FALSE( hwpTracker::isFinite( std::numeric_limits<float>::quiet_NaN() ) );
        REQUIRE_FALSE( hwpTracker::isFinite( std::numeric_limits<float>::infinity() ) );
        REQUIRE_FALSE( hwpTracker::isFinite( -std::numeric_limits<float>::infinity() ) );
    }

    SECTION( "position names match modulo 90" )
    {
        REQUIRE( hwpTracker::positionName( 0 ) == "Qplus" );
        REQUIRE( hwpTracker::positionName( 45 ) == "Qminus" );
        REQUIRE( hwpTracker::positionName( 22.5 ) == "Uplus" );
        REQUIRE( hwpTracker::positionName( 67.5 ) == "Uminus" );
        REQUIRE( hwpTracker::positionName( 90 ) == "Qplus" );
        REQUIRE( hwpTracker::positionName( 180 ) == "Qplus" );
        REQUIRE( hwpTracker::positionName( -22.5 ) == "Uminus" );
        REQUIRE( hwpTracker::positionName( -45 ) == "Qminus" );
        REQUIRE( hwpTracker::positionName( 112.5 ) == "Uplus" );
        REQUIRE( hwpTracker::positionName( 0.4 ) == "Qplus" );
        REQUIRE( hwpTracker::positionName( 89.6 ) == "Qplus" );
        REQUIRE( hwpTracker::positionName( 0.6 ) == "Unknown" );
        REQUIRE( hwpTracker::positionName( 10 ) == "Unknown" );
    }
}

/// Validate that stage moves are sent only when the stage is READY and are otherwise queued.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker queues moves until the stage is READY", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::requestMove();
    MagAOX::app::hwpTracker::newCallBack_m_indiP_hwpSetPos( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::setCallBack_m_indiP_stageFsm( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::appLogic();
#endif
    // clang-format on

    hwpTracker_test app( "hwptrack" );

    SECTION( "a move to a READY stage is sent at once" )
    {
        REQUIRE( app.stageState( "READY" ) == 0 );
        REQUIRE( app.setHwp( 22.5 ) == 0 );

        REQUIRE( app.m_hwpSetPos == Approx( 22.5 ) );
        REQUIRE( app.m_sent.size() == 1 );
        REQUIRE( app.m_sent[0] == Approx( 337.5 ) );
        REQUIRE_FALSE( app.m_movePending );
    }

    SECTION( "a move before the stage reports is queued and sent on READY" )
    {
        REQUIRE( app.setHwp( 22.5 ) == 0 );

        REQUIRE( app.m_sent.empty() );
        REQUIRE( app.m_movePending );

        REQUIRE( app.stageState( "READY" ) == 0 );

        REQUIRE( app.m_sent.size() == 1 );
        REQUIRE( app.m_sent[0] == Approx( 337.5 ) );
        REQUIRE_FALSE( app.m_movePending );
    }

    SECTION( "the latest request wins while the stage is moving" )
    {
        REQUIRE( app.stageState( "OPERATING" ) == 0 );
        REQUIRE( app.setHwp( 10 ) == 0 );
        REQUIRE( app.setHwp( 45 ) == 0 );

        REQUIRE( app.m_sent.empty() );

        REQUIRE( app.stageState( "READY" ) == 0 );

        REQUIRE( app.m_sent.size() == 1 );
        REQUIRE( app.m_sent[0] == Approx( 315 ) );
    }

    SECTION( "a queued move waits through a stage error" )
    {
        REQUIRE( app.stageState( "ERROR" ) == 0 );
        REQUIRE( app.setHwp( 45 ) == 0 );
        REQUIRE( app.stageState( "NOTCONNECTED" ) == 0 );

        REQUIRE( app.m_sent.empty() );
        REQUIRE( app.m_movePending );

        REQUIRE( app.stageState( "READY" ) == 0 );

        REQUIRE( app.m_sent.size() == 1 );
    }

    SECTION( "appLogic sends a queued move if the stage is READY" )
    {
        app.m_movePending = true;
        app.m_stageState  = stateCodes::READY;

        REQUIRE( app.logic() == 0 );

        REQUIRE( app.m_sent.size() == 1 );
        REQUIRE_FALSE( app.m_movePending );
    }

    SECTION( "an unrecognized stage state is treated as not ready" )
    {
        REQUIRE( app.stageState( "READY" ) == 0 );
        REQUIRE( app.stageState( "BOGUS" ) == 0 );

        REQUIRE( app.m_stageState == stateCodes::NOTCONNECTED );

        REQUIRE( app.setHwp( 45 ) == 0 );

        REQUIRE( app.m_sent.empty() );
        REQUIRE( app.m_movePending );
    }
}

/// Validate tracking start, periodic updates, stop, and the teldata requirements.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker tracking", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::setCallBack_m_indiP_teldata( pcf::IndiProperty() );
    MagAOX::app::hwpTracker::appLogic();
#endif
    // clang-format on

    hwpTracker_test app( "hwptrack" );

    REQUIRE( app.stageState( "READY" ) == 0 );

    SECTION( "tracking is refused before valid teldata" )
    {
        REQUIRE( app.setTracking( true ) == 0 );

        REQUIRE_FALSE( app.m_tracking );
        REQUIRE( app.m_sent.empty() );
    }

    SECTION( "invalid teldata is ignored and the last good values kept" )
    {
        REQUIRE( app.telData( 95, 10 ) == 0 );
        REQUIRE( app.telData( -1, 10 ) == 0 );
        REQUIRE_FALSE( app.m_haveTelData );

        REQUIRE( app.telData( 30, 40 ) == 0 );
        REQUIRE( app.m_haveTelData );
        REQUIRE( app.m_altitude == Approx( 60 ) );
        REQUIRE( app.m_parang == Approx( 40 ) );

        REQUIRE( app.telData( 95, 10 ) == 0 );
        REQUIRE( app.m_altitude == Approx( 60 ) );
        REQUIRE( app.m_parang == Approx( 40 ) );
    }

    SECTION( "starting tracking applies the offset, stopping removes it" )
    {
        REQUIRE( app.telData( 30, 40 ) == 0 ); // offset = -0.5*40 + 60 = 40

        REQUIRE( app.setTracking( true ) == 0 );

        REQUIRE( app.m_tracking );
        REQUIRE( app.m_hwpTrackingOffset == Approx( 40 ) );
        REQUIRE( app.m_sent.size() == 1 );
        REQUIRE( app.m_sent[0] == Approx( 320 ) );

        REQUIRE( app.setTracking( false ) == 0 );

        REQUIRE_FALSE( app.m_tracking );
        REQUIRE( app.m_hwpTrackingOffset == Approx( 0 ) );
        REQUIRE( app.m_sent.size() == 2 );
        REQUIRE( app.m_sent[1] == Approx( 0 ) );
    }

    SECTION( "periodic updates are skipped while the stage is not READY" )
    {
        REQUIRE( app.telData( 30, 40 ) == 0 );
        REQUIRE( app.setTracking( true ) == 0 );
        REQUIRE( app.m_sent.size() == 1 );

        REQUIRE( app.logic() == 0 );
        REQUIRE( app.m_sent.size() == 1 ); // interval not elapsed

        REQUIRE( app.telData( 30, 42 ) == 0 );
        app.m_lastUpdate = 0;

        REQUIRE( app.stageState( "OPERATING" ) == 0 );
        REQUIRE( app.logic() == 0 );
        REQUIRE( app.m_sent.size() == 1 );

        REQUIRE( app.stageState( "READY" ) == 0 );
        REQUIRE( app.m_sent.size() == 1 ); // nothing queued

        REQUIRE( app.logic() == 0 );
        REQUIRE( app.m_sent.size() == 2 );
        REQUIRE( app.m_hwpTrackingOffset == Approx( 39 ) );
    }

    SECTION( "parallactic angle is unwrapped across 0/360 while tracking" )
    {
        REQUIRE( app.telData( 30, 359 ) == 0 );
        REQUIRE( app.setTracking( true ) == 0 );

        REQUIRE( app.telData( 30, 1 ) == 0 );
        REQUIRE( app.m_parang == Approx( 361 ) );

        REQUIRE( app.telData( 30, 359.5 ) == 0 );
        REQUIRE( app.m_parang == Approx( 359.5 ) );

        REQUIRE( app.telData( 30, 0.5 ) == 0 );
        REQUIRE( app.telData( 30, 2 ) == 0 );
        REQUIRE( app.m_parang == Approx( 362 ) );

        app.m_lastUpdate = 0;
        REQUIRE( app.logic() == 0 );
        REQUIRE( app.m_hwpTrackingOffset == Approx( -0.5 * 362 + 60 ) );
    }

    SECTION( "unwrapping resets when tracking restarts" )
    {
        REQUIRE( app.telData( 30, 359 ) == 0 );
        REQUIRE( app.setTracking( true ) == 0 );
        REQUIRE( app.telData( 30, 1 ) == 0 );
        REQUIRE( app.m_parang == Approx( 361 ) );

        REQUIRE( app.setTracking( false ) == 0 );
        REQUIRE( app.telData( 30, 1 ) == 0 );
        REQUIRE( app.m_parang == Approx( 1 ) );

        REQUIRE( app.setTracking( true ) == 0 );
        REQUIRE( app.telData( 30, 2 ) == 0 );
        REQUIRE( app.m_parang == Approx( 2 ) );
    }
}

/// Validate the stage readback, reported angles and position names.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker stage readback", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::setCallBack_m_indiP_stagePos( pcf::IndiProperty() );
#endif
    // clang-format on

    hwpTracker_test app( "hwptrack" );

    REQUIRE( app.stageState( "READY" ) == 0 );

    SECTION( "a negative setpoint is reported in the user's frame" )
    {
        REQUIRE( app.setHwp( -22.5 ) == 0 );
        REQUIRE( app.m_sent.back() == Approx( 22.5 ) );

        REQUIRE( app.stagePos( 22.5 ) == 0 );

        REQUIRE( app.m_hwpActualPos == Approx( -22.5 ) );
        REQUIRE( app.m_hwpCurPos == Approx( -22.5 ) );
        REQUIRE( app.m_hwpPosName == "Uminus" );
    }

    SECTION( "angles are reported relative to the tracking offset" )
    {
        REQUIRE( app.telData( 30, 40 ) == 0 );
        REQUIRE( app.setTracking( true ) == 0 );
        REQUIRE( app.m_sent.back() == Approx( 320 ) );

        REQUIRE( app.stagePos( 320 ) == 0 );

        REQUIRE( app.m_hwpActualPos == Approx( 40 ) );
        REQUIRE( app.m_hwpCurPos == Approx( 0 ) );
        REQUIRE( app.m_hwpPosName == "Qplus" );
    }

    SECTION( "a setpoint of 90 reads back as Qplus" )
    {
        REQUIRE( app.setHwp( 90 ) == 0 );
        REQUIRE( app.stagePos( app.m_sent.back() ) == 0 );

        REQUIRE( app.m_hwpCurPos == Approx( 90 ) );
        REQUIRE( app.m_hwpPosName == "Qplus" );
    }

    SECTION( "near-zero angles are cleaned up" )
    {
        REQUIRE( app.stagePos( 360.005 ) == 0 );

        REQUIRE( app.m_hwpActualPos == 0 );
        REQUIRE( app.m_hwpCurPos == 0 );
        REQUIRE( app.m_hwpPosName == "Qplus" );
    }

    SECTION( "the readback is not changed by a queued move" )
    {
        REQUIRE( app.setHwp( 45 ) == 0 );
        REQUIRE( app.stagePos( 315 ) == 0 );
        REQUIRE( app.m_hwpCurPos == Approx( 45 ) );

        REQUIRE( app.stageState( "OPERATING" ) == 0 );
        REQUIRE( app.setHwp( 22.5 ) == 0 );
        REQUIRE( app.stagePos( 315 ) == 0 );

        REQUIRE( app.m_hwpCurPos == Approx( 45 ) );
        REQUIRE( app.m_hwpPosName == "Qminus" );
    }
}

/// Validate the app state derived from the stage state, teldata and tracking.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker state", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::trackerState();
#endif
    // clang-format on

    hwpTracker_test app( "hwptrack" );

    REQUIRE( app.trackerState() == stateCodes::NOTCONNECTED );

    REQUIRE( app.stageState( "READY" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::NOTCONNECTED ); // no teldata yet

    REQUIRE( app.telData( 30, 40 ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::READY );

    REQUIRE( app.stageState( "OPERATING" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::OPERATING );

    REQUIRE( app.stageState( "HOMING" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::OPERATING );

    REQUIRE( app.stageState( "READY" ) == 0 );
    REQUIRE( app.setTracking( true ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::OPERATING );

    REQUIRE( app.stageState( "ERROR" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::ERROR );

    REQUIRE( app.stageState( "FAILURE" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::ERROR );

    REQUIRE( app.stageState( "NOTHOMED" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::NOTCONNECTED );

    REQUIRE( app.stageState( "POWEROFF" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::NOTCONNECTED );

    REQUIRE( app.stageState( "BOGUS" ) == 0 );
    REQUIRE( app.trackerState() == stateCodes::NOTCONNECTED );
}

/// Validate that telemetry is recorded on change and when forced.
/**
 * \ingroup hwpTracker_unit_test
 */
TEST_CASE( "hwpTracker telemetry", "[hwpTracker]" )
{
    // clang-format off
#ifdef HWPTRACKER_TEST_DOXYGEN_REF
    MagAOX::app::hwpTracker::recordPolTrack( false );
    MagAOX::app::hwpTracker::recordTelem( nullptr );
#endif
    // clang-format on

    hwpTracker_test app( "hwptrack" );

    REQUIRE( app.stageState( "READY" ) == 0 );

    REQUIRE( app.setHwp( 45 ) == 0 );
    REQUIRE( app.stagePos( 315 ) == 0 ); // records via the readback

    REQUIRE( app.m_telemCurPos == Approx( 45 ) );

    REQUIRE( app.telData( 30, 40 ) == 0 );
    REQUIRE( app.setTracking( true ) == 0 ); // records via the tracking callback

    REQUIRE( app.m_telemTracking );

    REQUIRE( app.recordPolTrack( true ) == 0 );
    REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_poltrack *>( nullptr ) ) == 0 );
}

} // namespace hwpTrackerTest
} // namespace libXWCTest
