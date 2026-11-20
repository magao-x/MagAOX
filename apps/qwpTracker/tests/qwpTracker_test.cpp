/** \file qwpTracker_test.cpp
 * \brief Catch2 tests for the qwpTracker app: angle mapping, tracking-law tables, filter selection, modes, gating and
 * FSM.
 *
 * \ingroup qwpTracker_files
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <fstream>

#define QWPTRACKER_TEST_NOSEND
#include "../qwpTracker.hpp"

namespace libXWCTest
{

/// Namespace for `qwpTracker` unit tests.
/** \ingroup qwpTracker_unit_test
 */
namespace qwpTrackerTest
{

using MagAOX::app::qwpTracker;
using MagAOX::app::stateCodes;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test harness that injects stage and K-mirror updates and inspects the commands that would be sent.
class qwpTracker_test : public qwpTracker
{
  public:
    using qwpTracker::computeTargets;
    using qwpTracker::m_filter;
    using qwpTracker::m_filterDevName;
    using qwpTracker::m_haveK;
    using qwpTracker::m_indiP_law;
    using qwpTracker::m_k;
    using qwpTracker::m_kDevName;
    using qwpTracker::m_lawDir;
    using qwpTracker::m_laws;
    using qwpTracker::m_qwp;
    using qwpTracker::m_retryTol;
    using qwpTracker::m_tracking;

    /// Construct with a scratch directory for telemetry and the law tables.
    /** Writes a law for filter `r` that interpolates to `qwp1 = 2k`, `qwp2 = -2k` over K-mirror -90 to 90.
     */
    explicit qwpTracker_test( const std::string &name /**< [in] app name and scratch directory */ )
    {
        m_root = "/tmp/qwpTracker_test/" + name;
        std::filesystem::remove_all( m_root );
        std::filesystem::create_directories( m_root + "/telem" );
        std::filesystem::create_directories( m_root + "/calib/qwptrack" );

        m_calibDir = m_root + "/calib";
        writeLaw( "r", "# linear law\n-90,-180,180\n90,180,-180\n" );

        m_configName = name;
        m_basePath   = m_root;
        m_sysPath    = m_root + "/sys";
        m_tel.logPath( m_root + "/telem" );
        m_tel.logExt( "bintel" );
        m_tel.logName( m_configName );
        m_tel.m_logLevel = flatlogs::logPrio::LOG_TELEM;
        m_maxInterval    = 3600.0;
    }

    /// Shut down telemetry.
    ~qwpTracker_test()
    {
        m_tel.logShutdown( true );
    }

    /// Run appStartup from INITIALIZED.
    int startup()
    {
        state( stateCodes::INITIALIZED );
        return appStartup();
    }

    /// Run one loop.
    int logic()
    {
        return appLogic();
    }

    /// Load a configuration file through setupConfig and loadConfigImpl.
    int loadConfigFile( const std::string &text /**< [in] config file contents */ )
    {
        std::string path = m_root + "/qwptrack.conf";
        std::ofstream( path ) << text;

        setupConfig();
        config.readConfig( path );

        return loadConfigImpl( config );
    }

    /// Write a law table for a filter into the calib directory.
    void writeLaw( const std::string &filter, /**< [in] filter name */
                   const std::string &text /**< [in] table contents */ )
    {
        std::ofstream( m_calibDir + "/qwptrack/qwp_law_" + filter + ".csv" ) << text;
    }

    /// Remove all law tables.
    void clearLaws()
    {
        std::filesystem::remove_all( m_calibDir + "/qwptrack" );
        std::filesystem::create_directories( m_calibDir + "/qwptrack" );
    }

    /// Report the filter wheel selection.
    int filter( const std::string &sel, /**< [in] the selected filter, or "" for none */
                bool               busy = false /**< [in] whether the wheel is moving */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( m_filterDevName );
        ip.setName( "filterName" );
        ip.setState( busy ? pcf::IndiProperty::Busy : pcf::IndiProperty::Idle );

        for( const std::string f : { "r", "i", "z", "open" } )
        {
            ip.add( pcf::IndiElement( f, ( f == sel ) ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        }

        return setCallBack_m_indiP_filterName( ip );
    }

    /// Report a K-mirror position.
    int kPos( float k /**< [in] K-mirror angle */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_kDevName );
        ip.setName( "position" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].set( k );

        return setCallBack_m_indiP_stagek( ip );
    }

    /// Report a stage position.
    int stagePos( int   n, /**< [in] QWP index */
                  float deg /**< [in] stage angle */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_qwp[n].devName );
        ip.setName( "position" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].set( deg );

        return ( n == 0 ) ? setCallBack_m_indiP_qwp1Position( ip ) : setCallBack_m_indiP_qwp2Position( ip );
    }

    /// Report a stage state.
    int stageState( int                n, /**< [in] QWP index */
                    const std::string &st /**< [in] state string */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( m_qwp[n].devName );
        ip.setName( "fsm" );
        ip.add( pcf::IndiElement( "state" ) );
        ip["state"].set( st );

        return ( n == 0 ) ? setCallBack_m_indiP_qwp1Fsm( ip ) : setCallBack_m_indiP_qwp2Fsm( ip );
    }

    /// Set both stages READY at the given positions.
    void stagesReady( float pos1 = 0, /**< [in] QWP1 stage angle */
                      float pos2 = 0 /**< [in] QWP2 stage angle */ )
    {
        stagePos( 0, pos1 );
        stagePos( 1, pos2 );
        stageState( 0, "READY" );
        stageState( 1, "READY" );
    }

    /// Simulate a move by the stage: OPERATING, then READY at a final position.
    void move( int   n, /**< [in] QWP index */
               float deg /**< [in] where the stage stops */ )
    {
        stageState( n, "OPERATING" );
        stagePos( n, deg );
        stageState( n, "READY" );
    }

    /// Request tracking on or off.
    int track( bool on /**< [in] tracking requested */ )
    {
        pcf::IndiProperty ip = m_indiP_tracking;
        ip["toggle"].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );

        return newCallBack_m_indiP_tracking( ip );
    }

    /// Request a manual fast-axis target.
    int manual( int   n, /**< [in] QWP index */
                float tgt /**< [in] fast-axis target */ )
    {
        pcf::IndiProperty ip = ( n == 0 ) ? m_indiP_qwp1 : m_indiP_qwp2;
        ip["target"].set( tgt );

        return ( n == 0 ) ? newCallBack_m_indiP_qwp1( ip ) : newCallBack_m_indiP_qwp2( ip );
    }

  private:
    std::string m_root; ///< Scratch directory.
};
/// \endcond

/// Wrapping, folding and the fast-axis <-> stage mapping.
/** Covers both signs, several zeros, angles near 0/180/360 and negative angles, and the choice of the nearest of the
 * two equivalent stage angles.
 *
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker maps fast-axis angles to the nearest equivalent stage angle", "[qwpTracker]" )
{
    REQUIRE( qwpTracker::wrap360( 370 ) == Approx( 10 ) );
    REQUIRE( qwpTracker::wrap360( -10 ) == Approx( 350 ) );
    REQUIRE( qwpTracker::wrap360( 360 ) == Approx( 0 ) );
    REQUIRE( qwpTracker::wrap360( -1e-9f ) < 360 );

    REQUIRE( qwpTracker::fold180( 190 ) == Approx( 10 ) );
    REQUIRE( qwpTracker::fold180( -10 ) == Approx( 170 ) );
    REQUIRE( qwpTracker::fold180( 180 ) == Approx( 0 ) );

    int   sign = GENERATE( 1, -1 );
    float zero = GENERATE( 0.0f, 37.5f, 359.0f );
    float fast = GENERATE( 0.0f, 0.5f, 90.0f, 179.5f );
    float cur  = GENERATE( 0.0f, 100.0f, 270.0f, 359.5f );

    float s = qwpTracker::fastToStage( fast, zero, sign, cur );

    INFO( "sign=" << sign << " zero=" << zero << " fast=" << fast << " cur=" << cur << " stage=" << s );

    REQUIRE( s >= 0 );
    REQUIRE( s < 360 );

    // the stage angle gives back the fast axis
    float back = qwpTracker::stageToFast( s, zero, sign );
    float err  = std::fabs( back - fast );
    REQUIRE( std::min( err, 180 - err ) < 1e-3 );

    // and the move is never more than 90 degrees
    float d = qwpTracker::wrap360( s - cur );
    REQUIRE( std::min( d, 360 - d ) <= 90 + 1e-3 );
}

/// The nearest equivalent is chosen across the 0/360 boundary.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker chooses the short way across 0/360", "[qwpTracker]" )
{
    // fast 175 with zero 0, sign +1: candidates 175 and 355; from 5 the nearer is 355
    REQUIRE( qwpTracker::fastToStage( 175, 0, 1, 5 ) == Approx( 355 ) );

    // from 170 the nearer is 175
    REQUIRE( qwpTracker::fastToStage( 175, 0, 1, 170 ) == Approx( 175 ) );

    // sign -1, zero 10: fast 20 gives 350 or 170
    REQUIRE( qwpTracker::fastToStage( 20, 10, -1, 0 ) == Approx( 350 ) );
    REQUIRE( qwpTracker::fastToStage( 20, 10, -1, 180 ) == Approx( 170 ) );
}

/// Law tables are loaded from the calib directory, named by filter, and validated.
/** A missing directory or no tables starts with no laws; a malformed table fails startup.
 *
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker loads tracking-law tables", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::loadLaws(); qwpTracker::appStartup();
    #endif
    // clang-format on

    qwpTracker_test app( "laws" );

    SECTION( "valid tables load, keyed by filter, ignoring other files" )
    {
        app.writeLaw( "i", "-10,170,0\n0,190,10\n" );
        std::ofstream( app.calibDir() + "/qwptrack/notes.txt" ) << "x";

        REQUIRE( app.startup() == 0 );
        REQUIRE( app.m_laws.size() == 2 );
        REQUIRE( app.m_laws.count( "r" ) == 1 );
        REQUIRE( app.m_laws.at( "i" ).imr.size() == 2 );
        REQUIRE( app.m_laws.at( "i" ).qwp[0][1] == Approx( 190 ) );
    }

    SECTION( "no tables starts with no laws" )
    {
        app.clearLaws();
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.m_laws.empty() );
    }

    SECTION( "no directory starts with no laws" )
    {
        app.m_lawDir = "missing";
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.m_laws.empty() );
    }

    SECTION( "a single row fails" )
    {
        app.writeLaw( "i", "0,1,2\n" );
        REQUIRE( app.startup() < 0 );
    }

    SECTION( "non-increasing K-mirror angles fail" )
    {
        app.writeLaw( "i", "0,1,2\n0,2,3\n" );
        REQUIRE( app.startup() < 0 );
    }
}

/// The law interpolates linearly in the unwrapped QWP angles and refuses K-mirror angles outside the table.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker interpolates the tracking law", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::computeTargets(); qwpTracker::filterUpdate();
    #endif
    // clang-format on

    qwpTracker_test app( "interp" );
    app.writeLaw( "i", "-10,170,0\n0,190,10\n10,200,-20\n" );
    REQUIRE( app.startup() == 0 );

    float q1 = -1;
    float q2 = -1;

    // no filter selected
    REQUIRE_FALSE( app.computeTargets( 0, q1, q2 ) );

    app.filter( "i" );

    // grid points
    REQUIRE( app.computeTargets( -10, q1, q2 ) );
    REQUIRE( q1 == Approx( 170 ) );
    REQUIRE( q2 == Approx( 0 ) );

    REQUIRE( app.computeTargets( 10, q1, q2 ) );
    REQUIRE( q1 == Approx( 200 ) );
    REQUIRE( q2 == Approx( -20 ) );

    // midpoints, continuous across 180 (QWP1) and 0 (QWP2)
    REQUIRE( app.computeTargets( -5, q1, q2 ) );
    REQUIRE( q1 == Approx( 180 ) );
    REQUIRE( q2 == Approx( 5 ) );

    REQUIRE( app.computeTargets( 7.5, q1, q2 ) );
    REQUIRE( q1 == Approx( 197.5 ) );
    REQUIRE( q2 == Approx( -12.5 ) );

    // outside the table
    REQUIRE_FALSE( app.computeTargets( -10.01, q1, q2 ) );
    REQUIRE_FALSE( app.computeTargets( 10.01, q1, q2 ) );
}

/// The selected filter follows the wheel: one `On` element while not moving, otherwise none.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker follows the filter wheel", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::setCallBack_m_indiP_filterName(pcf::IndiProperty()); qwpTracker::filterUpdate();
    #endif
    // clang-format on

    qwpTracker_test app( "filter" );
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.m_indiP_law["current"].get() == "none" );

    app.filter( "r" );
    REQUIRE( app.m_filter == "r" );

    app.filter( "r", true );
    REQUIRE( app.m_filter.empty() );

    app.filter( "open" );
    REQUIRE( app.m_filter == "open" );

    app.filter( "" );
    REQUIRE( app.m_filter.empty() );
}

/// The tracker FSM state derived from tracking and the two stage states.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker derives its state from both stages", "[qwpTracker]" )
{
    using sc = stateCodes;

    REQUIRE( qwpTracker::trackerState( false, sc::READY, sc::READY ) == sc::READY );
    REQUIRE( qwpTracker::trackerState( true, sc::READY, sc::READY ) == sc::OPERATING );
    REQUIRE( qwpTracker::trackerState( true, sc::POWEROFF, sc::READY ) == sc::OPERATING );
    REQUIRE( qwpTracker::trackerState( false, sc::OPERATING, sc::READY ) == sc::OPERATING );
    REQUIRE( qwpTracker::trackerState( false, sc::READY, sc::OPERATING ) == sc::OPERATING );
    REQUIRE( qwpTracker::trackerState( false, sc::READY, sc::NOTHOMED ) == sc::NOTHOMED );
    REQUIRE( qwpTracker::trackerState( false, sc::HOMING, sc::READY ) == sc::HOMING );
    REQUIRE( qwpTracker::trackerState( false, sc::HOMING, sc::NOTHOMED ) == sc::NOTHOMED );
    REQUIRE( qwpTracker::trackerState( false, sc::POWEROFF, sc::NOTHOMED ) == sc::NOTCONNECTED );
    REQUIRE( qwpTracker::trackerState( false, sc::READY, sc::NODEVICE ) == sc::NOTCONNECTED );
}

/// Configuration defaults, overrides, and rejection of an invalid sign.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker configuration", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::setupConfig(); qwpTracker::loadConfigImpl(); qwpTracker::loadConfig();
    #endif
    // clang-format on

    SECTION( "defaults" )
    {
        qwpTracker_test app( "config_defaults" );
        REQUIRE( app.loadConfigFile( "" ) == 0 );

        REQUIRE( app.m_qwp[0].devName == "stageqwprot1" );
        REQUIRE( app.m_qwp[1].devName == "stageqwprot2" );
        REQUIRE( app.m_qwp[0].zero == 0 );
        REQUIRE( app.m_qwp[0].sign == 1 );
        REQUIRE( app.m_qwp[1].sign == -1 );
        REQUIRE( app.m_kDevName == "stagek" );
        REQUIRE( app.m_filterDevName == "fwsci1" );
        REQUIRE( app.m_lawDir == "qwptrack" );
        REQUIRE( app.m_retryTol == Approx( 0.5 ) );
    }

    SECTION( "overrides" )
    {
        qwpTracker_test app( "config_overrides" );
        REQUIRE( app.loadConfigFile( "[qwp1]\nzero=12.5\nsign=-1\ndevName=q1\n"
                                     "[qwp2]\nzero=200\nsign=1\ndevName=q2\n"
                                     "[k]\ndevName=kk\n"
                                     "[filter]\ndevName=fw\n"
                                     "[law]\ndir=laws\n"
                                     "[retry]\ntolerance=0.2\n" ) == 0 );

        REQUIRE( app.m_qwp[0].zero == Approx( 12.5 ) );
        REQUIRE( app.m_qwp[0].sign == -1 );
        REQUIRE( app.m_qwp[0].devName == "q1" );
        REQUIRE( app.m_qwp[1].zero == Approx( 200 ) );
        REQUIRE( app.m_qwp[1].sign == 1 );
        REQUIRE( app.m_qwp[1].devName == "q2" );
        REQUIRE( app.m_kDevName == "kk" );
        REQUIRE( app.m_filterDevName == "fw" );
        REQUIRE( app.m_lawDir == "laws" );
        REQUIRE( app.m_retryTol == Approx( 0.2 ) );
    }

    SECTION( "invalid sign" )
    {
        qwpTracker_test app( "config_badsign" );
        REQUIRE( app.loadConfigFile( "[qwp2]\nsign=2\n" ) < 0 );
    }
}

/// Stage and K-mirror callbacks update the reported angles; the first position initializes the target.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker reports fast-axis angles from stage positions", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::setCallBack_m_indiP_stagek(pcf::IndiProperty());
    qwpTracker::setCallBack_m_indiP_qwp1Position(pcf::IndiProperty());
    qwpTracker::setCallBack_m_indiP_qwp2Position(pcf::IndiProperty());
    qwpTracker::stagePosUpdate();
    #endif
    // clang-format on

    qwpTracker_test app( "report" );
    app.m_qwp[0].zero = 10;
    app.m_qwp[1].zero = 100;
    REQUIRE( app.startup() == 0 );

    REQUIRE( app.kPos( 3.5 ) == 0 );
    REQUIRE( app.m_haveK );
    REQUIRE( app.m_k == Approx( 3.5 ) );

    // QWP1 sign +1: 250 - 10 = 240 -> 60.  QWP2 sign -1: -(250 - 100) = -150 -> 30.
    app.stagePos( 0, 250 );
    app.stagePos( 1, 250 );

    REQUIRE( app.m_qwp[0].fastCur == Approx( 60 ) );
    REQUIRE( app.m_qwp[1].fastCur == Approx( 30 ) );
    REQUIRE( app.m_qwp[0].fastTgt == Approx( 60 ) );
    REQUIRE( app.m_qwp[1].fastTgt == Approx( 30 ) );

    // later positions update current but not the target
    app.stagePos( 0, 260 );
    REQUIRE( app.m_qwp[0].fastCur == Approx( 70 ) );
    REQUIRE( app.m_qwp[0].fastTgt == Approx( 60 ) );

    // 179.999 rounds to 180.00, which folds to 0
    app.stagePos( 0, 189.999 );
    REQUIRE( app.m_qwp[0].fastCur == Approx( 0 ) );
}

/// Tracking follows K-mirror updates once both stages are ready, and never repeats an identical command.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker tracks the K-mirror", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::appStartup(); qwpTracker::appLogic(); qwpTracker::updateStages(); qwpTracker::sendStageTarget();
    qwpTracker::newCallBack_m_indiP_tracking(pcf::IndiProperty());
    #endif
    // clang-format on

    qwpTracker_test app( "tracking" );
    REQUIRE( app.startup() == 0 );
    app.filter( "r" );

    SECTION( "startup sends nothing and tracking is off" )
    {
        app.stagesReady( 10, 20 );
        app.kPos( 5 );
        app.logic();

        REQUIRE_FALSE( app.m_tracking );
        REQUIRE( app.m_qwp[0].nSent == 0 );
        REQUIRE( app.m_qwp[1].nSent == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "no K angle yet sends nothing" )
    {
        app.stagesReady();
        app.track( true );
        app.logic();

        REQUIRE( app.m_qwp[0].nSent == 0 );
        REQUIRE( app.state() == stateCodes::OPERATING );
    }

    SECTION( "K updates are applied" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );

        // qwp1 = 10, sign +1 -> stage 10.  qwp2 = -10 -> 170, sign -1 -> stage -170 = 190, nearest to 0 is 10.
        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 10 ) );
        REQUIRE( app.m_qwp[1].nSent == 1 );
        REQUIRE( app.m_qwp[1].lastSent == Approx( 10 ) );
        REQUIRE( app.m_qwp[0].fastTgt == Approx( 10 ) );
        REQUIRE( app.m_qwp[1].fastTgt == Approx( 170 ) );

        // a repeated identical K sends nothing
        app.kPos( 5 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[1].nSent == 1 );

        // a new K is sent on the next loop
        app.kPos( 6 );
        REQUIRE( app.m_qwp[0].nSent == 1 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 2 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 12 ) );
    }

    SECTION( "a stage that is not ready receives its target when it becomes ready" )
    {
        app.stagePos( 0, 0 );
        app.stagePos( 1, 0 );
        app.stageState( 0, "READY" );
        app.stageState( 1, "POWEROFF" );

        app.kPos( 5 );
        app.track( true );

        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[1].nSent == 0 );

        app.stageState( 1, "READY" );
        app.logic();

        REQUIRE( app.m_qwp[1].nSent == 1 );
        REQUIRE( app.m_qwp[1].lastSent == Approx( 10 ) );
    }

    SECTION( "a moving stage is not retargeted; it receives only the latest target when it stops" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.stageState( 0, "OPERATING" );
        app.kPos( 6 );
        app.logic();
        app.kPos( 7 );
        app.logic();

        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[1].nSent == 3 ); // QWP2 stayed READY and followed every update

        app.stageState( 0, "READY" );
        app.logic();

        REQUIRE( app.m_qwp[0].nSent == 2 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 14 ) );
    }

    SECTION( "a stage that stops away from its target is sent it once more" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.move( 0, 8 ); // missed 10
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 2 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 10 ) );

        app.move( 0, 8 ); // missed again: no further retry
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 2 );

        // a new target re-arms the retry
        app.kPos( 6 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 3 );
        app.move( 0, 11 ); // missed 12
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 4 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 12 ) );
    }

    SECTION( "a stage that stops within tolerance is not resent" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );

        app.move( 0, 10.4 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 1 );
    }

    SECTION( "a miss after tracking stops is not resent" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );

        app.stageState( 0, "OPERATING" );
        app.track( false );
        app.stagePos( 0, 8 );
        app.stageState( 0, "READY" );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 1 );
    }

    SECTION( "a filter without a law holds, and a law becoming available re-targets without a K update" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.filter( "open" );
        app.kPos( 6 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[0].fastTgt == Approx( 10 ) );

        // moving wheel also holds
        app.filter( "r", true );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.filter( "r" );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 2 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 12 ) );
    }

    SECTION( "a K-mirror angle outside the table holds" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.kPos( 95 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.kPos( 6 );
        app.logic();
        REQUIRE( app.m_qwp[0].nSent == 2 );
    }

    SECTION( "stopping tracking does not move the stages" )
    {
        app.stagesReady( 0, 0 );
        app.kPos( 5 );
        app.track( true );
        REQUIRE( app.m_qwp[0].nSent == 1 );

        app.stagePos( 0, 9 );
        app.track( false );
        app.kPos( 30 );
        app.logic();

        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[0].fastTgt == Approx( 9 ) );
        REQUIRE( app.state() == stateCodes::READY );
    }
}

/// A law table whose unwrapped QWP angles cross 180 is tracked through the fold.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker tracks a wrapped law through 180", "[qwpTracker]" )
{
    qwpTracker_test app( "tracking_wrapped" );
    app.writeLaw( "i", "-10,170,0\n0,190,10\n" );
    REQUIRE( app.startup() == 0 );
    app.filter( "i" );
    app.stagesReady( 175, 0 );
    app.kPos( -5 );
    app.track( true );

    // qwp1 = 180 -> fast 0, stage 0 or 180, nearest to 175 is 180.  qwp2 = 5, sign -1 -> 355 or 175, nearest 355.
    REQUIRE( app.m_qwp[0].fastTgt == Approx( 0 ) );
    REQUIRE( app.m_qwp[0].lastSent == Approx( 180 ) );
    REQUIRE( app.m_qwp[1].fastTgt == Approx( 5 ) );
    REQUIRE( app.m_qwp[1].lastSent == Approx( 355 ) );
}

/// Manual targets are rejected while tracking and sent immediately otherwise.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker manual targets", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::newCallBack_m_indiP_qwp1(pcf::IndiProperty());
    qwpTracker::newCallBack_m_indiP_qwp2(pcf::IndiProperty());
    qwpTracker::manualTarget();
    #endif
    // clang-format on

    qwpTracker_test app( "manual" );
    app.m_qwp[0].zero = 30;
    REQUIRE( app.startup() == 0 );
    app.stagesReady( 0, 0 );

    SECTION( "not tracking: sent immediately, folded, and resent on repeat" )
    {
        REQUIRE( app.manual( 0, 200 ) == 0 );

        // 200 -> 20, stage 50 or 230, nearest to 0 is 50
        REQUIRE( app.m_qwp[0].nSent == 1 );
        REQUIRE( app.m_qwp[0].lastSent == Approx( 50 ) );
        REQUIRE( app.m_qwp[0].fastTgt == Approx( 20 ) );

        REQUIRE( app.manual( 0, 20 ) == 0 );
        REQUIRE( app.m_qwp[0].nSent == 2 );
    }

    SECTION( "tracking: rejected" )
    {
        app.track( true );
        REQUIRE( app.manual( 1, 45 ) == 0 );

        REQUIRE( app.m_qwp[1].nSent == 0 );
        REQUIRE( app.m_qwp[1].fastTgt == Approx( 0 ) );
    }
}

} // namespace qwpTrackerTest

} // namespace libXWCTest
