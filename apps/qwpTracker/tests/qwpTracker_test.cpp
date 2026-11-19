/** \file qwpTracker_test.cpp
 * \brief Catch2 tests for the qwpTracker app: angle mapping, tracking law, modes, gating and FSM.
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
    using qwpTracker::m_haveK;
    using qwpTracker::m_k;
    using qwpTracker::m_kDevName;
    using qwpTracker::m_qwp;
    using qwpTracker::m_tracking;

    /// Construct with a scratch directory for telemetry.
    explicit qwpTracker_test( const std::string &name /**< [in] app name and scratch directory */ )
    {
        m_root = "/tmp/qwpTracker_test/" + name;
        std::filesystem::remove_all( m_root );
        std::filesystem::create_directories( m_root + "/telem" );

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

/// The placeholder tracking law is pinned so a change to it is deliberate.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker dummy tracking law", "[qwpTracker]" )
{
    float k = GENERATE( 0.0f, 12.5f, -40.0f );

    float q1;
    float q2;
    qwpTracker::computeTargets( k, q1, q2 );

    REQUIRE( q1 == Approx( 2 * k ) );
    REQUIRE( q2 == Approx( -2 * k ) );
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
    }

    SECTION( "overrides" )
    {
        qwpTracker_test app( "config_overrides" );
        REQUIRE( app.loadConfigFile( "[qwp1]\nzero=12.5\nsign=-1\ndevName=q1\n"
                                     "[qwp2]\nzero=200\nsign=1\ndevName=q2\n"
                                     "[k]\ndevName=kk\n" ) == 0 );

        REQUIRE( app.m_qwp[0].zero == Approx( 12.5 ) );
        REQUIRE( app.m_qwp[0].sign == -1 );
        REQUIRE( app.m_qwp[0].devName == "q1" );
        REQUIRE( app.m_qwp[1].zero == Approx( 200 ) );
        REQUIRE( app.m_qwp[1].sign == 1 );
        REQUIRE( app.m_qwp[1].devName == "q2" );
        REQUIRE( app.m_kDevName == "kk" );
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
