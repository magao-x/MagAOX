/** \file qwpPupilTracker_test.cpp
 * \brief Catch2 tests for the qwpPupilTracker app: model evaluation, calibration loading, configuration, and the
 * ttmperi command logic.
 *
 * \ingroup qwpPupilTracker_files
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <fstream>

#define QWPPUPILTRACKER_TEST_NOSEND
#include "../qwpPupilTracker.hpp"

/** \defgroup qwpPupilTracker_unit_test qwpPupilTracker Unit Tests
 * \brief Unit tests for the qwpPupilTracker application.
 *
 * \ingroup application_unit_test
 */

namespace libXWCTest
{

/// Namespace for `qwpPupilTracker` unit tests.
/** \ingroup qwpPupilTracker_unit_test
 */
namespace qwpPupilTrackerTest
{

using MagAOX::app::qwpPupilTracker;
using MagAOX::app::stateCodes;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS

/// A synthetic model: dx = 0.3 cos(p1) + 0.1 sin(p1) + 0.05 cos(p1 + p2), dy = 0.2 sin(p1) + 0.5 cos(p2) + 0.02
/// sin(p1 + p2).
const char *g_model = "# synthetic model\n"
                      "# a, b, dx_cos, dx_sin, dy_cos, dy_sin\n"
                      "1, 0, 0.3, 0.1, 0.0, 0.2\n"
                      "0, 1, 0.0, 0.0, 0.5, 0.0\n"
                      "1, 1, 0.05, 0.0, 0.0, 0.02\n";

/// The synthetic control matrix, `CM[v][px]` as written from python.
const double g_cm[2][2] = { { 60, -5 }, { -7, 62 } };

/// Analytic evaluation of the synthetic model.
void synthShift( double &dx,   /**< [out] x shift */
                 double &dy,   /**< [out] y shift */
                 double  phi1, /**< [in] QWP1 stage angle, degrees */
                 double  phi2 /**< [in] QWP2 stage angle, degrees */ )
{
    double p1 = phi1 * M_PI / 180;
    double p2 = phi2 * M_PI / 180;

    dx = 0.3 * cos( p1 ) + 0.1 * sin( p1 ) + 0.05 * cos( p1 + p2 );
    dy = 0.2 * sin( p1 ) + 0.5 * cos( p2 ) + 0.02 * sin( p1 + p2 );
}

/// Test harness that injects stage and ttmperi updates and inspects the commands that would be sent.
class qwpPupilTracker_test : public qwpPupilTracker
{
  public:
    using qwpPupilTracker::computeOffset;
    using qwpPupilTracker::loadCM;
    using qwpPupilTracker::loadModel;
    using qwpPupilTracker::m_cm;
    using qwpPupilTracker::m_cmFile;
    using qwpPupilTracker::m_dv;
    using qwpPupilTracker::m_model;
    using qwpPupilTracker::m_modelFile;
    using qwpPupilTracker::m_offset;
    using qwpPupilTracker::m_phiRef;
    using qwpPupilTracker::m_qwp;
    using qwpPupilTracker::m_rebasePending;
    using qwpPupilTracker::m_rebaseTol;
    using qwpPupilTracker::m_shiftRef;
    using qwpPupilTracker::m_tracking;
    using qwpPupilTracker::m_ttm;
    using qwpPupilTracker::m_ttmDevName;

    /// Construct with a scratch directory holding telemetry and the synthetic calibration.
    explicit qwpPupilTracker_test( const std::string &name /**< [in] app name and scratch directory */ )
    {
        m_root = "/tmp/qwpPupilTracker_test/" + name;
        std::filesystem::remove_all( m_root );
        std::filesystem::create_directories( m_root + "/telem" );
        std::filesystem::create_directories( m_root + "/calib/qwptrack" );

        m_calibDir = m_root + "/calib";
        writeModel( g_model );
        writeCM( g_cm );

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
    ~qwpPupilTracker_test()
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
        std::string path = m_root + "/qwppupil.conf";
        std::ofstream( path ) << text;

        setupConfig();
        config.readConfig( path );

        return loadConfigImpl( config );
    }

    /// Write the model table.
    void writeModel( const std::string &text /**< [in] table contents */ )
    {
        std::ofstream( m_calibDir + "/qwptrack/qwp_pupil_model.csv" ) << text;
    }

    /// Write a control matrix as python would, so that FITS axis 1 is the pixel index.
    void writeCM( const double cm[2][2] /**< [in] `cm[v][px]` */ )
    {
        mx::improc::eigenImage<double> im( 2, 2 );

        for( int v = 0; v < 2; ++v )
        {
            for( int p = 0; p < 2; ++p )
            {
                im( p, v ) = cm[v][p];
            }
        }

        std::string path = m_calibDir + "/qwptrack/qwp_pupil_cm.fits";
        std::filesystem::remove( path );

        mx::fits::fitsFile<double> ff;
        ff.write( path, im );
    }

    /// Write an arbitrary image as the control matrix.
    void writeCMImage( const mx::improc::eigenImage<double> &im /**< [in] the image */ )
    {
        std::string path = m_calibDir + "/qwptrack/qwp_pupil_cm.fits";
        std::filesystem::remove( path );

        mx::fits::fitsFile<double> ff;
        ff.write( path, im );
    }

    /// Report a stage position.
    int stagePos( int    n, /**< [in] QWP index */
                  double deg /**< [in] stage angle */ )
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

    /// Report a ttmperi axis target.
    int ttmTarget( int    n, /**< [in] axis index */
                   double v /**< [in] target, V */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_ttmDevName );
        ip.setName( "axis" + std::to_string( n + 1 ) + "_voltage" );
        ip.add( pcf::IndiElement( "current" ) );
        ip.add( pcf::IndiElement( "target" ) );
        ip["current"].set( v );
        ip["target"].set( v );

        return ( n == 0 ) ? setCallBack_m_indiP_ttmPos1( ip ) : setCallBack_m_indiP_ttmPos2( ip );
    }

    /// Report the ttmperi state.
    int ttmState( const std::string &st /**< [in] state string */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( m_ttmDevName );
        ip.setName( "fsm" );
        ip.add( pcf::IndiElement( "state" ) );
        ip["state"].set( st );

        return setCallBack_m_indiP_ttmFsm( ip );
    }

    /// Put both stages READY at the given angles and ttmperi OPERATING at the given targets.
    void allReady( double phi1, /**< [in] QWP1 stage angle */
                   double phi2, /**< [in] QWP2 stage angle */
                   double v1,   /**< [in] ttmperi axis 1 target */
                   double v2 /**< [in] ttmperi axis 2 target */ )
    {
        stageState( 0, "READY" );
        stageState( 1, "READY" );
        stagePos( 0, phi1 );
        stagePos( 1, phi2 );
        ttmState( "OPERATING" );
        ttmTarget( 0, v1 );
        ttmTarget( 1, v2 );
    }

    /// Request tracking on or off.
    int track( bool on /**< [in] tracking requested */ )
    {
        pcf::IndiProperty ip = m_indiP_tracking;
        ip["toggle"].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );

        return newCallBack_m_indiP_tracking( ip );
    }

    /// The correction `-CM * (shift(phi) - shift(phi_ref))` for the synthetic calibration.
    void expectedDv( double &dv1,  /**< [out] axis 1 correction */
                     double &dv2,  /**< [out] axis 2 correction */
                     double  phi1, /**< [in] QWP1 stage angle */
                     double  phi2 /**< [in] QWP2 stage angle */ )
    {
        double dx, dy, rx, ry;
        synthShift( dx, dy, phi1, phi2 );
        synthShift( rx, ry, m_phiRef[0], m_phiRef[1] );

        dx -= rx;
        dy -= ry;

        dv1 = -( g_cm[0][0] * dx + g_cm[0][1] * dy );
        dv2 = -( g_cm[1][0] * dx + g_cm[1][1] * dy );
    }

  private:
    std::string m_root; ///< Scratch directory.
};
/// \endcond

/// The model evaluates the harmonic series in mechanical degrees, periodic in 360.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker evaluates the shift model", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::shift(); qwpPupilTracker::loadModel();
    #endif
    // clang-format on

    qwpPupilTracker_test app( "model" );
    REQUIRE( app.loadModel( "/tmp/qwpPupilTracker_test/model/calib/qwptrack/qwp_pupil_model.csv" ) == 0 );
    REQUIRE( app.m_model.a.size() == 3 );

    double dx, dy;

    qwpPupilTracker::shift( dx, dy, app.m_model, 0, 0 );
    REQUIRE( dx == Approx( 0.35 ) );
    REQUIRE( dy == Approx( 0.5 ) );

    qwpPupilTracker::shift( dx, dy, app.m_model, 90, 0 );
    REQUIRE( dx == Approx( 0.1 ) );
    REQUIRE( dy == Approx( 0.72 ) );

    qwpPupilTracker::shift( dx, dy, app.m_model, 90, 90 );
    REQUIRE( dx == Approx( 0.05 ) );
    REQUIRE( dy == Approx( 0.2 ) );

    double ex, ey;

    for( double p1 : { -270.0, 13.7, 157.42, 401.0 } )
    {
        for( double p2 : { -45.0, 165.8, 345.8, 450.0 } )
        {
            qwpPupilTracker::shift( dx, dy, app.m_model, p1, p2 );
            synthShift( ex, ey, p1, p2 );

            REQUIRE( dx == Approx( ex ).margin( 1e-12 ) );
            REQUIRE( dy == Approx( ey ).margin( 1e-12 ) );
        }
    }
}

/// The reference shift is the model evaluated at the configured mechanical reference angles.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker evaluates the reference shift at startup", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::loadCalib();
    #endif
    // clang-format on

    qwpPupilTracker_test app( "reference" );
    REQUIRE( app.startup() == 0 );

    REQUIRE( app.m_phiRef[0] == Approx( 157.42 ) );
    REQUIRE( app.m_phiRef[1] == Approx( 165.8 ) );

    double rx, ry;
    synthShift( rx, ry, 157.42, 165.8 );

    REQUIRE( app.m_shiftRef[0] == Approx( rx ) );
    REQUIRE( app.m_shiftRef[1] == Approx( ry ) );
}

/// The control matrix FITS written from python as `CM[v][px]` loads without being transposed.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker loads the control matrix in python orientation", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::loadCM();
    #endif
    // clang-format on

    qwpPupilTracker_test app( "cm" );
    REQUIRE( app.startup() == 0 );

    REQUIRE( app.m_cm[0][0] == Approx( 60 ) );
    REQUIRE( app.m_cm[0][1] == Approx( -5 ) ); // row v1, column dy
    REQUIRE( app.m_cm[1][0] == Approx( -7 ) ); // row v2, column dx
    REQUIRE( app.m_cm[1][1] == Approx( 62 ) );
}

/// Missing or malformed calibration files fail startup.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker rejects bad calibration files", "[qwpPupilTracker]" )
{
    SECTION( "missing model" )
    {
        qwpPupilTracker_test app( "bad_missing_model" );
        app.m_modelFile = "qwptrack/nope.csv";
        REQUIRE( app.startup() < 0 );
    }

    SECTION( "incomplete row" )
    {
        qwpPupilTracker_test app( "bad_short_row" );
        app.writeModel( "1, 0, 0.3, 0.1, 0.0, 0.2\n0, 1, 0.5\n" );
        REQUIRE( app.startup() < 0 );
    }

    SECTION( "non-integer frequency" )
    {
        qwpPupilTracker_test app( "bad_frequency" );
        app.writeModel( "1.5, 0, 0.3, 0.1, 0.0, 0.2\n" );
        REQUIRE( app.startup() < 0 );
    }

    SECTION( "empty model" )
    {
        qwpPupilTracker_test app( "bad_empty_model" );
        app.writeModel( "# nothing\n" );
        REQUIRE( app.startup() < 0 );
    }

    SECTION( "missing control matrix" )
    {
        qwpPupilTracker_test app( "bad_missing_cm" );
        app.m_cmFile = "qwptrack/nope.fits";
        REQUIRE( app.startup() < 0 );
    }

    SECTION( "control matrix not 2x2" )
    {
        qwpPupilTracker_test           app( "bad_cm_shape" );
        mx::improc::eigenImage<double> im( 3, 2 );
        im.setZero();
        app.writeCMImage( im );
        REQUIRE( app.startup() < 0 );
    }
}

/// Configuration defaults and overrides.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker configuration", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::setupConfig(); qwpPupilTracker::loadConfigImpl(); qwpPupilTracker::loadConfig();
    #endif
    // clang-format on

    SECTION( "defaults" )
    {
        qwpPupilTracker_test app( "config_defaults" );
        REQUIRE( app.loadConfigFile( "" ) == 0 );

        REQUIRE( app.m_qwp[0].devName == "stageqwprot1" );
        REQUIRE( app.m_qwp[1].devName == "stageqwprot2" );
        REQUIRE( app.m_phiRef[0] == Approx( 157.42 ) );
        REQUIRE( app.m_phiRef[1] == Approx( 165.8 ) );
        REQUIRE( app.m_ttmDevName == "ttmperi" );
        REQUIRE( app.m_rebaseTol == Approx( 0.01 ) );
        REQUIRE( app.m_modelFile == "qwptrack/qwp_pupil_model.csv" );
        REQUIRE( app.m_cmFile == "qwptrack/qwp_pupil_cm.fits" );
    }

    SECTION( "overrides" )
    {
        qwpPupilTracker_test app( "config_overrides" );
        REQUIRE( app.loadConfigFile( "[stages]\nqwp1=q1\nqwp2=q2\n"
                                     "[reference]\nqwp1=10\nqwp2=20\n"
                                     "[ttm]\ndevice=ttm\nrebaseTol=0.5\n"
                                     "[calib]\nmodel=m.csv\ncm=c.fits\n" ) == 0 );

        REQUIRE( app.m_qwp[0].devName == "q1" );
        REQUIRE( app.m_qwp[1].devName == "q2" );
        REQUIRE( app.m_phiRef[0] == Approx( 10 ) );
        REQUIRE( app.m_phiRef[1] == Approx( 20 ) );
        REQUIRE( app.m_ttmDevName == "ttm" );
        REQUIRE( app.m_rebaseTol == Approx( 0.5 ) );
        REQUIRE( app.m_modelFile == "m.csv" );
        REQUIRE( app.m_cmFile == "c.fits" );
    }
}

/// The offset property follows the stages whether or not tracking is on.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker publishes the offset in pixels and volts", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::computeOffset(); qwpPupilTracker::stagePosUpdate();
    qwpPupilTracker::setCallBack_m_indiP_qwp1Position(pcf::IndiProperty());
    qwpPupilTracker::setCallBack_m_indiP_qwp2Position(pcf::IndiProperty());
    #endif
    // clang-format on

    qwpPupilTracker_test app( "offset" );
    REQUIRE( app.startup() == 0 );

    app.stageState( 0, "READY" );
    app.stageState( 1, "READY" );
    app.stagePos( 0, 200 );
    app.stagePos( 1, 100 );

    double dx, dy, rx, ry, dv1, dv2;
    synthShift( dx, dy, 200, 100 );
    synthShift( rx, ry, 157.42, 165.8 );
    app.expectedDv( dv1, dv2, 200, 100 );

    // the property itself only updates with an INDI driver, so check the values it is set from
    REQUIRE( app.m_offset[0] == Approx( dx - rx ) );
    REQUIRE( app.m_offset[1] == Approx( dy - ry ) );
    REQUIRE( app.m_dv[0] == Approx( dv1 ) );
    REQUIRE( app.m_dv[1] == Approx( dv2 ) );

    REQUIRE_FALSE( app.m_tracking );
    REQUIRE( app.m_ttm[0].nSent == 0 );
}

/// Enabling captures the baseline without moving ttmperi, and QWP moves command `v0 - CM * offset`.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker tracks the QWPs with ttmperi", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::appStartup(); qwpPupilTracker::appLogic(); qwpPupilTracker::update();
    qwpPupilTracker::startTracking(); qwpPupilTracker::stopTracking(); qwpPupilTracker::sendTTM();
    qwpPupilTracker::newCallBack_m_indiP_tracking(pcf::IndiProperty());
    #endif
    // clang-format on

    qwpPupilTracker_test app( "tracking" );
    REQUIRE( app.startup() == 0 );

    double dv1, dv2;

    SECTION( "startup is not tracking and sends nothing" )
    {
        app.allReady( 200, 100, 10, 20 );
        app.logic();

        REQUIRE_FALSE( app.m_tracking );
        REQUIRE( app.m_ttm[0].nSent == 0 );
        REQUIRE( app.state() == stateCodes::READY );
    }

    SECTION( "enable is refused without ttmperi" )
    {
        app.stageState( 0, "READY" );
        app.stageState( 1, "READY" );
        app.stagePos( 0, 200 );
        app.stagePos( 1, 100 );

        app.track( true );

        REQUIRE_FALSE( app.m_tracking );
        REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    }

    SECTION( "enable is refused while a stage is not usable" )
    {
        app.allReady( 200, 100, 10, 20 );
        app.stageState( 1, "NOTHOMED" );

        app.track( true );

        REQUIRE_FALSE( app.m_tracking );
    }

    SECTION( "enable captures the baseline and sends nothing" )
    {
        app.allReady( 200, 100, 10, 20 );
        app.track( true );
        app.logic();

        app.expectedDv( dv1, dv2, 200, 100 );

        REQUIRE( app.m_tracking );
        REQUIRE( app.state() == stateCodes::OPERATING );
        REQUIRE( app.m_ttm[0].v0 == Approx( 10 - dv1 ) );
        REQUIRE( app.m_ttm[1].v0 == Approx( 20 - dv2 ) );
        REQUIRE( app.m_ttm[0].nSent == 0 );
        REQUIRE( app.m_ttm[1].nSent == 0 );
    }

    SECTION( "stage moves command v0 + dv, and repeats are not resent" )
    {
        app.allReady( 200, 100, 10, 20 );
        app.track( true );

        double v0[2] = { app.m_ttm[0].v0, app.m_ttm[1].v0 };

        // positions during a move are applied too, stages OPERATING
        app.stageState( 0, "OPERATING" );
        app.stagePos( 0, 210 );

        app.expectedDv( dv1, dv2, 210, 100 );

        REQUIRE( app.m_ttm[0].nSent == 1 );
        REQUIRE( app.m_ttm[1].nSent == 1 );
        REQUIRE( app.m_ttm[0].lastSent == Approx( v0[0] + dv1 ) );
        REQUIRE( app.m_ttm[1].lastSent == Approx( v0[1] + dv2 ) );

        app.stagePos( 0, 210 );
        app.logic();

        REQUIRE( app.m_ttm[0].nSent == 1 );
        REQUIRE( app.m_ttm[1].nSent == 1 );

        // returning to where tracking started restores the original voltages
        app.stagePos( 0, 200 );

        REQUIRE( app.m_ttm[0].lastSent == Approx( 10 ) );
        REQUIRE( app.m_ttm[1].lastSent == Approx( 20 ) );
    }

    SECTION( "disable stops commands and leaves ttmperi alone" )
    {
        app.allReady( 200, 100, 10, 20 );
        app.track( true );
        app.stagePos( 0, 210 );

        int    n0   = app.m_ttm[0].nSent;
        double last = app.m_ttm[0].lastSent;

        app.track( false );
        app.stagePos( 0, 220 );
        app.logic();

        REQUIRE_FALSE( app.m_tracking );
        REQUIRE( app.m_ttm[0].nSent == n0 );
        REQUIRE( app.m_ttm[0].lastSent == last );
        REQUIRE( app.state() == stateCodes::READY );
    }
}

/// Echoes of this app's commands are ignored; external ttmperi moves re-base the baseline.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker re-bases on external ttmperi moves", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::ttmPosUpdate(); qwpPupilTracker::rebase();
    qwpPupilTracker::ttmStateUpdate(); qwpPupilTracker::stageStateUpdate();
    qwpPupilTracker::setCallBack_m_indiP_ttmPos1(pcf::IndiProperty());
    qwpPupilTracker::setCallBack_m_indiP_ttmPos2(pcf::IndiProperty());
    qwpPupilTracker::setCallBack_m_indiP_ttmFsm(pcf::IndiProperty());
    #endif
    // clang-format on

    qwpPupilTracker_test app( "rebase" );
    REQUIRE( app.startup() == 0 );

    app.allReady( 200, 100, 10, 20 );
    app.track( true );

    double v0[2] = { app.m_ttm[0].v0, app.m_ttm[1].v0 };
    double dv1, dv2;

    SECTION( "echoes, including late ones, do not re-base" )
    {
        app.stagePos( 0, 210 );
        double first = app.m_ttm[0].lastSent;

        app.stagePos( 0, 220 );

        app.ttmTarget( 0, first ); // late echo of the first command
        app.ttmTarget( 0, app.m_ttm[0].lastSent );

        REQUIRE( app.m_ttm[0].v0 == Approx( v0[0] ) );
        REQUIRE( app.m_ttm[1].v0 == Approx( v0[1] ) );
    }

    SECTION( "an external move is kept and later moves are relative to it" )
    {
        app.ttmTarget( 0, 15 );

        app.expectedDv( dv1, dv2, 200, 100 );

        REQUIRE( app.m_ttm[0].v0 == Approx( 15 - dv1 ) );
        REQUIRE( app.m_ttm[1].v0 == Approx( v0[1] ) );

        app.logic();
        REQUIRE( app.m_ttm[0].nSent == 0 ); // the new target is kept, not overwritten

        app.stagePos( 0, 210 );
        app.expectedDv( dv1, dv2, 210, 100 );

        REQUIRE( app.m_ttm[0].lastSent == Approx( app.m_ttm[0].v0 + dv1 ) );
        REQUIRE( app.m_ttm[1].lastSent == Approx( v0[1] + dv2 ) );
    }

    SECTION( "ttmperi dropping out holds commands and re-bases on return" )
    {
        app.ttmState( "NOTCONNECTED" );
        REQUIRE( app.m_rebasePending );

        app.stagePos( 0, 210 );
        REQUIRE( app.m_ttm[0].nSent == 0 );

        app.ttmTarget( 0, 0 ); // restarted with a different target
        app.ttmTarget( 1, 0 );
        app.ttmState( "READY" );
        app.logic();

        app.expectedDv( dv1, dv2, 210, 100 );

        REQUIRE_FALSE( app.m_rebasePending );
        REQUIRE( app.m_ttm[0].v0 == Approx( 0 - dv1 ) );
        REQUIRE( app.m_ttm[1].v0 == Approx( 0 - dv2 ) );
        REQUIRE( app.m_ttm[0].nSent == 0 );

        app.stagePos( 0, 220 );
        REQUIRE( app.m_ttm[0].nSent == 1 );
    }

    SECTION( "a stage dropping out holds commands" )
    {
        app.stageState( 0, "NOTCONNECTED" );
        app.stagePos( 0, 210 );
        app.logic();

        REQUIRE( app.m_ttm[0].nSent == 0 );

        app.stageState( 0, "READY" );
        app.logic();

        REQUIRE( app.m_ttm[0].nSent == 1 );
    }
}

} // namespace qwpPupilTrackerTest

} // namespace libXWCTest
