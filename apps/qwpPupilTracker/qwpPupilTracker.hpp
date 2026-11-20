/** \file qwpPupilTracker.hpp
 * \brief The MagAO-X QWP pupil tracker header file
 *
 * \ingroup qwpPupilTracker_files
 */

#ifndef qwpPupilTracker_hpp
#define qwpPupilTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup qwpPupilTracker
 * \brief The MagAO-X application to keep the pupil aligned with ttmperi as the compensator QWPs rotate.
 *
 * <a href="../handbook/operating/software/apps/qwpPupilTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup qwpPupilTracker_files
 * \ingroup qwpPupilTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X QWP pupil tracker.
/** Rotating the dual-QWP compensator shifts the pupil for the whole lower bench.  This app predicts that shift from
 * the mechanical angles of the two QWP rotation stages and cancels it with `ttmperi`.
 *
 * The shift model is a harmonic series in the mechanical stage angles \f$\phi_1, \phi_2\f$,
 * \f[
 *  s_{x,y}(\phi_1,\phi_2) = \sum_j c_j \cos(a_j\phi_1 + b_j\phi_2) + s_j \sin(a_j\phi_1 + b_j\phi_2),
 * \f]
 * in camwfs pixels.  The offset is \f$\Delta(\phi) = s(\phi) - s(\phi_{ref})\f$, with the reference given as
 * mechanical stage angles.  The default reference is the calibration's fast-axis pair (90, 115), which the
 * calibration script placed at stage angles (157.42, 165.80).
 *
 * When tracking is enabled ttmperi is assumed aligned at the current QWP angles, and the app captures the baseline
 * \f$v_0 = v_{now} + CM\,\Delta(\phi_{now})\f$, the voltage that would be aligned at the reference.  While tracking,
 * every stage position update commands \f$v = v_0 - CM\,\Delta(\phi)\f$.  A ttmperi target that is not one of this
 * app's own recent commands means ttmperi was moved externally (e.g. a J-test), and the baseline is re-based so the
 * new target is kept.  Commands are only sent while both stages and ttmperi report READY or OPERATING.
 *
 * The model is read from `<calib>/<calib.model>`, a comma-separated table with one row per frequency,
 * `a, b, dx_cos, dx_sin, dy_cos, dy_sin`, with `#` comments.  The 2x2 control matrix (V/px) is read from
 * `<calib>/<calib.cm>`, a FITS file written from python with `CM[v][px]`, i.e. rows `v1, v2` and columns `dx, dy`.
 *
 * \ingroup qwpPupilTracker
 */
class qwpPupilTracker : public MagAOXApp<true>, public dev::telemeter<qwpPupilTracker>
{

    // Give the test harness access.
    friend class qwpPupilTracker_test;

    friend class dev::telemeter<qwpPupilTracker>;

    typedef dev::telemeter<qwpPupilTracker> telemeterT;

  public:
    /// The harmonic pupil-shift model, one entry per frequency.
    struct shiftModel
    {
        std::vector<double> a; ///< The integer multiplier of the QWP1 stage angle for each frequency.

        std::vector<double> b; ///< The integer multiplier of the QWP2 stage angle for each frequency.

        std::vector<double> dxCos; ///< The cosine coefficients of the x shift, in pixels.

        std::vector<double> dxSin; ///< The sine coefficients of the x shift, in pixels.

        std::vector<double> dyCos; ///< The cosine coefficients of the y shift, in pixels.

        std::vector<double> dySin; ///< The sine coefficients of the y shift, in pixels.
    };

    /// The state of one QWP rotation stage.
    struct qwpStage
    {
        std::string devName; ///< The INDI device name of the rotation stage.

        stateCodes::stateCodeT stageState{ stateCodes::NOTCONNECTED }; ///< The last state reported by the stage.

        bool havePos{ false }; ///< True once the stage has reported a position.

        double pos{ 0 }; ///< The last stage position reported by the stage, in degrees.
    };

    /// The state of one ttmperi axis.
    struct ttmAxis
    {
        bool haveTgt{ false }; ///< True once the axis has reported a target.

        double tgt{ 0 }; ///< The last target reported by ttmperi, in V.

        double v0{ 0 }; ///< The baseline: the voltage that would be aligned at the reference, in V.

        bool haveSent{ false }; ///< True when `lastSent` holds a command that need not be repeated.

        double lastSent{ 0 }; ///< The last voltage commanded, in V.

        std::vector<double> recent; ///< This app's recent commands, so their echoes are not taken as external moves.

        int nSent{ 0 }; ///< The number of commands issued, for diagnostics and tests.

        pcf::IndiProperty cmd; ///< The outbound `pos_N.target` command property.
    };

  protected:
    /** \name Configurable Parameters
     *@{
     */

    qwpStage m_qwp[2]; ///< QWP1 and QWP2 stage names and state.

    double m_phiRef[2]{ 157.42, 165.8 }; ///< The reference QWP1 and QWP2 mechanical stage angles, in degrees.

    std::string m_ttmDevName{ "ttmperi" }; ///< The device name of the tip/tilt mirror.

    double m_rebaseTol{ 0.01 }; ///< A ttmperi target farther than this from every recent command is external, in V.

    std::string m_modelFile{ "qwptrack/qwp_pupil_model.csv" }; ///< The model table, relative to the calib dir.

    std::string m_cmFile{ "qwptrack/qwp_pupil_cm.fits" }; ///< The control matrix FITS file, relative to the calib dir.

    ///@}

    shiftModel m_model; ///< The pupil-shift model loaded at startup.

    double m_cm[2][2]{ { 0, 0 }, { 0, 0 } }; ///< The control matrix, V/px, `m_cm[v][px]`.

    double m_shiftRef[2]{ 0, 0 }; ///< The model shift at the reference, in pixels.

    ttmAxis m_ttm[2]; ///< The two ttmperi axes.

    stateCodes::stateCodeT m_ttmState{ stateCodes::NOTCONNECTED }; ///< The last state reported by ttmperi.

    bool m_rebasePending{ false }; ///< True when ttmperi was unavailable while tracking and must be re-based on return.

    bool m_tracking{ false }; ///< True when ttmperi follows the QWPs.

    double m_offset[2]{ 0, 0 }; ///< The current predicted pupil offset relative to the reference, in pixels.

    double m_dv[2]{ 0, 0 }; ///< The ttmperi correction for `m_offset`, `-CM * m_offset`, in V.

    bool m_telemTracking{ false }; ///< The tracking flag in the last telemetry record.

    float m_telemOffset[2]{ 0, 0 }; ///< The offset in the last telemetry record.

    float m_telemDv[2]{ 0, 0 }; ///< The correction in the last telemetry record.

  public:
    /// Default c'tor.
    qwpPupilTracker();

    /// D'tor, declared and defined for noexcept.
    ~qwpPupilTracker() noexcept
    {
    }

    /// Set up configuration entries.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     *
     * \returns 0 on success
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load configuration values, shutting down on an invalid configuration.
    virtual void loadConfig();

    /// Load the calibration and register the INDI properties and telemetry.
    virtual int appStartup();

    /// Implementation of the FSM for qwpPupilTracker.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    virtual int appShutdown();

    /** \name Calibration and Model
     *@{
     */

    /// Load and validate the model table and the control matrix, and compute the reference.
    /**
     * \returns 0 on success
     * \returns -1 if a file cannot be read or is invalid
     */
    int loadCalib();

    /// Load and validate the model table.
    /** Every data row must have exactly six columns, since `readColumns` fills missing columns with a placeholder,
     * and `a`, `b` must be integers.
     *
     * \returns 0 on success
     * \returns -1 if the table cannot be read or is invalid
     */
    int loadModel( const std::string &path /**< [in] the full path to the table */ );

    /// Load and validate the control matrix.
    /**
     * \returns 0 on success
     * \returns -1 if the file cannot be read or is not 2x2 and finite
     */
    int loadCM( const std::string &path /**< [in] the full path to the FITS file */ );

    /// Evaluate the model shift at a pair of mechanical stage angles.
    static void shift( double           &dx,    /**< [out] the x shift, in pixels */
                       double           &dy,    /**< [out] the y shift, in pixels */
                       const shiftModel &model, /**< [in] the model */
                       double            phi1,  /**< [in] the QWP1 stage angle, in degrees */
                       double            phi2 /**< [in] the QWP2 stage angle, in degrees */ );

    /// Whether a device state allows tracking: READY or OPERATING.
    /**
     * \returns true if `st` is READY or OPERATING
     */
    static bool usable( stateCodes::stateCodeT st /**< [in] the device state */ );

    ///@}

  protected:
    /// Compute the offset and, while tracking, command ttmperi.
    /** Takes `m_indiMutex`; the caller must not hold it.  Commands are sent after the mutex is released.  Nothing is
     * sent unless both stages and ttmperi are usable, and an axis is not sent a voltage within 1 uV of its last
     * command.
     *
     * \returns 0 on success
     * \returns -1 if a command could not be sent
     */
    int update();

    /// Whether both stages have usable positions.  The caller must hold `m_indiMutex`.
    /**
     * \returns true if both stages have reported a position and are READY or OPERATING
     */
    bool stagesUsable();

    /// Whether ttmperi can be commanded.  The caller must hold `m_indiMutex`.
    /**
     * \returns true if both axes have reported a target and ttmperi is READY or OPERATING
     */
    bool ttmUsable();

    /// Compute `m_offset` and `m_dv` from the current stage positions.  The caller must hold `m_indiMutex`.
    void computeOffset();

    /// Re-base one ttmperi axis so that its current target is kept.  The caller must hold `m_indiMutex`.
    void rebase( int n /**< [in] the axis index, 0 or 1 */ );

    /// Send a voltage to one ttmperi axis.
    /**
     * \returns 0 on success
     * \returns -1 on an error from sendNewProperty
     */
    int sendTTM( int    n, /**< [in] the axis index, 0 or 1 */
                 double v /**< [in] the voltage to command */ );

    /// Handle a stage `position` update.
    /**
     * \returns 0 on success
     * \returns -1 if a command could not be sent
     */
    int stagePosUpdate( int                      n, /**< [in] the QWP index, 0 or 1 */
                        const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Parse the state from an `fsm` property.
    /**
     * \returns true if the property held a non-empty `state`
     */
    static bool parseState( stateCodes::stateCodeT  &st, /**< [out] the parsed state */
                            const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Handle a stage `fsm` update.
    /**
     * \returns 0 always
     */
    int stageStateUpdate( int                      n, /**< [in] the QWP index, 0 or 1 */
                          const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Handle a ttmperi `fsm` update.
    /** A ttmperi that drops out while tracking may come back somewhere else, so it is re-based when usable again.
     *
     * \returns 0 always
     */
    int ttmStateUpdate( const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Handle a ttmperi `pos_N` update.
    /**
     * \returns 0 always
     */
    int ttmPosUpdate( int                      n, /**< [in] the axis index, 0 or 1 */
                      const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Start tracking, capturing the baseline.
    /**
     * \returns 0 on success or refusal
     * \returns -1 if a command could not be sent
     */
    int startTracking();

    /// Stop tracking, leaving ttmperi where it is.
    /**
     * \returns 0 always
     */
    int stopTracking();

    /** @name INDI
     *
     * @{
     */
  protected:
    pcf::IndiProperty m_indiP_qwp1Position; ///< The subscribed QWP1 stage `position` property.

    pcf::IndiProperty m_indiP_qwp1Fsm; ///< The subscribed QWP1 stage `fsm` property.

    pcf::IndiProperty m_indiP_qwp2Position; ///< The subscribed QWP2 stage `position` property.

    pcf::IndiProperty m_indiP_qwp2Fsm; ///< The subscribed QWP2 stage `fsm` property.

    pcf::IndiProperty m_indiP_ttmPos1; ///< The subscribed ttmperi `pos_1` property.

    pcf::IndiProperty m_indiP_ttmPos2; ///< The subscribed ttmperi `pos_2` property.

    pcf::IndiProperty m_indiP_ttmFsm; ///< The subscribed ttmperi `fsm` property.

    pcf::IndiProperty m_indiP_tracking; ///< The INDI toggle used to enable or disable tracking.

    /// The predicted offset, `dx` and `dy` in camwfs pixels, and its correction `v1` and `v2` in V.
    pcf::IndiProperty m_indiP_offset;

  public:
    /// Handle QWP1 stage position updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_qwp1Position );

    /// Handle QWP1 stage state updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_qwp1Fsm );

    /// Handle QWP2 stage position updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_qwp2Position );

    /// Handle QWP2 stage state updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_qwp2Fsm );

    /// Handle ttmperi axis 1 updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_ttmPos1 );

    /// Handle ttmperi axis 2 updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_ttmPos2 );

    /// Handle ttmperi state updates.
    INDI_SETCALLBACK_DECL( qwpPupilTracker, m_indiP_ttmFsm );

    /// Handle tracking toggle requests.
    INDI_NEWCALLBACK_DECL( qwpPupilTracker, m_indiP_tracking );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */

    /// Check whether telemetry records are due.
    /**
     * \returns the result of telemeter::checkRecordTimes
     */
    int checkRecordTimes();

    /// Record the qwppupil telemetry unconditionally.
    /**
     * \returns 0 always
     */
    int recordTelem( const telem_qwppupil * /**< [in] tag selecting the telemetry type */ );

    /// Record the qwppupil telemetry if it changed.  The caller must hold `m_indiMutex`.
    /**
     * \returns 0 always
     */
    int recordQwpPupil( bool force = false /**< [in] record even if nothing changed */ );

    ///@}
};

qwpPupilTracker::qwpPupilTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_qwp[0].devName = "stageqwprot1";
    m_qwp[1].devName = "stageqwprot2";

    return;
}

void qwpPupilTracker::setupConfig()
{
    for( int n = 0; n < 2; ++n )
    {
        std::string q = "qwp" + std::to_string( n + 1 );

        config.add( "stages." + q,
                    "",
                    "stages." + q,
                    argType::Required,
                    "stages",
                    q,
                    false,
                    "string",
                    "The device name of the " + q + " rotation stage.  Default is '" + m_qwp[n].devName + "'" );

        config.add( "reference." + q,
                    "",
                    "reference." + q,
                    argType::Required,
                    "reference",
                    q,
                    false,
                    "float",
                    "The " + q + " mechanical stage angle of the reference state, in degrees.  Default is " +
                        std::to_string( m_phiRef[n] ) );
    }

    config.add( "ttm.device",
                "",
                "ttm.device",
                argType::Required,
                "ttm",
                "device",
                false,
                "string",
                "The device name of the tip/tilt mirror.  Default is 'ttmperi'" );

    config.add( "ttm.rebaseTol",
                "",
                "ttm.rebaseTol",
                argType::Required,
                "ttm",
                "rebaseTol",
                false,
                "float",
                "A ttmperi target farther than this from every recent command is an external move, in V.  "
                "Default is 0.01" );

    config.add( "calib.model",
                "",
                "calib.model",
                argType::Required,
                "calib",
                "model",
                false,
                "string",
                "The pupil-shift model table, relative to the calib directory.  "
                "Default is 'qwptrack/qwp_pupil_model.csv'" );

    config.add( "calib.cm",
                "",
                "calib.cm",
                argType::Required,
                "calib",
                "cm",
                false,
                "string",
                "The ttmperi control matrix FITS file, relative to the calib directory.  "
                "Default is 'qwptrack/qwp_pupil_cm.fits'" );

    TELEMETER_SETUP_CONFIG( config );
}

int qwpPupilTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    for( int n = 0; n < 2; ++n )
    {
        std::string q = "qwp" + std::to_string( n + 1 );

        _config( m_qwp[n].devName, "stages." + q );
        _config( m_phiRef[n], "reference." + q );
    }

    _config( m_ttmDevName, "ttm.device" );
    _config( m_rebaseTol, "ttm.rebaseTol" );
    _config( m_modelFile, "calib.model" );
    _config( m_cmFile, "calib.cm" );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void qwpPupilTracker::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

int qwpPupilTracker::appStartup()
{
    if( loadCalib() < 0 )
    {
        return -1;
    }

    REG_INDI_SETPROP( m_indiP_qwp1Position, m_qwp[0].devName, "position" );

    REG_INDI_SETPROP( m_indiP_qwp1Fsm, m_qwp[0].devName, "fsm" );

    REG_INDI_SETPROP( m_indiP_qwp2Position, m_qwp[1].devName, "position" );

    REG_INDI_SETPROP( m_indiP_qwp2Fsm, m_qwp[1].devName, "fsm" );

    REG_INDI_SETPROP( m_indiP_ttmPos1, m_ttmDevName, "pos_1" );

    REG_INDI_SETPROP( m_indiP_ttmPos2, m_ttmDevName, "pos_2" );

    REG_INDI_SETPROP( m_indiP_ttmFsm, m_ttmDevName, "fsm" );

    for( int n = 0; n < 2; ++n )
    {
        m_ttm[n].cmd = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_ttm[n].cmd.setDevice( m_ttmDevName );
        m_ttm[n].cmd.setName( "pos_" + std::to_string( n + 1 ) );
        m_ttm[n].cmd.add( pcf::IndiElement( "target" ) );
    }

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_tracking, "tracking" );

    if( createROIndiNumber( m_indiP_offset, "offset", "Pupil Offset" ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    for( const auto *el : { "dx", "dy", "v1", "v2" } )
    {
        m_indiP_offset.add( pcf::IndiElement( el ) );
        m_indiP_offset[el] = 0.0;
    }

    if( registerIndiPropertyReadOnly( m_indiP_offset ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    TELEMETER_APP_STARTUP;

    state( stateCodes::NOTCONNECTED );

    return 0;
}

int qwpPupilTracker::appLogic()
{
    update();

    TELEMETER_APP_LOGIC;

    return 0;
}

int qwpPupilTracker::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int qwpPupilTracker::loadCalib()
{
    if( loadModel( m_calibDir + "/" + m_modelFile ) < 0 || loadCM( m_calibDir + "/" + m_cmFile ) < 0 )
    {
        return -1;
    }

    shift( m_shiftRef[0], m_shiftRef[1], m_model, m_phiRef[0], m_phiRef[1] );

    log<text_log>( "reference stage angles " + std::to_string( m_phiRef[0] ) + ", " + std::to_string( m_phiRef[1] ) +
                   ", shift " + std::to_string( m_shiftRef[0] ) + ", " + std::to_string( m_shiftRef[1] ) + " px" );

    return 0;
}

int qwpPupilTracker::loadModel( const std::string &path )
{
    m_model = shiftModel();

    if( mx::ioutils::readColumns<mx::ioutils::readColCommaDelim>(
            path, m_model.a, m_model.b, m_model.dxCos, m_model.dxSin, m_model.dyCos, m_model.dySin ) !=
        mx::error_t::noerror )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "error reading pupil-shift model " + path } );
    }

    size_t N = m_model.a.size();

    if( N == 0 || m_model.b.size() != N || m_model.dxCos.size() != N || m_model.dxSin.size() != N ||
        m_model.dyCos.size() != N || m_model.dySin.size() != N )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "pupil-shift model needs at least one complete row in " + path } );
    }

    // readColumns fills a short row with a placeholder, so count the columns of each data row
    std::ifstream fin( path );
    std::string   line;
    size_t        row = 0;

    while( std::getline( fin, line ) )
    {
        line = line.substr( 0, line.find( '#' ) );

        if( line.find_first_not_of( " \t\r" ) == std::string::npos )
        {
            continue;
        }

        if( std::count( line.begin(), line.end(), ',' ) != 5 )
        {
            return log<software_critical, -1>(
                { __FILE__, __LINE__, "row " + std::to_string( row ) + " does not have 6 columns in " + path } );
        }

        ++row;
    }

    for( size_t n = 0; n < N; ++n )
    {
        if( m_model.a[n] != std::round( m_model.a[n] ) || m_model.b[n] != std::round( m_model.b[n] ) )
        {
            return log<software_critical, -1>(
                { __FILE__, __LINE__, "non-integer frequency at row " + std::to_string( n ) + " in " + path } );
        }

        for( double v :
             { m_model.a[n], m_model.b[n], m_model.dxCos[n], m_model.dxSin[n], m_model.dyCos[n], m_model.dySin[n] } )
        {
            if( !std::isfinite( v ) )
            {
                return log<software_critical, -1>(
                    { __FILE__, __LINE__, "non-finite value at row " + std::to_string( n ) + " in " + path } );
            }
        }
    }

    log<text_log>( "read pupil-shift model from " + path + ": " + std::to_string( N ) + " frequencies" );

    return 0;
}

int qwpPupilTracker::loadCM( const std::string &path )
{
    mx::fits::fitsFile<double>     ff;
    mx::improc::eigenImage<double> im;

    if( ff.read( im, path ) != mx::error_t::noerror )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "error reading control matrix " + path } );
    }

    if( im.rows() != 2 || im.cols() != 2 || !im.allFinite() )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "control matrix must be 2x2 and finite in " + path } );
    }

    // FITS axis 1 is the fastest (python's last index), so im(px, v) == CM[v][px]
    for( int v = 0; v < 2; ++v )
    {
        for( int p = 0; p < 2; ++p )
        {
            m_cm[v][p] = im( p, v );
        }
    }

    log<text_log>( "read control matrix from " + path + ": [[" + std::to_string( m_cm[0][0] ) + ", " +
                   std::to_string( m_cm[0][1] ) + "], [" + std::to_string( m_cm[1][0] ) + ", " +
                   std::to_string( m_cm[1][1] ) + "]] V/px" );

    return 0;
}

void qwpPupilTracker::shift( double &dx, double &dy, const shiftModel &model, double phi1, double phi2 )
{
    double p1 = phi1 * M_PI / 180.0;
    double p2 = phi2 * M_PI / 180.0;

    dx = 0;
    dy = 0;

    for( size_t n = 0; n < model.a.size(); ++n )
    {
        double arg = model.a[n] * p1 + model.b[n] * p2;
        double c   = std::cos( arg );
        double s   = std::sin( arg );

        dx += model.dxCos[n] * c + model.dxSin[n] * s;
        dy += model.dyCos[n] * c + model.dySin[n] * s;
    }
}

bool qwpPupilTracker::usable( stateCodes::stateCodeT st )
{
    return st == stateCodes::READY || st == stateCodes::OPERATING;
}

int qwpPupilTracker::update()
{
    double v[2]    = { 0, 0 };
    bool   send[2] = { false, false };

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        bool stagesOK = stagesUsable();
        bool ttmOK    = ttmUsable();

        if( stagesOK )
        {
            computeOffset();
        }

        if( m_tracking && ttmOK && m_rebasePending )
        {
            m_rebasePending = false;

            rebase( 0 );
            rebase( 1 );

            log<text_log>( m_ttmDevName + " returned while tracking, re-based on its targets" );
        }

        for( int n = 0; m_tracking && stagesOK && ttmOK && n < 2; ++n )
        {
            ttmAxis &ax = m_ttm[n];

            v[n] = ax.v0 + m_dv[n];

            // v0 + dv need not round back to exactly the last command, so compare with a tolerance
            if( ax.haveSent && std::fabs( v[n] - ax.lastSent ) <= 1e-6 )
            {
                continue;
            }

            ax.lastSent = v[n];
            ax.haveSent = true;

            ax.recent.push_back( v[n] );

            if( ax.recent.size() > 16 )
            {
                ax.recent.erase( ax.recent.begin() );
            }

            send[n] = true;
        }

        if( m_tracking )
        {
            state( stateCodes::OPERATING );
        }
        else
        {
            state( ( stagesOK && ttmOK ) ? stateCodes::READY : stateCodes::NOTCONNECTED );
        }

        recordQwpPupil();
    } // mutex scope

    int rv = 0;

    for( int n = 0; n < 2; ++n )
    {
        if( send[n] && sendTTM( n, v[n] ) < 0 )
        {
            rv = -1;
        }
    }

    return rv;
}

bool qwpPupilTracker::stagesUsable()
{
    return m_qwp[0].havePos && m_qwp[1].havePos && usable( m_qwp[0].stageState ) && usable( m_qwp[1].stageState );
}

bool qwpPupilTracker::ttmUsable()
{
    return m_ttm[0].haveTgt && m_ttm[1].haveTgt && usable( m_ttmState );
}

void qwpPupilTracker::computeOffset()
{
    shift( m_offset[0], m_offset[1], m_model, m_qwp[0].pos, m_qwp[1].pos );

    m_offset[0] -= m_shiftRef[0];
    m_offset[1] -= m_shiftRef[1];

    m_dv[0] = -( m_cm[0][0] * m_offset[0] + m_cm[0][1] * m_offset[1] );
    m_dv[1] = -( m_cm[1][0] * m_offset[0] + m_cm[1][1] * m_offset[1] );

    updatesIfChanged( m_indiP_offset,
                      std::vector<const char *>( { "dx", "dy", "v1", "v2" } ),
                      std::vector<double>( { m_offset[0], m_offset[1], m_dv[0], m_dv[1] } ) );
}

void qwpPupilTracker::rebase( int n )
{
    ttmAxis &ax = m_ttm[n];

    ax.v0       = ax.tgt - m_dv[n];
    ax.lastSent = ax.tgt;
    ax.haveSent = true;
    ax.recent.clear();
}

int qwpPupilTracker::sendTTM( int n, double v )
{
    ++m_ttm[n].nSent;

#ifdef QWPPUPILTRACKER_TEST_NOSEND
    static_cast<void>( v );
    return 0; // unit tests inspect m_ttm[n].lastSent and nSent instead of contacting an indiserver
#else
    if( sendNewProperty( m_ttm[n].cmd, "target", v ) < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "failed to send target to " + m_ttm[n].cmd.createUniqueKey() } );
    }

    return 0;
#endif
}

int qwpPupilTracker::stagePosUpdate( int n, const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    double pos = ipRecv["current"].get<double>();

    if( !std::isfinite( pos ) )
    {
        return log<software_error, 0>(
            { __FILE__, __LINE__, "received non-finite position from " + ipRecv.getDevice() } );
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        m_qwp[n].pos     = pos;
        m_qwp[n].havePos = true;
    } // mutex scope

    return update();
}

bool qwpPupilTracker::parseState( stateCodes::stateCodeT &st, const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.find( "state" ) )
    {
        return false;
    }

    std::string str = ipRecv["state"].getValue();

    if( str.empty() )
    {
        return false;
    }

    st = stateCodes::str2CodeFast( str );

    return true;
}

int qwpPupilTracker::stageStateUpdate( int n, const pcf::IndiProperty &ipRecv )
{
    stateCodes::stateCodeT st;

    if( !parseState( st, ipRecv ) )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_qwp[n].stageState = st;

    return 0;
}

int qwpPupilTracker::ttmStateUpdate( const pcf::IndiProperty &ipRecv )
{
    stateCodes::stateCodeT st;

    if( !parseState( st, ipRecv ) )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_ttmState = st;

    if( m_tracking && !usable( st ) )
    {
        m_rebasePending = true;
    }

    return 0;
}

int qwpPupilTracker::ttmPosUpdate( int n, const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    double tgt = ipRecv["target"].get<double>();

    if( !std::isfinite( tgt ) )
    {
        return log<software_error, 0>( { __FILE__, __LINE__, "received non-finite target from " + m_ttmDevName } );
    }

    bool external = false;

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        ttmAxis &ax = m_ttm[n];

        bool changed = !ax.haveTgt || tgt != ax.tgt;

        ax.tgt     = tgt;
        ax.haveTgt = true;

        if( !m_tracking || !changed )
        {
            return 0;
        }

        // while ttmperi is unavailable its targets are not trusted; re-base once it returns
        if( !usable( m_ttmState ) )
        {
            m_rebasePending = true;
            return 0;
        }

        for( double r : ax.recent )
        {
            if( std::fabs( tgt - r ) <= m_rebaseTol )
            {
                return 0;
            }
        }

        if( ax.haveSent && std::fabs( tgt - ax.lastSent ) <= m_rebaseTol )
        {
            return 0;
        }

        rebase( n );
        external = true;
    } // mutex scope

    if( external )
    {
        log<text_log>( m_ttmDevName + " pos_" + std::to_string( n + 1 ) + " moved externally to " +
                       std::to_string( tgt ) + ", re-based" );
    }

    return 0;
}

int qwpPupilTracker::startTracking()
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( !stagesUsable() || !ttmUsable() )
        {
            updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

            return log<text_log, 0>( "refused to start tracking: QWP stages and " + m_ttmDevName +
                                         " must all report positions and be READY or OPERATING",
                                     logPrio::LOG_WARNING );
        }

        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::On, INDI_IDLE );

        computeOffset();

        // ttmperi is aligned now, so its targets are the commands for the current offset
        rebase( 0 );
        rebase( 1 );

        m_tracking      = true;
        m_rebasePending = false;
    } // mutex scope

    log<text_log>( "started QWP pupil tracking, baseline " + std::to_string( m_ttm[0].v0 ) + ", " +
                   std::to_string( m_ttm[1].v0 ) + " V" );

    return update();
}

int qwpPupilTracker::stopTracking()
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        m_tracking      = false;
        m_rebasePending = false;
    } // mutex scope

    log<text_log>( "stopped QWP pupil tracking" );

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_qwp1Position )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp1Position, ipRecv );

    return stagePosUpdate( 0, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_qwp1Fsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp1Fsm, ipRecv );

    return stageStateUpdate( 0, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_qwp2Position )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp2Position, ipRecv );

    return stagePosUpdate( 1, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_qwp2Fsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp2Fsm, ipRecv );

    return stageStateUpdate( 1, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_ttmPos1 )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ttmPos1, ipRecv );

    return ttmPosUpdate( 0, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_ttmPos2 )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ttmPos2, ipRecv );

    return ttmPosUpdate( 1, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpPupilTracker, m_indiP_ttmFsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ttmFsm, ipRecv );

    return ttmStateUpdate( ipRecv );
}

INDI_NEWCALLBACK_DEFN( qwpPupilTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tracking, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        return startTracking();
    }

    return stopTracking();
}

int qwpPupilTracker::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_qwppupil() );
}

int qwpPupilTracker::recordTelem( const telem_qwppupil * )
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    return recordQwpPupil( true );
}

int qwpPupilTracker::recordQwpPupil( bool force )
{
    float off[2] = { static_cast<float>( m_offset[0] ), static_cast<float>( m_offset[1] ) };
    float dv[2]  = { static_cast<float>( m_dv[0] ), static_cast<float>( m_dv[1] ) };

    if( force || m_tracking != m_telemTracking || off[0] != m_telemOffset[0] || off[1] != m_telemOffset[1] ||
        dv[0] != m_telemDv[0] || dv[1] != m_telemDv[1] )
    {
        telem<telem_qwppupil>( { m_tracking, off[0], off[1], dv[0], dv[1] } );

        m_telemTracking  = m_tracking;
        m_telemOffset[0] = off[0];
        m_telemOffset[1] = off[1];
        m_telemDv[0]     = dv[0];
        m_telemDv[1]     = dv[1];
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // qwpPupilTracker_hpp
