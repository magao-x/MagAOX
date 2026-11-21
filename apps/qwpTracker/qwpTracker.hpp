/** \file qwpTracker.hpp
 * \brief The MagAO-X dual QWP rotation tracker header file
 *
 * \ingroup qwpTracker_files
 */

#ifndef qwpTracker_hpp
#define qwpTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <cmath>
#include <filesystem>
#include <map>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/ioutils/readColumns.hpp>
#include <mx/math/gslInterpolator.hpp>

/** \defgroup qwpTracker
 * \brief The MagAO-X application to track pupil rotation with the dual rotating QWP compensator.
 *
 * <a href="../handbook/operating/software/apps/qwpTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup qwpTracker_files
 * \ingroup qwpTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X dual QWP rotation tracker.
/** Reads the K-mirror angle from `stagek` and commands the fast-axis angles of the two quarter-wave plates on
 * `stageqwprot1` and `stageqwprot2`. This app owns the mapping between fast-axis (polarization) angle and mechanical
 * stage angle, `stage = zero + sign * fastAxis`, so that the stage apps deal only in mechanical angles.
 *
 * A QWP is 180 degree periodic, so each fast-axis target is sent as whichever of the two equivalent stage angles is
 * nearest the stage's current position.  Fast-axis angles are reported in [0, 180).  Targets are sent only to a
 * READY stage; a target that changes while a stage is moving is held and the latest is sent when it stops, since the
 * ELL21 mounts do not reliably accept a new target mid-move.  A stage that stops more than `retry.tolerance` from its
 * last command is sent that command once more.
 *
 * While tracking, targets follow `computeTargets()` on every K-mirror update.  When not tracking, targets are set
 * manually through the `qwp1` and `qwp2` properties.
 *
 * The tracking law is a per-filter lookup table of fast-axis angles against K-mirror angle, read at startup from
 * `<calib>/<law.dir>/qwp_law_<filter>.csv`, one file per filter named after the filter wheel's (default `fwsci1`)
 * filter names.  Each file has comma-separated columns `imr, qwp1, qwp2` in degrees, with `imr` the K-mirror
 * `position.current` (strictly increasing) and the QWP angles unwrapped so they can be linearly interpolated; lines
 * starting with `#` are comments.  The tables are produced from the polarimetric calibration and kept in the calib
 * repo.  While the filter has no law, the wheel is moving, or the K-mirror is outside the table, tracking holds.
 *
 * \ingroup qwpTracker
 */
class qwpTracker : public MagAOXApp<true>, public dev::telemeter<qwpTracker>
{

    // Give the test harness access.
    friend class qwpTracker_test;

    friend class dev::telemeter<qwpTracker>;

    typedef dev::telemeter<qwpTracker> telemeterT;

  public:
    /// The state of one QWP and its rotation stage.
    struct qwpStage
    {
        std::string devName; ///< The INDI device name of the rotation stage.

        float zero{ 0 }; ///< The stage angle, in degrees, at which the fast axis is at 0.

        int sign{ 1 }; ///< The sign relating fast-axis rotation to stage rotation, +1 or -1.

        stateCodes::stateCodeT stageState{ stateCodes::NOTCONNECTED }; ///< The last state reported by the stage.

        bool havePos{ false }; ///< True once the stage has reported a position.

        float stagePos{ 0 }; ///< The last stage position reported by the stage, in degrees.

        float fastCur{ 0 }; ///< The current fast-axis angle, in degrees in [0, 180).

        float fastTgt{ 0 }; ///< The target fast-axis angle, in degrees in [0, 180).

        bool pending{ false }; ///< True when `fastTgt` still has to be sent to the stage.

        bool haveSent{ false }; ///< True when `lastSent` holds a command that need not be repeated.

        float lastSent{ 0 }; ///< The last stage angle commanded, in degrees.

        bool settled{ false }; ///< True when the stage went from OPERATING to READY and has not been checked yet.

        bool retried{ false }; ///< True when `lastSent` has already been resent once, or must not be resent.

        int nSent{ 0 }; ///< The number of stage commands issued, for diagnostics and tests.

        pcf::IndiProperty cmd; ///< The outbound `position.target` command property for the stage.
    };

    /// The tracking law for one filter, loaded from the calib directory.
    /** The interpolators keep pointers into the vectors, so a law is built in place and never copied or moved.
     */
    struct qwpLaw
    {
        std::vector<double> imr; ///< The K-mirror angles of the table rows, strictly increasing, in degrees.

        std::vector<double> qwp[2]; ///< The QWP1 and QWP2 fast-axis angles at each `imr`, unwrapped, in degrees.

        /// Linear interpolators of `qwp[n]` against `imr`.
        mx::math::gslInterpolator<mx::math::gsl_interp_linear<double>> terp[2];

        /// Default c'tor.
        qwpLaw() = default;

        /// Deleted copy c'tor, since the interpolators hold pointers into the vectors.
        qwpLaw( const qwpLaw & ) = delete;

        /// Deleted assignment, since the interpolators hold pointers into the vectors.
        qwpLaw &operator=( const qwpLaw & ) = delete;
    };

  protected:
    /** \name Configurable Parameters
     *@{
     */

    qwpStage m_qwp[2]; ///< QWP1 and QWP2: calibration from config, plus stage and target state.

    std::string m_kDevName{ "stagek" }; ///< The device name of the K-mirror stage providing `position.current`.

    std::string m_filterDevName{ "fwsci1" }; ///< The device name of the filter wheel selecting the tracking law.

    std::string m_lawDir{ "qwptrack" }; ///< The directory, relative to the calib directory, of the law tables.

    /// The distance, in degrees, from the last command beyond which a stopped stage has missed its target.
    float m_retryTol{ 0.5 };

    ///@}

    std::map<std::string, qwpLaw> m_laws; ///< The tracking laws loaded at startup, keyed by filter name.

    std::string m_filter; ///< The filter selected on the filter wheel, empty while moving or unknown.

    bool m_lawOK{ true }; ///< Whether the last tracking update had a law, so holds are logged once.

    bool m_tracking{ false }; ///< True when the QWPs follow the tracking law.

    float m_k{ 0 }; ///< The most recent K-mirror angle, in degrees.

    bool m_haveK{ false }; ///< True once at least one K-mirror angle has been received.

    bool m_recompute{ false }; ///< True when a K-mirror angle or filter change has not yet been applied by the law.

    float m_telemQwp1{ 0 }; ///< The QWP1 fast-axis angle in the last telemetry record.

    float m_telemQwp2{ 0 }; ///< The QWP2 fast-axis angle in the last telemetry record.

    bool m_telemTracking{ false }; ///< The tracking flag in the last telemetry record.

  public:
    /// Default c'tor.
    qwpTracker();

    /// D'tor, declared and defined for noexcept.
    ~qwpTracker() noexcept
    {
    }

    /// Set up configuration entries.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     *
     * \returns 0 on success
     * \returns -1 if a sign is not +1 or -1
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load configuration values, shutting down on an invalid configuration.
    virtual void loadConfig();

    /// Register the INDI properties and telemetry.
    virtual int appStartup();

    /// Implementation of the FSM for qwpTracker.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    virtual int appShutdown();

    /** \name Tracking Law and Angle Mapping
     *@{
     */

    /// Load and validate every `qwp_law_<filter>.csv` table in the law directory.
    /** A missing directory or no tables is not an error: tracking then holds.
     *
     * \returns 0 on success, including when no tables are found
     * \returns -1 if a table cannot be read or is invalid
     */
    int loadLaws();

    /// The tracking law: interpolate the QWP fast-axis targets for the selected filter at a K-mirror angle.
    /** The caller must hold `m_indiMutex`.
     *
     * \returns true if the targets were computed
     * \returns false if the selected filter has no law or `k` is outside its table
     */
    bool computeTargets( float  k,    /**< [in] the K-mirror angle, in degrees */
                         float &qwp1, /**< [out] the QWP1 fast-axis target, unwrapped, in degrees */
                         float &qwp2 /**< [out] the QWP2 fast-axis target, unwrapped, in degrees */ );

    /// Wrap an angle to [0, 360).
    /**
     * \returns the wrapped angle in degrees
     */
    static float wrap360( float deg /**< [in] the angle, in degrees */ );

    /// Fold an angle to [0, 180), the period of a QWP fast axis.
    /**
     * \returns the folded angle in degrees
     */
    static float fold180( float deg /**< [in] the angle, in degrees */ );

    /// Convert a stage angle to a fast-axis angle.
    /**
     * \returns `sign * (stage - zero)` folded to [0, 180)
     */
    static float stageToFast( float stage, /**< [in] the stage angle, in degrees */
                              float zero,  /**< [in] the stage angle at which the fast axis is at 0 */
                              int   sign /**< [in] the sign relating fast-axis to stage rotation */ );

    /// Convert a fast-axis angle to the nearest equivalent stage angle.
    /** The candidates are `wrap360(zero + sign * fast)` and that angle + 180.
     *
     * \returns the candidate, in [0, 360), with the smaller circular distance from `stageCur`
     */
    static float fastToStage( float fast, /**< [in] the fast-axis angle, in degrees */
                              float zero, /**< [in] the stage angle at which the fast axis is at 0 */
                              int   sign, /**< [in] the sign relating fast-axis to stage rotation */
                              float stageCur /**< [in] the current stage angle, in degrees */ );

    /// Derive the tracker FSM state from the tracking flag and the two stage states.
    /**
     * \returns OPERATING if tracking or either stage is OPERATING
     * \returns READY if both stages are READY
     * \returns otherwise the worse stage state, with anything other than HOMING or NOTHOMED reported as NOTCONNECTED
     */
    static stateCodes::stateCodeT trackerState( bool                   tracking, /**< [in] whether tracking is on */
                                                stateCodes::stateCodeT stage1,   /**< [in] the QWP1 stage state */
                                                stateCodes::stateCodeT stage2 /**< [in] the QWP2 stage state */ );

    ///@}

  protected:
    /// Compute tracking targets if needed, send pending targets to READY stages, and update FSM and telemetry.
    /** Takes `m_indiMutex`; the caller must not hold it.  Commands are sent after the mutex is released.  A stage that
     * is moving (OPERATING) keeps its target pending, and receives only the latest target once it is READY.  A stage
     * that has stopped more than `m_retryTol` from its last command, with no newer target pending, is sent that
     * command once more.
     *
     * \returns 0 on success
     * \returns -1 if a stage command could not be sent
     */
    int updateStages();

    /// Send a stage target to one QWP stage.
    /**
     * \returns 0 on success
     * \returns -1 on an error from sendNewProperty
     */
    int sendStageTarget( int   n, /**< [in] the QWP index, 0 or 1 */
                         float deg /**< [in] the stage angle to command, in degrees */ );

    /// The app's `qwp1` or `qwp2` property.
    /**
     * \returns `m_indiP_qwp1` for n == 0, otherwise `m_indiP_qwp2`
     */
    pcf::IndiProperty &qwpProp( int n /**< [in] the QWP index, 0 or 1 */ );

    /// Handle a stage `position` update.
    /**
     * \returns 0 always
     */
    int stagePosUpdate( int                      n, /**< [in] the QWP index, 0 or 1 */
                        const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Handle a stage `fsm` update.
    /**
     * \returns 0 always
     */
    int stageStateUpdate( int                      n, /**< [in] the QWP index, 0 or 1 */
                          const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Handle a filter wheel `filterName` update.
    /** The selected filter is the single `On` element while the property is not busy, otherwise none.
     *
     * \returns 0 always
     */
    int filterUpdate( const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Handle a manual fast-axis target request.
    /** Rejected while tracking.
     *
     * \returns 0 on success or rejection
     * \returns -1 if the target could not be sent
     */
    int manualTarget( int                      n, /**< [in] the QWP index, 0 or 1 */
                      const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /** @name INDI
     *
     * @{
     */
  protected:
    pcf::IndiProperty m_indiP_stagek; ///< The subscribed K-mirror stage `position` property.

    pcf::IndiProperty m_indiP_qwp1Position; ///< The subscribed QWP1 stage `position` property.

    pcf::IndiProperty m_indiP_qwp1Fsm; ///< The subscribed QWP1 stage `fsm` property.

    pcf::IndiProperty m_indiP_qwp2Position; ///< The subscribed QWP2 stage `position` property.

    pcf::IndiProperty m_indiP_qwp2Fsm; ///< The subscribed QWP2 stage `fsm` property.

    pcf::IndiProperty m_indiP_filterName; ///< The subscribed filter wheel `filterName` property.

    pcf::IndiProperty m_indiP_law; ///< The filter name of the active tracking law, or `none`.

    pcf::IndiProperty m_indiP_tracking; ///< The INDI toggle used to enable or disable tracking.

    pcf::IndiProperty m_indiP_qwp1; ///< The QWP1 fast-axis angle, `current` and `target`, in degrees.

    pcf::IndiProperty m_indiP_qwp2; ///< The QWP2 fast-axis angle, `current` and `target`, in degrees.

  public:
    /// Handle K-mirror position updates.
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_stagek );

    /// Handle QWP1 stage position updates.
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_qwp1Position );

    /// Handle QWP1 stage state updates.
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_qwp1Fsm );

    /// Handle QWP2 stage position updates.
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_qwp2Position );

    /// Handle QWP2 stage state updates.
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_qwp2Fsm );

    /// Handle filter wheel selection updates.
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_filterName );

    /// Handle tracking toggle requests.
    INDI_NEWCALLBACK_DECL( qwpTracker, m_indiP_tracking );

    /// Handle manual QWP1 fast-axis target requests.
    INDI_NEWCALLBACK_DECL( qwpTracker, m_indiP_qwp1 );

    /// Handle manual QWP2 fast-axis target requests.
    INDI_NEWCALLBACK_DECL( qwpTracker, m_indiP_qwp2 );

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

    /// Record the qwptrack telemetry unconditionally.
    /**
     * \returns 0 always
     */
    int recordTelem( const telem_qwptrack * /**< [in] tag selecting the telemetry type */ );

    /// Record the qwptrack telemetry if it changed.
    /**
     * \returns 0 always
     */
    int recordQwpTrack( bool force = false /**< [in] record even if nothing changed */ );

    ///@}
};

qwpTracker::qwpTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_qwp[0].devName = "stageqwprot1";
    m_qwp[1].devName = "stageqwprot2";
    m_qwp[1].sign    = -1;

    return;
}

void qwpTracker::setupConfig()
{
    for( int n = 0; n < 2; ++n )
    {
        std::string sec = "qwp" + std::to_string( n + 1 );

        config.add( sec + ".zero",
                    "",
                    sec + ".zero",
                    argType::Required,
                    sec,
                    "zero",
                    false,
                    "float",
                    "The stage angle at which the " + sec + " fast axis is at 0.  Default is 0." );

        config.add( sec + ".sign",
                    "",
                    sec + ".sign",
                    argType::Required,
                    sec,
                    "sign",
                    false,
                    "int",
                    "The " + sec + " stage rotation sign, +1 or -1. Default is " + std::to_string( m_qwp[n].sign ) +
                        "." );

        config.add( sec + ".devName",
                    "",
                    sec + ".devName",
                    argType::Required,
                    sec,
                    "devName",
                    false,
                    "string",
                    "The device name of the " + sec + " stage.  Default is '" + m_qwp[n].devName + "'" );
    }

    config.add( "k.devName",
                "",
                "k.devName",
                argType::Required,
                "k",
                "devName",
                false,
                "string",
                "The device name of the K-mirror stage.  Default is 'stagek'" );

    config.add( "filter.devName",
                "",
                "filter.devName",
                argType::Required,
                "filter",
                "devName",
                false,
                "string",
                "The device name of the filter wheel selecting the tracking law.  Default is 'fwsci1'" );

    config.add( "law.dir",
                "",
                "law.dir",
                argType::Required,
                "law",
                "dir",
                false,
                "string",
                "The directory, relative to the calib directory, holding the qwp_law_<filter>.csv tracking law "
                "tables.  Default is 'qwptrack'" );

    config.add(
        "retry.tolerance",
        "",
        "retry.tolerance",
        argType::Required,
        "retry",
        "tolerance",
        false,
        "float",
        "The distance, in degrees, from its last command beyond which a stopped stage is sent that command once "
        "more.  Default is 0.5, matching elliptecLowLevel motion.targetTolerance." );

    TELEMETER_SETUP_CONFIG( config );
}

int qwpTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    for( int n = 0; n < 2; ++n )
    {
        std::string sec = "qwp" + std::to_string( n + 1 );

        _config( m_qwp[n].zero, sec + ".zero" );
        _config( m_qwp[n].sign, sec + ".sign" );
        _config( m_qwp[n].devName, sec + ".devName" );

        if( m_qwp[n].sign != 1 && m_qwp[n].sign != -1 )
        {
            return log<text_log, -1>( sec + ".sign must be +1 or -1", logPrio::LOG_CRITICAL );
        }
    }

    _config( m_kDevName, "k.devName" );
    _config( m_filterDevName, "filter.devName" );
    _config( m_lawDir, "law.dir" );
    _config( m_retryTol, "retry.tolerance" );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void qwpTracker::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

int qwpTracker::appStartup()
{
    if( loadLaws() < 0 )
    {
        return -1;
    }

    REG_INDI_SETPROP( m_indiP_stagek, m_kDevName, "position" );

    REG_INDI_SETPROP( m_indiP_qwp1Position, m_qwp[0].devName, "position" );

    REG_INDI_SETPROP( m_indiP_qwp1Fsm, m_qwp[0].devName, "fsm" );

    REG_INDI_SETPROP( m_indiP_qwp2Position, m_qwp[1].devName, "position" );

    REG_INDI_SETPROP( m_indiP_qwp2Fsm, m_qwp[1].devName, "fsm" );

    REG_INDI_SETPROP( m_indiP_filterName, m_filterDevName, "filterName" );

    for( auto &q : m_qwp )
    {
        q.cmd = pcf::IndiProperty( pcf::IndiProperty::Number );
        q.cmd.setDevice( q.devName );
        q.cmd.setName( "position" );
        q.cmd.add( pcf::IndiElement( "target" ) );
    }

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_tracking, "tracking" );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_qwp1, "qwp1", -360, 360, 1e-3, "%g", "QWP1 fast axis", "" );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_qwp2, "qwp2", -360, 360, 1e-3, "%g", "QWP2 fast axis", "" );

    if( createROIndiText( m_indiP_law, "law", "current", "Tracking Law", "", "Filter" ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    m_indiP_law["current"].set( "none" );

    if( registerIndiPropertyReadOnly( m_indiP_law ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    TELEMETER_APP_STARTUP;

    state( stateCodes::NOTCONNECTED );

    return 0;
}

int qwpTracker::appLogic()
{
    updateStages();

    TELEMETER_APP_LOGIC;

    return 0;
}

int qwpTracker::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int qwpTracker::loadLaws()
{
    m_laws.clear();

    std::string dir = m_calibDir + "/" + m_lawDir;

    std::vector<std::string> files;

    if( std::filesystem::is_directory( dir ) &&
        mx::ioutils::getFileNames( files, dir, "qwp_law_", "", ".csv" ) != mx::error_t::noerror )
    {
        return log<software_critical, -1>( "error listing tracking laws in " + dir );
    }

    for( const auto &file : files )
    {
        std::string filter = std::filesystem::path( file ).stem().string().substr( std::string( "qwp_law_" ).size() );

        qwpLaw &law = m_laws[filter];

        if( mx::ioutils::readColumns<mx::ioutils::readColCommaDelim>( file, law.imr, law.qwp[0], law.qwp[1] ) !=
            mx::error_t::noerror )
        {
            return log<software_critical, -1>( "error reading tracking law " + file );
        }

        if( law.imr.size() != law.qwp[0].size() || law.imr.size() != law.qwp[1].size() || law.imr.size() < 2 )
        {
            return log<software_critical, -1>( "tracking law needs at least two complete rows in " + file );
        }

        for( size_t n = 0; n < law.imr.size(); ++n )
        {
            if( !std::isfinite( law.imr[n] ) || !std::isfinite( law.qwp[0][n] ) || !std::isfinite( law.qwp[1][n] ) )
            {
                return log<software_critical, -1>( "non-finite value at row " + std::to_string( n ) + " in " + file );
            }

            if( n > 0 && law.imr[n] <= law.imr[n - 1] )
            {
                return log<software_critical, -1>( "K-mirror angles must be strictly increasing in " + file );
            }
        }

        try
        {
            law.terp[0].setup( law.imr, law.qwp[0] );
            law.terp[1].setup( law.imr, law.qwp[1] );
        }
        catch( const std::exception &e )
        {
            return log<software_critical, -1>( "exception setting up interpolators for " + file + ": " + e.what() );
        }

        log<text_log>( "read tracking law for filter " + filter + " from " + file + ": " +
                       std::to_string( law.imr.size() ) + " rows, K-mirror " + std::to_string( law.imr.front() ) +
                       " to " + std::to_string( law.imr.back() ) );
    }

    if( m_laws.empty() )
    {
        log<text_log>( "no tracking laws found in " + dir + ", tracking will hold", logPrio::LOG_WARNING );
    }

    return 0;
}

bool qwpTracker::computeTargets( float k, float &qwp1, float &qwp2 )
{
    auto it = m_laws.find( m_filter );

    if( it == m_laws.end() )
    {
        return false;
    }

    qwpLaw &law = it->second;

    if( k < law.imr.front() || k > law.imr.back() )
    {
        return false;
    }

    qwp1 = law.terp[0]( k );
    qwp2 = law.terp[1]( k );

    return true;
}

float qwpTracker::wrap360( float deg )
{
    float w = std::fmod( deg, 360.0f );

    if( w < 0 )
    {
        w += 360;
    }

    // fmod of a tiny negative number can round up to exactly 360
    if( w >= 360 )
    {
        w -= 360;
    }

    return w;
}

float qwpTracker::fold180( float deg )
{
    float f = std::fmod( deg, 180.0f );

    if( f < 0 )
    {
        f += 180;
    }

    if( f >= 180 )
    {
        f -= 180;
    }

    return f;
}

float qwpTracker::stageToFast( float stage, float zero, int sign )
{
    return fold180( sign * ( stage - zero ) );
}

float qwpTracker::fastToStage( float fast, float zero, int sign, float stageCur )
{
    float s0 = wrap360( zero + sign * fast );
    float s1 = wrap360( s0 + 180 );

    auto dist = []( float a, float b )
    {
        float d = wrap360( a - b );
        return std::min( d, 360 - d );
    };

    return ( dist( s0, stageCur ) <= dist( s1, stageCur ) ) ? s0 : s1;
}

stateCodes::stateCodeT
qwpTracker::trackerState( bool tracking, stateCodes::stateCodeT stage1, stateCodes::stateCodeT stage2 )
{
    if( tracking || stage1 == stateCodes::OPERATING || stage2 == stateCodes::OPERATING )
    {
        return stateCodes::OPERATING;
    }

    if( stage1 == stateCodes::READY && stage2 == stateCodes::READY )
    {
        return stateCodes::READY;
    }

    auto rank = []( stateCodes::stateCodeT s )
    {
        if( s == stateCodes::READY )
        {
            return 3;
        }
        else if( s == stateCodes::HOMING )
        {
            return 2;
        }
        else if( s == stateCodes::NOTHOMED )
        {
            return 1;
        }

        return 0;
    };

    stateCodes::stateCodeT worst = ( rank( stage1 ) <= rank( stage2 ) ) ? stage1 : stage2;

    if( rank( worst ) == 0 )
    {
        return stateCodes::NOTCONNECTED;
    }

    return worst;
}

int qwpTracker::updateStages()
{
    float tgt[2]   = { 0, 0 };
    bool  send[2]  = { false, false };
    bool  retry[2] = { false, false };

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( m_tracking && m_recompute )
        {
            m_recompute = false;

            float law[2];
            bool  ok = computeTargets( m_k, law[0], law[1] );

            if( ok != m_lawOK )
            {
                m_lawOK = ok;

                if( !ok && m_laws.count( m_filter ) == 0 )
                {
                    log<text_log>( "holding QWPs: no tracking law for filter '" + m_filter + "'",
                                   logPrio::LOG_WARNING );
                }
                else if( !ok )
                {
                    log<text_log>( "holding QWPs: K-mirror " + std::to_string( m_k ) +
                                       " outside the tracking law for " + m_filter,
                                   logPrio::LOG_WARNING );
                }
                else
                {
                    log<text_log>( "resumed QWP tracking with the law for filter " + m_filter );
                }
            }

            for( int n = 0; ok && n < 2; ++n )
            {
                m_qwp[n].fastTgt = fold180( law[n] );
                m_qwp[n].pending = true;
                updateIfChanged( qwpProp( n ), "target", m_qwp[n].fastTgt );
            }
        }

        for( int n = 0; n < 2; ++n )
        {
            qwpStage &q = m_qwp[n];

            // a stage that stopped away from its last command, with nothing newer to send, gets one more try
            if( q.settled )
            {
                q.settled = false;

                float d = wrap360( q.stagePos - q.lastSent );

                if( q.haveSent && !q.pending && !q.retried && std::min( d, 360 - d ) > m_retryTol )
                {
                    q.pending  = true;
                    q.haveSent = false;
                    retry[n]   = true;
                }
            }

            // the ELL21 does not reliably accept a new target while moving, so only command a stopped stage
            if( !q.pending || !q.havePos || q.stageState != stateCodes::READY )
            {
                continue;
            }

            q.pending = false;

            float s = fastToStage( q.fastTgt, q.zero, q.sign, q.stagePos );

            if( q.haveSent && s == q.lastSent )
            {
                continue;
            }

            q.lastSent = s;
            q.haveSent = true;
            q.retried  = retry[n];
            tgt[n]     = s;
            send[n]    = true;
        }

        state( trackerState( m_tracking, m_qwp[0].stageState, m_qwp[1].stageState ) );

        recordQwpTrack();
    } // mutex scope

    int rv = 0;

    for( int n = 0; n < 2; ++n )
    {
        if( send[n] && retry[n] )
        {
            log<text_log>( m_qwp[n].devName + " stopped away from its target, resending " + std::to_string( tgt[n] ),
                           logPrio::LOG_WARNING );
        }

        if( send[n] && sendStageTarget( n, tgt[n] ) < 0 )
        {
            rv = -1;
        }
    }

    return rv;
}

int qwpTracker::sendStageTarget( int n, float deg )
{
    ++m_qwp[n].nSent;

#ifdef QWPTRACKER_TEST_NOSEND
    static_cast<void>( deg );
    return 0; // unit tests inspect m_qwp[n].lastSent and nSent instead of contacting an indiserver
#else
    if( sendNewProperty( m_qwp[n].cmd, "target", deg ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to send target to " + m_qwp[n].devName } );
    }

    return 0;
#endif
}

pcf::IndiProperty &qwpTracker::qwpProp( int n )
{
    return ( n == 0 ) ? m_indiP_qwp1 : m_indiP_qwp2;
}

int qwpTracker::stagePosUpdate( int n, const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    float pos = ipRecv["current"].get<float>();

    if( !std::isfinite( pos ) )
    {
        return log<software_error, 0>(
            { __FILE__, __LINE__, "received non-finite position from " + ipRecv.getDevice() } );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    qwpStage &q = m_qwp[n];

    q.stagePos = pos;

    // round to two decimal points before folding, so 179.999 reports as 0
    q.fastCur = fold180( std::round( stageToFast( pos, q.zero, q.sign ) * 100 ) / 100 );

    updateIfChanged( qwpProp( n ), "current", q.fastCur );

    if( !q.havePos )
    {
        q.havePos = true;
        q.fastTgt = q.fastCur;
        updateIfChanged( qwpProp( n ), "target", q.fastTgt );
    }

    return 0;
}

int qwpTracker::stageStateUpdate( int n, const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.find( "state" ) )
    {
        return 0;
    }

    std::string st = ipRecv["state"].getValue();

    if( st.empty() )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    stateCodes::stateCodeT code = stateCodes::str2CodeFast( st );

    if( code == stateCodes::READY && m_qwp[n].stageState == stateCodes::OPERATING )
    {
        m_qwp[n].settled = true;
    }

    m_qwp[n].stageState = code;

    return 0;
}

int qwpTracker::filterUpdate( const pcf::IndiProperty &ipRecv )
{
    std::string filter;

    if( ipRecv.getState() != pcf::IndiProperty::Busy )
    {
        int nOn = 0;

        for( const auto &el : ipRecv.getElements() )
        {
            if( el.second.getSwitchState() == pcf::IndiElement::On )
            {
                filter = el.first;
                ++nOn;
            }
        }

        if( nOn != 1 )
        {
            filter.clear();
        }
    }

    std::string law;

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( filter == m_filter )
        {
            return 0;
        }

        m_filter    = filter;
        m_recompute = m_haveK;

        law = ( m_laws.count( filter ) > 0 ) ? filter : "none";

        updateIfChanged( m_indiP_law, "current", law );
    } // mutex scope

    log<text_log>( m_filterDevName + " filter " + ( filter.empty() ? "unknown" : filter ) + ", tracking law " + law );

    return 0;
}

int qwpTracker::manualTarget( int n, const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    float tgt = ipRecv["target"].get<float>();

    std::string name = "qwp" + std::to_string( n + 1 );

    if( !std::isfinite( tgt ) )
    {
        return log<software_error, 0>( { __FILE__, __LINE__, "non-finite " + name + " target" } );
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( m_tracking )
        {
            return log<text_log, 0>( "rejected manual " + name + " target while tracking", logPrio::LOG_WARNING );
        }

        qwpStage &q = m_qwp[n];

        q.fastTgt  = fold180( tgt );
        q.pending  = true;
        q.haveSent = false; // an explicit request is always sent

        updateIfChanged( qwpProp( n ), "target", q.fastTgt );
    } // mutex scope

    log<text_log>( name + " fast axis set to " + std::to_string( fold180( tgt ) ) );

    return updateStages();
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_stagek )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stagek, ipRecv );

    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    float k = ipRecv["current"].get<float>();

    if( !std::isfinite( k ) )
    {
        return log<software_error, 0>( { __FILE__, __LINE__, "received non-finite K-mirror position" } );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_k         = k;
    m_haveK     = true;
    m_recompute = true;

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_qwp1Position )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp1Position, ipRecv );

    return stagePosUpdate( 0, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_qwp1Fsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp1Fsm, ipRecv );

    return stageStateUpdate( 0, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_qwp2Position )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp2Position, ipRecv );

    return stagePosUpdate( 1, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_qwp2Fsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp2Fsm, ipRecv );

    return stageStateUpdate( 1, ipRecv );
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_filterName )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_filterName, ipRecv );

    return filterUpdate( ipRecv );
}

INDI_NEWCALLBACK_DEFN( qwpTracker, m_indiP_qwp1 )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp1, ipRecv );

    return manualTarget( 0, ipRecv );
}

INDI_NEWCALLBACK_DEFN( qwpTracker, m_indiP_qwp2 )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp2, ipRecv );

    return manualTarget( 1, ipRecv );
}

INDI_NEWCALLBACK_DEFN( qwpTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tracking, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );

            updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::On, INDI_IDLE );

            m_tracking  = true;
            m_recompute = m_haveK;
            m_lawOK     = true; // log a hold again on each start

            for( auto &q : m_qwp )
            {
                q.haveSent = false; // re-apply the law even if it matches the last command
            }
        } // mutex scope

        log<text_log>( "started QWP rotation tracking" );

        return updateStages();
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        m_tracking  = false;
        m_recompute = false;

        for( int n = 0; n < 2; ++n )
        {
            m_qwp[n].pending = false;
            m_qwp[n].retried = true; // stopping must not move the stages, including by a retry
            m_qwp[n].fastTgt = m_qwp[n].fastCur;
            updateIfChanged( qwpProp( n ), "target", m_qwp[n].fastTgt );
        }
    } // mutex scope

    log<text_log>( "stopped QWP rotation tracking" );

    return 0;
}

int qwpTracker::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_qwptrack() );
}

int qwpTracker::recordTelem( const telem_qwptrack * )
{
    return recordQwpTrack( true );
}

int qwpTracker::recordQwpTrack( bool force )
{
    if( force || m_qwp[0].fastCur != m_telemQwp1 || m_qwp[1].fastCur != m_telemQwp2 || m_tracking != m_telemTracking )
    {
        telem<telem_qwptrack>( { m_qwp[0].fastCur, m_qwp[1].fastCur, m_tracking } );

        m_telemQwp1     = m_qwp[0].fastCur;
        m_telemQwp2     = m_qwp[1].fastCur;
        m_telemTracking = m_tracking;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // qwpTracker_hpp
