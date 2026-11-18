/** \file hwpTracker.hpp
 * \brief The MagAO-X HWP rotation tracker header file
 *
 * \ingroup hwpTracker_files
 */

#ifndef hwpTracker_hpp
#define hwpTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <cmath>

#include <mx/math/geo.hpp>

/** \defgroup hwpTracker
 * \brief The MagAO-X application to track pupil rotation with the HWP.
 *
 * <a href="../handbook/operating/software/apps/hwpTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup hwpTracker_files
 * \ingroup hwpTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X HWP rotation tracker
/** Converts a requested HWP angle into a stage target, optionally adding a tracking offset computed from the
 * telescope altitude and parallactic angle.  Stage commands are only sent while the stage is READY; a move requested
 * while the stage is busy is queued and sent when it next reports READY.
 *
 * \ingroup hwpTracker
 */
class hwpTracker : public MagAOXApp<true>, public dev::telemeter<hwpTracker>
{

    // Give the test harness access.
    friend class hwpTracker_test;

    friend class dev::telemeter<hwpTracker>;

    typedef dev::telemeter<hwpTracker> telemeterT;

  public:
    static constexpr float posTolerance = 0.5; ///< Tolerance for matching a named HWP position [deg].

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    float m_zero{ 360 }; ///< The stage position at HWP angle 0 [deg].  Default is 360.

    int m_sign{ -1 }; ///< The sign relating HWP angle to stage position, +1 or -1.  Default is -1.

    std::string m_devName{ "stagepolrot" }; ///< The device name of the HWP stage.  Default is 'stagepolrot'.

    std::string m_tcsDevName{ "tcsi" }; ///< The device name of the TCS interface providing 'teldata'.  Default is 'tcsi'.

    float m_updateInterval{ 10 }; ///< The interval at which to update positions while tracking [sec].  Default is 10.

    int m_trackOffsetSign{ -1 }; ///< The sign applied to the parallactic angle term, +1 for Ns-W, -1 for Ns-E.

    float m_pupilOffset{ 0 }; ///< The pupil offset used for HWP tracking (must match twice the kTracker offset) [deg].

    ///@}

    /** \name Tracking State - Data
     * These members are protected by m_indiMutex.
     *@{
     */

    bool m_tracking{ false }; ///< True when the HWP tracking offset is applied.

    float m_altitude{ 0 }; ///< The last valid telescope altitude [deg].

    float m_parang{ 0 }; ///< The last valid parallactic angle [deg], unwrapped to be continuous while tracking.

    bool m_haveTelData{ false }; ///< True once a valid teldata update has been received.

    bool m_telDataValid{ true }; ///< Validity of the last teldata update, used to log only on transitions.

    float m_hwpSetPos{ 0 }; ///< The requested HWP angle, without tracking offset [deg].

    float m_hwpTrackingOffset{ 0 }; ///< The tracking offset applied in the last stage command [deg].

    float m_hwpCurPos{ 0 }; ///< The measured HWP angle with the tracking offset removed [deg].

    float m_hwpActualPos{ 0 }; ///< The measured HWP angle including the tracking offset [deg].

    std::string m_hwpPosName; ///< The name of the measured HWP position, or 'Unknown'.

    stateCodes::stateCodeT m_stageState{ stateCodes::NOTCONNECTED }; ///< The last state reported by the stage.

    bool m_haveStageState{ false }; ///< True once the stage has reported its state.

    bool m_movePending{ false }; ///< True when a stage move is waiting for the stage to become READY.

    double m_lastUpdate{ 0 }; ///< The time of the last periodic tracking update [sec].

    ///@}

    /** \name Telemetry - Data
     * The last recorded values, protected by m_indiMutex.
     *@{
     */

    float m_telemCurPos{ 0 }; ///< The last recorded measured HWP angle without offset [deg].

    float m_telemActualPos{ 0 }; ///< The last recorded measured HWP angle with offset [deg].

    std::string m_telemPosName; ///< The last recorded HWP position name.

    bool m_telemTracking{ false }; ///< The last recorded tracking flag.

    ///@}

  public:
    /// Default c'tor.
    hwpTracker();

    /// D'tor, declared and defined for noexcept.
    ~hwpTracker() noexcept
    {
    }

    /// Set up the configuration options.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     *
     * \returns 0 on success
     * \returns -1 on an invalid configuration
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load the configuration, shutting down on an invalid configuration.
    virtual void loadConfig();

    /// Register the INDI properties and set the initial state.
    /**
     * \returns 0 on success
     * \returns -1 on an error requiring shutdown
     */
    virtual int appStartup();

    /// Implementation of the FSM for hwpTracker.
    /** Sends periodic tracking updates and queued moves when the stage is READY, and updates the app state.
     *
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     * \returns 0 on success
     */
    virtual int appShutdown();

    /// Calculate the HWP tracking offset.
    /**
     * \returns the tracking offset [deg]
     */
    static float trackingOffset( float altitude,    /**< [in] the telescope altitude [deg] */
                                 float parang,      /**< [in] the parallactic angle [deg] */
                                 int   trackSign,   /**< [in] the sign of the parallactic angle term */
                                 float pupilOffset  /**< [in] the pupil offset [deg] */ );

    /// Convert a HWP angle to a stage target in [0, 360).
    /**
     * \returns the stage target [deg]
     */
    static float stageTarget( float hwpPos, /**< [in] the HWP angle including any tracking offset [deg] */
                              float zero,   /**< [in] the stage position at HWP angle 0 [deg] */
                              int   sign    /**< [in] the sign relating HWP angle to stage position */ );

    /// Shift an angle by a multiple of 360 degrees to lie within 180 degrees of a reference.
    /**
     * \returns the shifted angle [deg]
     */
    static float nearestAngle( float angle, /**< [in] the angle to shift [deg] */
                               float ref    /**< [in] the reference angle [deg] */ );

    /// Get the name of a HWP angle, matching the named positions modulo 90 degrees.
    /**
     * \returns one of Qplus, Qminus, Uplus, Uminus, or Unknown
     */
    static std::string positionName( float hwpPos /**< [in] the HWP angle without tracking offset [deg] */ );

  protected:
    /// Command the stage to the current setpoint, or queue the move if the stage is not READY.
    /** Must be called without m_indiMutex held.
     *
     * \returns 0 on success or when queued
     * \returns -1 on an error sending the command
     */
    int requestMove();

    /// Send a target to the stage.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    virtual int sendStageTarget( float target /**< [in] the stage target [deg] */ );

    /// Determine the app state from the stage state, teldata receipt and tracking.
    /** Must be called without m_indiMutex held.
     *
     * \returns the state the app should be in
     */
    stateCodes::stateCodeT trackerState();

    /** @name INDI
     *
     * @{
     */
  protected:
    pcf::IndiProperty m_indiP_tracking; ///< Toggle to enable or disable tracking.

    pcf::IndiProperty m_indiP_hwpSetPos; ///< The requested (target) and measured (current) HWP angle.

    pcf::IndiProperty m_indiP_hwpPosName; ///< The name of the measured HWP position.

    pcf::IndiProperty m_indiP_hwpTrackingOffset; ///< The tracking offset applied in the last stage command.

    pcf::IndiProperty m_indiP_hwpActualPos; ///< The measured HWP angle including the tracking offset.

    pcf::IndiProperty m_indiP_teldata; ///< Subscription to the TCS 'teldata' property.

    pcf::IndiProperty m_indiP_stageTarget; ///< Template for sending 'position.target' to the stage.

    pcf::IndiProperty m_indiP_stagePos; ///< Subscription to the stage 'position' property.

    pcf::IndiProperty m_indiP_stageFsm; ///< Subscription to the stage 'fsm' property.

  public:
    /// Handle requests to enable or disable tracking.
    INDI_NEWCALLBACK_DECL( hwpTracker, m_indiP_tracking );

    /// Handle requests for a new HWP angle.
    INDI_NEWCALLBACK_DECL( hwpTracker, m_indiP_hwpSetPos );

    /// Record the telescope altitude and parallactic angle.
    INDI_SETCALLBACK_DECL( hwpTracker, m_indiP_teldata );

    /// Record the measured stage position.
    INDI_SETCALLBACK_DECL( hwpTracker, m_indiP_stagePos );

    /// Record the stage state, sending any queued move when it becomes READY.
    INDI_SETCALLBACK_DECL( hwpTracker, m_indiP_stageFsm );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */

    /// Check whether telemetry needs to be recorded.
    int checkRecordTimes();

    /// Record telemetry on the telemeter's schedule.
    int recordTelem( const telem_poltrack * /**< [in] tag for the telemetry type */ );

    /// Record the HWP tracking telemetry if it changed.
    int recordPolTrack( bool force = false /**< [in] record even if nothing changed */ );

    ///@}
};

hwpTracker::hwpTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void hwpTracker::setupConfig()
{
    config.add( "hwp.zero",
                "",
                "hwp.zero",
                argType::Required,
                "hwp",
                "zero",
                false,
                "float",
                "The stage position at HWP angle 0.  Default is 360." );

    config.add( "hwp.sign",
                "",
                "hwp.sign",
                argType::Required,
                "hwp",
                "sign",
                false,
                "int",
                "The HWP rotation sign, +1 or -1. Default is -1." );

    config.add( "hwp.devName",
                "",
                "hwp.devName",
                argType::Required,
                "hwp",
                "devName",
                false,
                "string",
                "The device name of the HWP stage.  Default is 'stagepolrot'" );

    config.add( "tcs.devName",
                "",
                "tcs.devName",
                argType::Required,
                "tcs",
                "devName",
                false,
                "string",
                "The device name of the TCS Interface providing 'teldata.zd' and `teldata.pa`.  Default is 'tcsi'" );

    config.add( "tracking.updateInterval",
                "",
                "tracking.updateInterval",
                argType::Required,
                "tracking",
                "updateInterval",
                false,
                "float",
                "The interval at which to update positions, in seconds.  Default is 10 sec." );

    config.add( "tracking.sign",
                "",
                "tracking.sign",
                argType::Required,
                "tracking",
                "sign",
                false,
                "int",
                "The sign to apply to the HWP tracking offset. 1 for Ns-W, -1 for Ns-E.  Default is -1." );

    config.add( "tracking.pupilOffset",
                "",
                "tracking.pupilOffset",
                argType::Required,
                "tracking",
                "pupilOffset",
                false,
                "float",
                "The HWP tracking pupil offset. Must match twice the kTracker offset. Default is 0." );

    TELEMETER_SETUP_CONFIG( config );
}

int hwpTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_zero, "hwp.zero" );
    _config( m_sign, "hwp.sign" );
    _config( m_devName, "hwp.devName" );
    _config( m_tcsDevName, "tcs.devName" );
    _config( m_updateInterval, "tracking.updateInterval" );
    _config( m_trackOffsetSign, "tracking.sign" );
    _config( m_pupilOffset, "tracking.pupilOffset" );

    if( m_sign != 1 && m_sign != -1 )
    {
        return log<text_log, -1>( "hwp.sign must be +1 or -1", logPrio::LOG_CRITICAL );
    }

    if( m_trackOffsetSign != 1 && m_trackOffsetSign != -1 )
    {
        return log<text_log, -1>( "tracking.sign must be +1 or -1", logPrio::LOG_CRITICAL );
    }

    if( !std::isfinite( m_updateInterval ) || m_updateInterval <= 0 )
    {
        return log<text_log, -1>( "tracking.updateInterval must be finite and positive", logPrio::LOG_CRITICAL );
    }

    if( !std::isfinite( m_zero ) || !std::isfinite( m_pupilOffset ) )
    {
        return log<text_log, -1>( "hwp.zero and tracking.pupilOffset must be finite", logPrio::LOG_CRITICAL );
    }

    if( m_devName == "" || m_tcsDevName == "" )
    {
        return log<text_log, -1>( "hwp.devName and tcs.devName must not be empty", logPrio::LOG_CRITICAL );
    }

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void hwpTracker::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = 1;
    }
}

int hwpTracker::appStartup()
{
    createStandardIndiToggleSw( m_indiP_tracking, "tracking" );
    registerIndiPropertyNew( m_indiP_tracking, INDI_NEWCALLBACK( m_indiP_tracking ) );

    createStandardIndiNumber<float>(
        m_indiP_hwpSetPos, "hwp_position", -360.0, 360.0, 1e-3, "%.03f", "HWP Set Position", "HWP Status" );
    registerIndiPropertyNew( m_indiP_hwpSetPos, INDI_NEWCALLBACK( m_indiP_hwpSetPos ) );

    REG_INDI_NEWPROP_NOCB( m_indiP_hwpTrackingOffset, "hwp_tracking_offset", pcf::IndiProperty::Number );
    m_indiP_hwpTrackingOffset.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpTrackingOffset["value"].set( 0 );

    REG_INDI_NEWPROP_NOCB( m_indiP_hwpPosName, "hwp_position_name", pcf::IndiProperty::Text );
    m_indiP_hwpPosName.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpPosName["value"].set( "" );

    REG_INDI_NEWPROP_NOCB( m_indiP_hwpActualPos, "hwp_position_actual", pcf::IndiProperty::Number );
    m_indiP_hwpActualPos.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpActualPos["value"].set( 0 );

    m_indiP_stageTarget = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_stageTarget.setDevice( m_devName );
    m_indiP_stageTarget.setName( "position" );
    m_indiP_stageTarget.add( pcf::IndiElement( "target" ) );

    REG_INDI_SETPROP( m_indiP_teldata, m_tcsDevName, "teldata" );

    REG_INDI_SETPROP( m_indiP_stagePos, m_devName, "position" );

    REG_INDI_SETPROP( m_indiP_stageFsm, m_devName, "fsm" );

    TELEMETER_APP_STARTUP;

    state( stateCodes::NOTCONNECTED );

    return 0;
}

int hwpTracker::appLogic()
{
    double now = mx::sys::get_curr_time();

    bool move = false;

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( m_stageState == stateCodes::READY )
        {
            if( m_tracking && now - m_lastUpdate > m_updateInterval )
            {
                m_lastUpdate = now;
                move         = true;
            }

            if( m_movePending )
            {
                move = true;
            }
        }
    } // mutex scope

    if( move )
    {
        requestMove();
    }

    state( trackerState() );

    TELEMETER_APP_LOGIC;

    return 0;
}

int hwpTracker::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

float hwpTracker::trackingOffset( float altitude, float parang, int trackSign, float pupilOffset )
{
    // While on Nasmyth East, the sign is negative
    // While on Nasmyth West, the sign is positive
    return trackSign * 0.5 * parang + altitude + pupilOffset;
}

float hwpTracker::stageTarget( float hwpPos, float zero, int sign )
{
    float target = mx::math::angleMod<mx::math::degreesT<float>>( zero + sign * hwpPos );

    // Guard against fmod rounding a tiny negative angle up to exactly 360.
    if( target >= 360 )
    {
        target = 0;
    }

    return target;
}

float hwpTracker::nearestAngle( float angle, float ref )
{
    float diff = angle - ref;

    return ref + diff - 360 * std::round( diff / 360 );
}

std::string hwpTracker::positionName( float hwpPos )
{
    static const std::vector<std::pair<std::string, float>> positions = {
        { "Qplus", 0 }, { "Qminus", 45 }, { "Uplus", 22.5 }, { "Uminus", 67.5 } };

    for( auto &pos : positions )
    {
        float diff = hwpPos - pos.second;

        diff -= 90 * std::round( diff / 90 );

        if( std::fabs( diff ) < posTolerance )
        {
            return pos.first;
        }
    }

    return "Unknown";
}

int hwpTracker::requestMove()
{
    float hwpPos = 0;
    float offset = 0;

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( m_stageState != stateCodes::READY )
        {
            m_movePending = true;
            return 0;
        }

        m_movePending = false;

        if( m_tracking )
        {
            m_hwpTrackingOffset = trackingOffset( m_altitude, m_parang, m_trackOffsetSign, m_pupilOffset );
        }
        else
        {
            m_hwpTrackingOffset = 0;
        }

        offset = m_hwpTrackingOffset;
        hwpPos = m_hwpSetPos + offset;
    } // mutex scope

    float target = stageTarget( hwpPos, m_zero, m_sign );

    updateIfChanged<float>( m_indiP_hwpTrackingOffset, "value", offset );

    log<text_log>( "HWP set to " + std::to_string( hwpPos ) + ", stage target " + std::to_string( target ) );

    return sendStageTarget( target );
}

int hwpTracker::sendStageTarget( float target )
{
    if( sendNewProperty( m_indiP_stageTarget, "target", target ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to send HWP stage target" } );
    }

    return 0;
}

stateCodes::stateCodeT hwpTracker::trackerState()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( !m_haveStageState || !m_haveTelData )
    {
        return stateCodes::NOTCONNECTED;
    }

    if( m_stageState == stateCodes::ERROR || m_stageState == stateCodes::FAILURE )
    {
        return stateCodes::ERROR;
    }

    if( m_stageState == stateCodes::OPERATING || m_stageState == stateCodes::HOMING ||
        ( m_stageState == stateCodes::READY && m_tracking ) )
    {
        return stateCodes::OPERATING;
    }

    if( m_stageState == stateCodes::READY )
    {
        return stateCodes::READY;
    }

    return stateCodes::NOTCONNECTED;
}

INDI_NEWCALLBACK_DEFN( hwpTracker, m_indiP_hwpSetPos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_hwpSetPos, ipRecv );

    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    float target = ipRecv["target"].get<float>();

    if( !std::isfinite( target ) )
    {
        return log<software_error, 0>( { __FILE__, __LINE__, "ignoring non-finite HWP position request" } );
    }

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );
        m_hwpSetPos = target;
    } // mutex scope

    updateIfChanged<float>( m_indiP_hwpSetPos, "target", target );

    requestMove();

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tracking, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        bool haveTelData = false;

        { // mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );

            haveTelData = m_haveTelData;

            if( haveTelData )
            {
                // m_parang holds the raw parallactic angle while not tracking, so unwrapping starts from it.
                m_tracking   = true;
                m_lastUpdate = mx::sys::get_curr_time();
            }
        } // mutex scope

        if( !haveTelData )
        {
            updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

            return log<text_log, 0>( "cannot start HWP rotation tracking: no valid teldata from " + m_tcsDevName,
                                     logPrio::LOG_WARNING );
        }

        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::On, INDI_IDLE );

        log<text_log>( "started HWP rotation tracking" );
    }
    else
    {
        { // mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );
            m_tracking = false;
        } // mutex scope

        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        log<text_log>( "stopped HWP rotation tracking" );
    }

    requestMove();

    recordPolTrack();

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpTracker, m_indiP_teldata )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_teldata, ipRecv );

    if( !ipRecv.find( "zd" ) || !ipRecv.find( "pa" ) )
    {
        return 0;
    }

    float zd = ipRecv["zd"].get<float>();
    float pa = ipRecv["pa"].get<float>();

    bool valid = std::isfinite( zd ) && std::isfinite( pa ) && zd >= 0 && zd <= 90;

    bool changed = false;

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        changed        = ( valid != m_telDataValid );
        m_telDataValid = valid;

        if( valid )
        {
            m_altitude = 90 - zd;

            // Unwrap while tracking so the 0.5*pa term does not jump by 180 degrees when pa crosses 0/360.
            if( m_tracking )
            {
                m_parang = nearestAngle( pa, m_parang );
            }
            else
            {
                m_parang = pa;
            }

            m_haveTelData = true;
        }
    } // mutex scope

    if( changed && !valid )
    {
        log<text_log>( "ignoring invalid teldata: zd=" + std::to_string( zd ) + " pa=" + std::to_string( pa ),
                       logPrio::LOG_WARNING );
    }
    else if( changed )
    {
        log<text_log>( "teldata valid again" );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpTracker, m_indiP_stagePos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stagePos, ipRecv );

    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    float stagePos = ipRecv["current"].get<float>();

    if( !std::isfinite( stagePos ) )
    {
        return 0;
    }

    float       actualPos = 0;
    float       curPos    = 0;
    std::string posName;

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        // Report angles in the user's frame, i.e. nearest to the commanded angle, since the target was wrapped.
        actualPos = nearestAngle( m_sign * ( stagePos - m_zero ), m_hwpSetPos + m_hwpTrackingOffset );

        if( std::fabs( actualPos ) < 1e-2 )
        {
            actualPos = 0;
        }

        // round to two decimal points
        curPos = std::round( ( actualPos - m_hwpTrackingOffset ) * 100 ) / 100;

        posName = positionName( curPos );

        m_hwpActualPos = actualPos;
        m_hwpCurPos    = curPos;
        m_hwpPosName   = posName;
    } // mutex scope

    updateIfChanged<float>( m_indiP_hwpActualPos, "value", actualPos );

    updateIfChanged<float>( m_indiP_hwpSetPos, "current", curPos );

    updateIfChanged<std::string>( m_indiP_hwpPosName, "value", posName );

    recordPolTrack();

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpTracker, m_indiP_stageFsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stageFsm, ipRecv );

    if( !ipRecv.find( "state" ) )
    {
        return 0;
    }

    std::string stateStr = ipRecv["state"].get<std::string>();

    stateCodes::stateCodeT stageState = stateCodes::str2Code( stateStr );

    if( stateCodes::codeText( stageState ) != stateStr )
    {
        log<text_log>( "unrecognized state '" + stateStr + "' from " + m_devName, logPrio::LOG_WARNING );

        stageState = stateCodes::NOTCONNECTED;
    }

    bool move = false;

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        m_stageState     = stageState;
        m_haveStageState = true;

        move = ( stageState == stateCodes::READY && m_movePending );
    } // mutex scope

    if( move )
    {
        requestMove();
    }

    return 0;
}

int hwpTracker::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_poltrack() );
}

int hwpTracker::recordTelem( const telem_poltrack * )
{
    return recordPolTrack( true );
}

int hwpTracker::recordPolTrack( bool force )
{
    float       curPos    = 0;
    float       actualPos = 0;
    std::string posName;
    bool        tracking = false;

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( !force && m_hwpCurPos == m_telemCurPos && m_hwpActualPos == m_telemActualPos &&
            m_hwpPosName == m_telemPosName && m_tracking == m_telemTracking )
        {
            return 0;
        }

        curPos    = m_hwpCurPos;
        actualPos = m_hwpActualPos;
        posName   = m_hwpPosName;
        tracking  = m_tracking;

        m_telemCurPos    = curPos;
        m_telemActualPos = actualPos;
        m_telemPosName   = posName;
        m_telemTracking  = tracking;
    } // mutex scope

    telem<telem_poltrack>( { curPos, actualPos, posName, tracking } );

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // hwpTracker_hpp
