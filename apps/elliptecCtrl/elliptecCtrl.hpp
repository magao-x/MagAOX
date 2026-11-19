/** \file elliptecCtrl.hpp
 * \brief The MagAO-X Elliptec stage controller header.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup elliptecCtrl_files
 */

#ifndef elliptecCtrl_hpp
#define elliptecCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

/** \defgroup elliptecCtrl
 * \brief The MagAO-X application to control one Thorlabs Elliptec rotation stage.
 *
 * \ingroup apps
 */

/** \defgroup elliptecCtrl_files
 * \ingroup elliptecCtrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X controller for one Elliptec rotation stage on a bus owned by `elliptecLowLevel`.
/** Several Elliptec devices can share one serial bus, so this app does no serial I/O. It follows its stage's
 * elements (`stage.stageName`) in the `elliptecLowLevel` properties `curr_state`, `curr_pos` and `last_error`,
 * and commands the stage by sending `tgt_pos`, `req_home` and `req_halt`. Power management also belongs to the
 * low-level app, which reports POWEROFF for every stage while the bus is off.
 *
 * This app owns the standard `stdMotionStage` interface (presets, preset names, home, stop), the absolute and
 * relative angle controls, the post-home offset, the power-on home, and telemetry.
 *
 * The stage is modeled as a single 360 degree circle: positions are wrapped into [0, 360).
 *
 * \ingroup elliptecCtrl
 */
class elliptecCtrl : public MagAOXApp<>, public dev::stdMotionStage<elliptecCtrl>, public dev::telemeter<elliptecCtrl>
{
    friend class dev::stdMotionStage<elliptecCtrl>;

    friend class dev::telemeter<elliptecCtrl>;

  public:
    /// The telemeter base type.
    typedef dev::telemeter<elliptecCtrl> telemeterT;

    /// Construct the controller with power management disabled (the low-level app owns power).
    elliptecCtrl();

    /// Destroy the controller.
    ~elliptecCtrl() noexcept
    {
    }

    /// Set up the application configuration.
    virtual void setupConfig() override;

    /// Implementation of configuration loading.
    /**
     * \returns 0 on success
     * \returns -1 on an invalid configuration
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration from which to load */ );

    /// Load the application configuration.
    virtual void loadConfig() override;

    /// Register INDI properties and subscribe to the low-level app.
    virtual int appStartup() override;

    /// Update INDI and telemetry, and request the power-on home when due.
    virtual int appLogic() override;

    /// Shut the application down.
    virtual int appShutdown() override;

    /// Request a stop of the current motion. Called without `m_indiMutex`; acquires it.
    int stop();

    /// Begin a homing sequence. Caller holds `m_indiMutex`.
    int startHoming();

    /// Resolve the active 0-based preset index for the current angle, or -1 if not at a preset.
    float presetNumber();

    /// Move to a target supplied by `stdMotionStage`. Caller holds `m_indiMutex`.
    int moveTo( float target /**< [in] preset number (m_movingState == 0) or preset position in degrees */ );

    /// Check whether staged telemetry records need to be forced.
    int checkRecordTimes();

    /// Record stage-state telemetry on a telemeter request.
    int recordTelem( const telem_stage * /**< [in] telemetry tag used for overload resolution */ );

    /// Record position telemetry on a telemeter request.
    int recordTelem( const telem_position * /**< [in] telemetry tag used for overload resolution */ );

    /// Record the standard stage-state telemetry.
    int recordStage( bool force = false /**< [in] force the record */ );

    /// Record the position telemetry.
    int recordPosition( bool force = false /**< [in] force the record */ );

    /// Handle a new position target in degrees.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_indiP_position );

    /// Follow the low-level state of this stage.
    INDI_SETCALLBACK_DECL( elliptecCtrl, m_indiP_llState );

    /// Follow the low-level position of this stage.
    INDI_SETCALLBACK_DECL( elliptecCtrl, m_indiP_llPos );

    /// Follow the low-level error text of this stage.
    INDI_SETCALLBACK_DECL( elliptecCtrl, m_indiP_llError );

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    /// INDI device name of the `elliptecLowLevel` app that owns this stage's bus.
    std::string m_lowLevelName{ "elliptecLowLevel" };

    /// Name of this stage's elements in the low-level properties; defaults to this app's name.
    std::string m_stageName;

    /// Angular tolerance in degrees for being at a preset.
    double m_presetToleranceDeg{ 0.5 };

    ///@}

    /** \name Runtime State - Data
     * @{
     */

    /// Last state string reported by the low-level app for this stage.
    std::string m_llState;

    /// Last position in degrees reported by the low-level app.
    double m_posDeg{ 0.0 };

    /// Last error text reported by the low-level app, empty when OK.
    std::string m_lastError;

    /// Whether a home has been requested and the low level has not yet reported its end.
    bool m_homing{ false };

    /// Whether the next NOTHOMED report should trigger the power-on home (when `stage.powerOnHome` is set).
    bool m_powerOnHomePending{ true };

    /// The last command sent to the low-level app, for diagnostics and tests.
    /** Unit tests define `ELLIPTECCTRL_TEST_NOSEND` so commands are recorded here but not sent. */
    pcf::IndiProperty m_lastCommand;

    /// Whether position telemetry has been recorded since startup.
    bool m_positionRecorded{ false };

    /// Position at the last position telemetry record.
    double m_lastRecordedPosDeg{ 0.0 };

    ///@}

    /** \name INDI Properties - Data
     * @{
     */

    /// Position property in degrees on [0, 360), with `current` and `target` elements.
    pcf::IndiProperty m_indiP_position;

    /// Subscription to the low-level `curr_state`.
    pcf::IndiProperty m_indiP_llState;

    /// Subscription to the low-level `curr_pos`.
    pcf::IndiProperty m_indiP_llPos;

    /// Subscription to the low-level `last_error`.
    pcf::IndiProperty m_indiP_llError;

    ///@}

    /** \name Helpers
     *
     * Require the caller to hold `m_indiMutex`.
     *
     * @{
     */

    /// Send this stage's element of a low-level number property.
    int sendNumber_( const std::string &prop, /**< [in] low-level property name */
                     double             value /**< [in] value for this stage's element */
    );

    /// Send a switch request (`On`) for this stage.
    int sendRequest_( const std::string &prop /**< [in] low-level property name */ );

    /// Command an absolute move in degrees.
    int moveToDeg_( double deg /**< [in] target angle */ );

    /// Whether a motion request may be sent now; logs a warning naming the request when not.
    bool motionAllowed_( const std::string &what /**< [in] request name for the log */ );

    /// Shortest angular distance between two angles, in [0, 180].
    static double circularDistanceDeg_( double a, /**< [in] first angle in degrees */
                                        double b  /**< [in] second angle in degrees */
    );

    /// Resolve the preset index matching an angle on the circle, or -1.
    int presetIndexForPosition_( double deg /**< [in] angle in degrees */ ) const;

    /// Check that distinct preset positions are separable by the tolerance.
    /**
     * \returns 0 if valid, -1 if two distinct presets are closer than twice the tolerance
     */
    int validatePresets_();

    /// Update `m_preset` and `m_preset_target` from the position and motion state.
    void syncPresetState_();

    /// Apply a low-level state string to this app's FSM and the homing sequence.
    void applyState_( const std::string &st /**< [in] low-level state string */ );

    /// Publish `position.current`, BUSY while moving or homing.
    void updatePosition_();

    ///@}
};

/* ========================= impl ========================= */

inline elliptecCtrl::elliptecCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_presetNotation   = "preset";
    m_powerMgtEnabled  = false;
    m_defaultPositions = false;
    m_moving           = -2;
}

/* ---------- Config ---------- */

inline void elliptecCtrl::setupConfig()
{
    config.add( "stage.lowLevelName",
                "",
                "stage.lowLevelName",
                argType::Required,
                "stage",
                "lowLevelName",
                false,
                "string",
                "INDI name of the elliptecLowLevel app that owns this stage's bus" );

    config.add( "stage.stageName",
                "",
                "stage.stageName",
                argType::Required,
                "stage",
                "stageName",
                false,
                "string",
                "Name of this stage in the low-level app; defaults to this app's name" );

    config.add( "stage.presetTolerance",
                "0.5",
                "stage.presetTolerance",
                argType::Optional,
                "stage",
                "presetTolerance",
                false,
                "double",
                "Angular tolerance (deg) for being at a preset" );

    if( dev::stdMotionStage<elliptecCtrl>::setupConfig( config ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Error from stdMotionStage::setupConfig" } );
        m_shutdown = true;
    }

    TELEMETER_SETUP_CONFIG( config );
}

inline int elliptecCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_lowLevelName, "stage.lowLevelName" );

    m_stageName = m_configName;
    _config( m_stageName, "stage.stageName" );

    _config( m_presetToleranceDeg, "stage.presetTolerance" );
    if( m_presetToleranceDeg <= 0 )
    {
        return log<text_log, -1>( "stage.presetTolerance must be positive", logPrio::LOG_CRITICAL );
    }

    if( dev::stdMotionStage<elliptecCtrl>::loadConfig( _config ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error from stdMotionStage::loadConfig" } );
    }

    if( validatePresets_() < 0 )
    {
        return -1;
    }

    if( m_homePreset >= static_cast<int>( std::min( m_presetPositions.size(), m_presetNames.size() ) ) )
    {
        return log<text_log, -1>( "stage.homePreset is not a configured preset", logPrio::LOG_CRITICAL );
    }

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

inline void elliptecCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

/* ---------- Startup/Logic ---------- */

inline int elliptecCtrl::appStartup()
{
    if( state() == stateCodes::UNINITIALIZED )
    {
        return log<text_log, -1>( "UNINITIALIZED in appStartup", logPrio::LOG_CRITICAL );
    }

    createStandardIndiNumber<float>( m_indiP_position, "position", 0.0, 360.0, 0.0, "%.3f" );
    m_indiP_position["current"].set( 0 );
    m_indiP_position["target"].set( 0 );
    if( registerIndiPropertyNew( m_indiP_position, INDI_NEWCALLBACK( m_indiP_position ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::stdMotionStage<elliptecCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    REG_INDI_SETPROP( m_indiP_llState, m_lowLevelName, std::string( "curr_state" ) );
    REG_INDI_SETPROP( m_indiP_llPos, m_lowLevelName, std::string( "curr_pos" ) );
    REG_INDI_SETPROP( m_indiP_llError, m_lowLevelName, std::string( "last_error" ) );

    TELEMETER_APP_STARTUP;

    state( stateCodes::NOTCONNECTED );
    return 0;
}

inline int elliptecCtrl::appLogic()
{
    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( state() == stateCodes::INITIALIZED )
        {
            state( stateCodes::NOTCONNECTED );
        }

        // The power-on home is requested here, once per power-on, when the low level first reports NOTHOMED.
        if( m_llState == "NOTHOMED" && m_powerOnHomePending )
        {
            m_powerOnHomePending = false;

            if( m_powerOnHome )
            {
                log<text_log>( "homing after power-on" );
                startHoming();
            }
        }

        syncPresetState_();
        if( dev::stdMotionStage<elliptecCtrl>::updateINDI() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
        }
        updatePosition_();
    }

    TELEMETER_APP_LOGIC;
    return 0;
}

inline int elliptecCtrl::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

/* ---------- stdMotionStage surface ---------- */

inline int elliptecCtrl::stop()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    indi::updateSwitchIfChanged( m_indiP_stop, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

    if( sendRequest_( "req_halt" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "could not send halt request" } );
    }

    return 0;
}

inline int elliptecCtrl::startHoming()
{
    indi::updateSwitchIfChanged( m_indiP_home, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

    if( !motionAllowed_( "home" ) )
    {
        return 0;
    }

    if( sendRequest_( "req_home" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "could not send home request" } );
    }

    clearPresetNameTracking();
    m_preset_target = 0;
    m_homing        = true;
    m_moving        = 2;
    state( stateCodes::HOMING );
    recordStage( true );
    return 0;
}

inline float elliptecCtrl::presetNumber()
{
    return static_cast<float>( presetIndexForPosition_( m_posDeg ) );
}

inline int elliptecCtrl::moveTo( float target )
{
    double deg = target;
    int    idx = -1;

    if( m_movingState == 0 )
    {
        if( m_presetPositions.empty() )
        {
            return -1;
        }

        idx = static_cast<int>( std::lround( target ) ) - 1;
        idx = std::max( 0, std::min( idx, static_cast<int>( m_presetPositions.size() ) - 1 ) );
        deg = m_presetPositions[static_cast<size_t>( idx )];
    }
    else
    {
        idx = presetIndexForPosition_( deg );
    }

    if( !motionAllowed_( "preset move" ) )
    {
        return 0;
    }

    if( idx >= 0 )
    {
        m_preset_target = static_cast<float>( idx + 1 );
    }

    return moveToDeg_( deg );
}

/* ---------- telemeter wrappers ---------- */

inline int elliptecCtrl::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_stage(), telem_position() );
}

inline int elliptecCtrl::recordTelem( const telem_stage * )
{
    return recordStage( true );
}

inline int elliptecCtrl::recordTelem( const telem_position * )
{
    return recordPosition( true );
}

inline int elliptecCtrl::recordStage( bool force )
{
    return dev::stdMotionStage<elliptecCtrl>::recordStage( force );
}

inline int elliptecCtrl::recordPosition( bool force )
{
    if( !m_positionRecorded || m_posDeg != m_lastRecordedPosDeg || force )
    {
        telem<telem_position>( static_cast<float>( m_posDeg ) );
        m_lastRecordedPosDeg = m_posDeg;
        m_positionRecorded   = true;
    }

    return 0;
}

/* ---------- INDI callbacks ---------- */

INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_indiP_position )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_position, ipRecv );

    double tgt = 0.0;
    if( indiTargetUpdate( m_indiP_position, tgt, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( !motionAllowed_( "position move" ) )
    {
        return 0;
    }

    clearPresetNameTracking();
    m_movingState   = 0;
    m_preset_target = 0;
    return moveToDeg_( tgt );
}

INDI_SETCALLBACK_DEFN( elliptecCtrl, m_indiP_llState )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_llState, ipRecv );

    if( !ipRecv.find( m_stageName ) )
    {
        return 0; // not our stage
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );
    applyState_( ipRecv[m_stageName].getValue() );
    return 0;
}

INDI_SETCALLBACK_DEFN( elliptecCtrl, m_indiP_llPos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_llPos, ipRecv );

    if( !ipRecv.find( m_stageName ) )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );
    m_posDeg = ipRecv[m_stageName].get<double>();
    syncPresetState_();
    recordPosition();
    updatePosition_();
    return 0;
}

INDI_SETCALLBACK_DEFN( elliptecCtrl, m_indiP_llError )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_llError, ipRecv );

    if( !ipRecv.find( m_stageName ) )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    std::string err = ipRecv[m_stageName].getValue(); // get<std::string>() would stop at the first space
    if( err != m_lastError )
    {
        if( !err.empty() )
        {
            log<text_log>( "stage error: " + err, logPrio::LOG_WARNING );
        }
        else
        {
            log<text_log>( "stage error cleared" );
        }

        m_lastError = err;
    }
    return 0;
}

/* ---------- Helpers ---------- */

inline int elliptecCtrl::sendNumber_( const std::string &prop, double value )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( m_lowLevelName );
    ip.setName( prop );
    ip.setPerm( pcf::IndiProperty::ReadWrite );
    ip.setState( pcf::IndiProperty::Idle );
    ip.add( pcf::IndiElement( m_stageName ) );
    ip[m_stageName].set( value );

    m_lastCommand = ip;

#ifdef ELLIPTECCTRL_TEST_NOSEND
    return 0; // unit tests inspect m_lastCommand instead of contacting an indiserver
#else
    return sendNewProperty( ip );
#endif
}

inline int elliptecCtrl::sendRequest_( const std::string &prop )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( m_lowLevelName );
    ip.setName( prop );
    ip.setPerm( pcf::IndiProperty::ReadWrite );
    ip.setState( pcf::IndiProperty::Idle );
    ip.add( pcf::IndiElement( m_stageName ) );
    ip[m_stageName].setSwitchState( pcf::IndiElement::On );

    m_lastCommand = ip;

#ifdef ELLIPTECCTRL_TEST_NOSEND
    return 0; // unit tests inspect m_lastCommand instead of contacting an indiserver
#else
    return sendNewProperty( ip );
#endif
}

inline int elliptecCtrl::moveToDeg_( double deg )
{
    deg = std::fmod( deg, 360.0 );
    if( deg < 0 )
    {
        deg += 360.0;
    }

    updateIfChanged( m_indiP_position, "target", deg, INDI_BUSY );

    if( sendNumber_( "tgt_pos", deg ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "could not send target" } );
    }

    m_moving = 1;
    state( stateCodes::OPERATING );
    recordStage( true );
    return 0;
}

inline bool elliptecCtrl::motionAllowed_( const std::string &what )
{
    if( m_llState != "READY" && m_llState != "NOTHOMED" && m_llState != "OPERATING" )
    {
        log<text_log>( what + " rejected: stage is " +
                           ( m_llState.empty() ? std::string( "not reported" ) : m_llState ),
                       logPrio::LOG_WARNING );
        return false;
    }

    if( m_homing )
    {
        log<text_log>( what + " rejected: homing in progress", logPrio::LOG_WARNING );
        return false;
    }

    return true;
}

inline double elliptecCtrl::circularDistanceDeg_( double a, double b )
{
    double d = std::abs( std::fmod( a - b, 360.0 ) );
    return std::min( d, 360.0 - d );
}

inline int elliptecCtrl::presetIndexForPosition_( double deg ) const
{
    for( size_t i = 0; i < m_presetPositions.size(); ++i )
    {
        if( circularDistanceDeg_( m_presetPositions[i], deg ) <= m_presetToleranceDeg )
        {
            return static_cast<int>( i );
        }
    }

    return -1;
}

inline int elliptecCtrl::validatePresets_()
{
    for( size_t i = 0; i < m_presetPositions.size(); ++i )
    {
        for( size_t j = i + 1; j < m_presetPositions.size(); ++j )
        {
            double d = circularDistanceDeg_( m_presetPositions[i], m_presetPositions[j] );
            if( d > 0 && d < 2 * m_presetToleranceDeg )
            {
                std::string n1 = ( i < m_presetNames.size() ) ? m_presetNames[i] : std::to_string( i );
                std::string n2 = ( j < m_presetNames.size() ) ? m_presetNames[j] : std::to_string( j );
                return log<text_log, -1>( "presets " + n1 + " and " + n2 +
                                              " are closer than twice stage.presetTolerance",
                                          logPrio::LOG_CRITICAL );
            }
        }
    }

    return 0;
}

inline void elliptecCtrl::syncPresetState_()
{
    const int idx = presetIndexForPosition_( m_posDeg );

    m_preset = ( idx < 0 ) ? 0 : static_cast<float>( idx + 1 );

    if( m_moving <= 0 )
    {
        m_preset_target = m_preset;
    }
}

inline void elliptecCtrl::applyState_( const std::string &st )
{
    const std::string prev = m_llState;
    m_llState              = st;

    if( st == "POWEROFF" )
    {
        state( stateCodes::POWEROFF );
        m_moving             = -2;
        m_homing             = false;
        m_powerOnHomePending = true;
    }
    else if( st == "POWERON" )
    {
        state( stateCodes::POWERON );
        m_moving = -2;
    }
    else if( st == "NODEVICE" )
    {
        state( stateCodes::NODEVICE );
        m_moving = -2;
        m_homing = false;
    }
    else if( st == "NOTCONNECTED" )
    {
        state( stateCodes::NOTCONNECTED );
        m_moving = -2;
        m_homing = false;
    }
    else if( st == "NOTHOMED" )
    {
        if( m_homing && prev != "NOTHOMED" )
        {
            log<text_log>( "homing did not complete", logPrio::LOG_WARNING );
        }
        m_homing = false;
        state( stateCodes::NOTHOMED );
        m_moving = -1;
    }
    else if( st == "HOMING" )
    {
        m_homing = true;
        state( stateCodes::HOMING );
        m_moving = 2;
    }
    else if( st == "OPERATING" )
    {
        state( stateCodes::OPERATING );
        m_moving = 1;
    }
    else if( st == "READY" )
    {
        const bool homed = m_homing;

        m_homing = false;
        state( stateCodes::READY );
        m_moving = 0;

        if( homed && m_homePreset >= 0 )
        {
            // The home finished: move to the configured preset (stdMotionStage presets are 1-based).
            log<text_log>( "homed; moving to preset " + m_presetNames[static_cast<size_t>( m_homePreset )] );
            m_movingState = 0;
            moveTo( m_homePreset + 1 );
        }
    }
    else
    {
        state( stateCodes::NOTCONNECTED );
        m_moving = -2;
    }

    syncPresetState_();
    recordStage();
    updatePosition_();
}

inline void elliptecCtrl::updatePosition_()
{
    updateIfChanged( m_indiP_position, "current", m_posDeg, ( m_moving > 0 ) ? INDI_BUSY : INDI_IDLE );
}

} // namespace app
} // namespace MagAOX

#endif // elliptecCtrl_hpp
