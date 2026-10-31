/** \file closedLoopIndi.hpp
 * \brief The MagAO-X INDI Closed Loop header file
 *
 * \ingroup closedLoopIndi_files
 */

#ifndef closedLoopIndi_hpp
#define closedLoopIndi_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <mx/math/floatUtils.hpp>

/** \defgroup closedLoopIndi
 * \brief The MagAO-X application to do closed-loop control using INDI properties
 *
 * <a href="../handbook/operating/software/apps/closedLoopIndi.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup closedLoopIndi_files
 * \ingroup closedLoopIndi
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X application to do closed-loop control using INDI properties
/**
 * Guarded input is opt-in through `input.guarded`. Each measurement contains the configured two residuals and
 * counter plus `valid`, `session`, and `timestamp`. The session is a monotonically increasing producer epoch
 * (for example Unix milliseconds), the counter increases within that epoch, and timestamp is Unix seconds
 * at measurement completion. Epoch and counter are exactly represented nonnegative integers no larger than
 * 2^53-1, with a positive epoch. Producer/controller clocks must agree; future timestamps are rejected.
 *
 * A fresh first sample establishes a session but cannot command. Manual arming requires a fresh known session
 * and finite controller currents; only a newer sample completed after arming can command. Invalidity, stale
 * input, or a new session opens the loop. Upstream automatic closing is disabled in guarded mode. The
 * `processed` acknowledgment identifies each processed sample and whether targets were transmitted; it is
 * not proof of physical motion. Consumers must verify the target currents and settling independently.
 *
 * \ingroup closedLoopIndi
 */
class closedLoopIndi : public MagAOXApp<true>
{
    // Give the test harness access.
    friend class closedLoopIndi_test;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string              m_inputDevice;   ///< The device with the input disturbances and frame counter.
    std::string              m_inputProperty; ///< The property with the input disturbances and frame counter.
    std::vector<std::string> m_inputElements{
        "x", "y" }; ///< The elements with the input disturbances.  Must be two, defaults are "x" and "y".
    std::string m_inputCounterElement{
        "counter" }; ///< The element with the frame counter, a monotonically increasing integer. Default is "counter".

    bool m_inputGuarded{ false }; ///< Require the session, timestamp, and validity protocol when true.

    double m_inputMaxAge{ 5 }; ///< Maximum age in seconds of a guarded input at reception or arming.

    std::vector<std::string>
        m_ctrlDevices; ///< Device names of the controller(s). If only one, it's used for both properties.  Max two.
    std::vector<std::string>
        m_ctrlProperties; ///< Properties of the ctrl device(s) to which to give the commands. Must specify two.
    std::vector<std::string> m_ctrlCurrents{ "current",
                                             "current" }; ///< current elements of the properties on which to base the
                                                          ///< commands. Must specify 0 or 2. Default is 'current'.
    std::vector<std::string> m_ctrlTargets{ "target",
                                            "target" }; ///< target elements of the properties to which to send the
                                                        ///< commands. Must specify 0 or 2. Default is 'target'.

    mx::improc::eigenImage<float> m_references; ///< The reference values of the disturbances

    bool m_operatingOK{ false }; ///< If true then it's ok for the ctrl device to be OPERATING when a command is sent

    std::unordered_map<std::string, std::string> m_fsmStates; ///< The FSM states of the control devices.

    std::vector<float> m_currents; ///< The current commands

    mx::improc::eigenImage<float> m_intMat; ///< The interaction matrix.  Default is [1 0][0 1].

    std::vector<float> m_defaultGains; ///< The default gains, per-axis

    std::string m_upstreamDevice; ///< The upstream device to monitor to automatically open this loop if it's loop opens
    std::string m_upstreamProperty{ "loop_state" }; ///< The name of the toggle switch to monitor
    bool        m_upstreamFollowClosed{ false };    ///< If true, this loop also closes when the upstream loop closes.

    ///@}

    int64_t                       m_counter{ -1 }; ///< The latest accepted loop counter.
    mx::improc::eigenImage<float> m_measurements;  ///< The latest value of the measurements
    float                         m_delta0{ 0 };   ///< Latest first-axis residual.
    float                         m_delta1{ 0 };   ///< Latest second-axis residual.

    mx::improc::eigenImage<float> m_commands; ///< The latest commands

    float m_ggain{ 0 }; ///< The global gain

    std::vector<float> m_gains; ///< The axis gains

    bool m_loopClosed{ false }; ///< Whether or not the loop is closed

    std::recursive_mutex m_loopMutex; ///< Serializes loop callbacks, watchdog checks, and command publication.

    double m_inputSession{ 0 }; ///< Greatest accepted producer epoch; older sessions cannot rearm this instance.

    double m_inputTimestamp{ 0 }; ///< Timestamp of the latest accepted guarded input, in Unix seconds.

    double m_armedAfter{ 0 }; ///< Manual-arm time; an earlier input cannot command after arming.

    bool m_inputReady{ false }; ///< A complete fresh input has established the current producer session.

    std::vector<bool> m_currentValid; ///< Tracks receipt of finite current values for both controlled axes.

  public:
    /// Default c'tor.
    closedLoopIndi();

    /// D'tor, declared and defined for noexcept.
    ~closedLoopIndi() noexcept = default;

    /// Set up the app configuration interface.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load the application configuration.
    virtual void loadConfig();

    /// Register loop controls, input subscriptions, and optional guarded acknowledgments.
    virtual int appStartup();

    /// Implementation of the FSM for closedLoopIndi.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    virtual int appShutdown();

    /// Change the loop state
    int toggleLoop( bool onoff /**< [in] true to close the loop and false to open it */ );

    /// Update the loop with a new command
    int updateLoop();

    /// Send commands to the control devices
    int sendCommands( std::vector<float> &commands /**< [in] command values to transmit */ );

    /// Parse a complete finite number using mxlib's fast-math-safe classifier.
    static bool readNumber( double            &value, /**< [out] Parsed finite value. */
                            const std::string &text /**< [in] Numeric text without trailing characters. */ );

    /// Read a finite numeric element without relying on fast-math finite assumptions.
    static bool readNumber( double                  &value,    /**< [out] Parsed finite value. */
                            const pcf::IndiProperty &property, /**< [in] Complete input property. */
                            const std::string       &element /**< [in] Element to parse. */ );

    /// Return Unix time in seconds for input-age and arm-boundary checks.
    static double inputTime();

    /// Check all controller states and, in guarded mode, received current values.
    bool controllersReady() const;

    /// Process a complete guarded input; invalid or new-session inputs open the loop.
    int guardedInput( const pcf::IndiProperty &property /**< [in] Atomic measurement and validity tuple. */ );

    /// Publish the result for one identified input after processing its commands.
    /** Status: -2 send failure, -1 rejected, 0 open, 1 sent, 2 unready, 3 zero, 4 new session, 5 before arm.
     */
    void
    publishProcessed( double                    session, /**< [in] Producer epoch identifying this input. */
                      double                    counter, /**< [in] Counter identifying this input. */
                      int                       status,  /**< [in] Processing outcome. */
                      const std::vector<float> &targets /**< [in] Commanded targets, or currents when not applied. */ );

    // INDI

    pcf::IndiProperty m_indiP_processed; ///< Guarded-input acknowledgment with identity, targets, and outcome.

    pcf::IndiProperty m_indiP_deltas; ///< Published loop residual values.

    pcf::IndiProperty m_indiP_reference0; ///< INDI property exposing the first reference value.
    /// Handle updates to m_indiP_reference0.
    INDI_NEWCALLBACK_DECL( closedLoopIndi, m_indiP_reference0 );

    pcf::IndiProperty m_indiP_reference1; ///< INDI property exposing the second reference value.
    /// Handle updates to m_indiP_reference1.
    INDI_NEWCALLBACK_DECL( closedLoopIndi, m_indiP_reference1 );

    pcf::IndiProperty m_indiP_inputs; ///< INDI property monitored for incoming disturbances.
    /// Handle updates to m_indiP_inputs.
    INDI_SETCALLBACK_DECL( closedLoopIndi, m_indiP_inputs );

    pcf::IndiProperty m_indiP_ggain; ///< INDI property exposing the global loop gain.
    /// Handle updates to m_indiP_ggain.
    INDI_NEWCALLBACK_DECL( closedLoopIndi, m_indiP_ggain );

    pcf::IndiProperty m_indiP_ctrlEnabled; ///< INDI property exposing the loop open/closed state.
    /// Handle updates to m_indiP_ctrlEnabled.
    INDI_NEWCALLBACK_DECL( closedLoopIndi, m_indiP_ctrlEnabled );

    pcf::IndiProperty m_indiP_counterReset; ///< INDI property used to request a counter reset.
    /// Handle updates to m_indiP_counterReset.
    INDI_NEWCALLBACK_DECL( closedLoopIndi, m_indiP_counterReset );

    pcf::IndiProperty m_indiP_ctrl0_fsm; ///< The INDI property for fsm state of axis 0
    /// Handle updates to m_indiP_ctrl0_fsm.
    INDI_SETCALLBACK_DECL( closedLoopIndi, m_indiP_ctrl0_fsm );

    pcf::IndiProperty m_indiP_ctrl0; ///< The INDI property used for control of axis 0
    /// Handle updates to m_indiP_ctrl0.
    INDI_SETCALLBACK_DECL( closedLoopIndi, m_indiP_ctrl0 );

    pcf::IndiProperty m_indiP_ctrl1_fsm; ///< The INDI property for fsm state of axis 1
    /// Handle updates to m_indiP_ctrl1_fsm.
    INDI_SETCALLBACK_DECL( closedLoopIndi, m_indiP_ctrl1_fsm );

    pcf::IndiProperty m_indiP_ctrl1; ///< The INDI property used for control of axis 1
    /// Handle updates to m_indiP_ctrl1.
    INDI_SETCALLBACK_DECL( closedLoopIndi, m_indiP_ctrl1 );

    pcf::IndiProperty m_indiP_upstream; ///< Property used to monitor the upstream loop state.
    /// Handle updates to m_indiP_upstream.
    INDI_SETCALLBACK_DECL( closedLoopIndi, m_indiP_upstream );
};

closedLoopIndi::closedLoopIndi() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void closedLoopIndi::setupConfig()
{
    config.add( "input.device",
                "",
                "input.device",
                argType::Required,
                "input",
                "device",
                false,
                "string",
                "The device with the input disturbances and frame counter." );
    config.add( "input.property",
                "",
                "input.property",
                argType::Required,
                "input",
                "property",
                false,
                "string",
                "The property with the input disturbances and counter." );
    config.add( "input.elements",
                "",
                "input.elements",
                argType::Required,
                "input",
                "elements",
                false,
                "vector<string>",
                " The elements with the input disturbances.  Must be two, defaults are 'x' and 'y'." );
    config.add( "input.counterElement",
                "",
                "input.counterElement",
                argType::Required,
                "input",
                "counterElement",
                false,
                "string",
                "The element with the frame counter, a monotonically increasing integer. Default is 'counter'." );

    config.add( "input.guarded",
                "",
                "input.guarded",
                argType::Required,
                "input",
                "guarded",
                false,
                "bool",
                "Require atomic valid/session/timestamp input metadata and manual rearming. Default false." );
    config.add( "input.maxAge",
                "",
                "input.maxAge",
                argType::Required,
                "input",
                "maxAge",
                false,
                "double",
                "Maximum guarded-input age in seconds. Must be positive; default 5." );

    config.add( "input.references",
                "",
                "input.references",
                argType::Required,
                "input",
                "references",
                false,
                "vector<float>",
                "The reference values for the input disturbances." );

    config.add( "ctrl.devices",
                "",
                "ctrl.devices",
                argType::Required,
                "ctrl",
                "devices",
                false,
                "string",
                "Device names of the controller(s). If only one, it's used for both properties.  Max two." );
    config.add( "ctrl.properties",
                "",
                "ctrl.properties",
                argType::Required,
                "ctrl",
                "properties",
                false,
                "string",
                "Properties of the ctrl device(s) to which to give the commands. Must specify two." );
    config.add(
        "ctrl.currents",
        "",
        "ctrl.currents",
        argType::Required,
        "ctrl",
        "currents",
        false,
        "vector<string>",
        "current elements of the properties on which to base the commands. Must specify 0 or 2. Default is 'current'" );
    config.add(
        "ctrl.targets",
        "",
        "ctrl.targets",
        argType::Required,
        "ctrl",
        "targets",
        false,
        "vector<string>",
        "target elements of the properties to which to send the commands. Must specify 0 or 2. Default is 'target'." );
    config.add( "ctrl.operatingOK",
                "",
                "ctrl.operatingOK",
                argType::Required,
                "ctrl",
                "operatingOK",
                false,
                "bool",
                "If true then it's ok for the ctrl device to be OPERATING when a command is sent. Default is false." );

    config.add( "loop.intMat00",
                "",
                "loop.intMat00",
                argType::Required,
                "loop",
                "intMat00",
                false,
                "float",
                "element (0,0) of the interaction matrix. Default is 1." );
    config.add( "loop.intMat01",
                "",
                "loop.intMat01",
                argType::Required,
                "loop",
                "intMat01",
                false,
                "float",
                "element (0,1) of the interaction matrix. Default is 0." );
    config.add( "loop.intMat10",
                "",
                "loop.intMat10",
                argType::Required,
                "loop",
                "intMat10",
                false,
                "float",
                "element (1,0) of the interaction matrix. Default is 0." );
    config.add( "loop.intMat11",
                "",
                "loop.intMat11",
                argType::Required,
                "loop",
                "intMat11",
                false,
                "float",
                "element (1,1) of the interaction matrix. Default is 1." );

    config.add(
        "loop.gain", "", "loop.gain", argType::Required, "loop", "gain", false, "float", "default global loop gain." );
    config.add( "loop.gains",
                "",
                "loop.gains",
                argType::Required,
                "loop",
                "gains",
                false,
                "vector<float>",
                "default loop gains.  If single number, it is applied to all axes." );
    config.add( "loop.upstream",
                "",
                "loop.upstream",
                argType::Required,
                "loop",
                "upstream",
                false,
                "string",
                "Upstream loop device name.  This loop will open, and optionally close, with the upstream loop.  "
                "Default none." );
    config.add( "loop.upstreamProperty",
                "",
                "loop.upstreamProperty",
                argType::Required,
                "loop",
                "upstreamProperty",
                false,
                "string",
                "Property of upstream loop device to follow.  Must be a toggle.  Default is loop_state." );
    config.add( "loop.upstreamFollowClosed",
                "",
                "loop.upstreamFollowClosed",
                argType::Required,
                "loop",
                "upstreamFollowClosed",
                false,
                "bool",
                "If true, this loop also closes when the upstream loop closes. Default is false." );
}

int closedLoopIndi::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_inputDevice, "input.device" );
    _config( m_inputProperty, "input.property" );
    _config( m_inputElements, "input.elements" );
    _config( m_inputCounterElement, "input.counterElement" );
    _config( m_inputGuarded, "input.guarded" );
    _config( m_inputMaxAge, "input.maxAge" );
    if( m_inputGuarded && ( !mx::math::isFinite( m_inputMaxAge ) || m_inputMaxAge <= 0 ) )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "input.maxAge must be finite and positive" } );
    }

    if( m_inputDevice == "" )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "no input device specified" } );
    }

    if( m_inputProperty == "" )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "no input property specified" } );
    }

    if( m_inputElements.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "must specify only two input.elements" } );
    }

    std::vector<float> refs( { 0, 0 } );
    _config( refs, "input.references" );
    if( refs.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "input.references must have 2 elements" } );
    }

    m_references.resize( 2, 1 );
    m_references( 0, 0 ) = refs[0];
    m_references( 1, 0 ) = refs[1];

    _config( m_ctrlDevices, "ctrl.devices" );
    _config( m_ctrlProperties, "ctrl.properties" );
    _config( m_ctrlCurrents, "ctrl.currents" );
    _config( m_ctrlTargets, "ctrl.targets" );
    _config( m_operatingOK, "ctrl.operatingOK" );

    if( m_ctrlDevices.size() == 1 )
    {
        m_ctrlDevices.push_back( m_ctrlDevices[0] );
    }
    else if( m_ctrlDevices.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "must specify two ctrl.devices" } );
    }

    if( m_ctrlProperties.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "must specify two ctrl.properties" } );
    }

    if( m_ctrlTargets.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "must specify two ctrl.targets" } );
    }

    if( m_ctrlCurrents.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "must specify two ctrl.currents" } );
    }

    if( m_ctrlTargets.size() != 2 )
    {
        m_shutdown = 1;
        return log<software_error, -1>( { __FILE__, __LINE__, "must specify two ctrl.targets" } );
    }

    float im00 = 1;
    float im01 = 0;
    float im10 = 0;
    float im11 = 1;

    if( m_inputGuarded )
    {
        float *entries[] = { &im00, &im01, &im10, &im11 };
        size_t n         = 0;
        for( const auto &key : { "loop.intMat00", "loop.intMat01", "loop.intMat10", "loop.intMat11" } )
        {
            std::string text;
            double      value = 0;
            _config( text, key );
            if( !_config.isSet( key ) || !readNumber( value, text ) ||
                std::abs( value ) > std::numeric_limits<float>::max() )
            {
                m_shutdown = 1;
                return log<software_error, -1>(
                    { __FILE__, __LINE__, "guarded input requires an explicit finite calibrated 2-by-2 matrix" } );
            }
            *entries[n++] = value;
        }
        if( static_cast<double>( im00 ) * im11 - static_cast<double>( im01 ) * im10 == 0 )
        {
            m_shutdown = 1;
            return log<software_error, -1>( { __FILE__, __LINE__, "guarded control matrix must be nonsingular" } );
        }
    }
    else
    {
        _config( im00, "loop.intMat00" );
        _config( im01, "loop.intMat01" );
        _config( im10, "loop.intMat10" );
        _config( im11, "loop.intMat11" );
    }

    m_intMat.resize( 2, 2 );
    m_intMat( 0, 0 ) = im00;
    m_intMat( 0, 1 ) = im01;
    m_intMat( 1, 0 ) = im10;
    m_intMat( 1, 1 ) = im11;

    _config( m_ggain, "loop.gain" );

    _config( m_defaultGains, "loop.gains" );
    _config( m_upstreamDevice, "loop.upstream" );
    _config( m_upstreamProperty, "loop.upstreamProperty" );
    _config( m_upstreamFollowClosed, "loop.upstreamFollowClosed" );

    return 0;
}

void closedLoopIndi::loadConfig()
{
    loadConfigImpl( config );
}

int closedLoopIndi::appStartup()
{
    REG_INDI_SETPROP( m_indiP_inputs, m_inputDevice, m_inputProperty );

    if( m_inputGuarded )
    {
        CREATE_REG_INDI_RO_NUMBER( m_indiP_processed, "processed", "Processed input", "loop" );
        for( const auto &name : { "session", "counter", "target0", "target1", "applied", "status" } )
        {
            m_indiP_processed.add( pcf::IndiElement( name ) );
            m_indiP_processed[name] = 0;
        }
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_deltas, "deltas", "Deltas", "Deltas" );
    m_indiP_deltas.add( pcf::IndiElement( "delta0" ) );
    m_indiP_deltas["delta0"] = 0;
    m_indiP_deltas.add( pcf::IndiElement( "delta1" ) );
    m_indiP_deltas["delta1"] = 0;

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_reference0, "reference0", -1e15, 1e15, 1, "%g", "reference0", "references" );
    m_indiP_reference0["current"] = m_references( 0, 0 );
    m_indiP_reference0["target"]  = m_references( 0, 0 );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_reference1, "reference1", -1e15, 1e15, 1, "%g", "reference1", "references" );
    m_indiP_reference1["current"] = m_references( 1, 0 );
    m_indiP_reference1["target"]  = m_references( 1, 0 );

    if( m_ctrlTargets.size() != m_defaultGains.size() )
    {
        if( m_defaultGains.size() == 1 )
        {
            m_defaultGains.push_back( m_defaultGains[0] );
            log<text_log>( "Setting loop.gains gains to be same size as ctrl.Targets", logPrio::LOG_NOTICE );
        }
        else
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "ctrl.Targets and loop.gains are not the same size" } );
        }
    }

    m_gains.resize( m_defaultGains.size() );
    for( size_t n = 0; n < m_defaultGains.size(); ++n )
        m_gains[n] = m_defaultGains[n];

    CREATE_REG_INDI_NEW_NUMBERU( m_indiP_ggain, "loop_gain", 0, 1, 0, "%0.2f", "gain", "loop" );
    m_indiP_ggain["current"] = m_ggain;
    m_indiP_ggain["target"]  = m_ggain;

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_ctrlEnabled, "loop_state" );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_counterReset, "counter_reset" );

    m_measurements.resize( 2, 1 );
    m_measurements.setZero();

    m_currents.resize( m_ctrlDevices.size(), -1e15 );
    m_currentValid.assign( m_ctrlDevices.size(), false );

    REG_INDI_SETPROP( m_indiP_ctrl0_fsm, m_ctrlDevices[0], "fsm" );

    REG_INDI_SETPROP( m_indiP_ctrl0, m_ctrlDevices[0], m_ctrlProperties[0] );

    if( m_ctrlDevices[1] != m_ctrlDevices[0] )
    {
        REG_INDI_SETPROP( m_indiP_ctrl1_fsm, m_ctrlDevices[1], "fsm" );
    }

    REG_INDI_SETPROP( m_indiP_ctrl1, m_ctrlDevices[1], m_ctrlProperties[1] );

    m_commands.resize( 2, 2 );
    m_commands.setZero();

    // Get the loop state for managing offloading
    if( m_upstreamDevice != "" )
    {
        REG_INDI_SETPROP( m_indiP_upstream, m_upstreamDevice, m_upstreamProperty );
    }

    return 0;
}

int closedLoopIndi::appLogic()
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    if( m_inputGuarded && m_inputReady &&
        ( inputTime() < m_inputTimestamp || inputTime() - m_inputTimestamp > m_inputMaxAge ) )
    {
        m_inputReady = false;
        toggleLoop( false );
    }
    if( m_loopClosed )
    {
        state( stateCodes::OPERATING );
    }
    else
    {
        state( stateCodes::READY );
    }

    return 0;
}

int closedLoopIndi::appShutdown()
{
    return 0;
}

int closedLoopIndi::toggleLoop( bool onoff )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    if( m_inputGuarded && onoff && !m_loopClosed )
    {
        double now = inputTime();
        if( !m_inputReady || now < m_inputTimestamp || now - m_inputTimestamp > m_inputMaxAge || !controllersReady() )
        {
            return 0;
        }
        m_armedAfter = now;
    }
    if( !m_loopClosed && onoff ) // not enabled so change
    {
        m_loopClosed = true;
        log<loop_closed>();
        updateSwitchIfChanged( m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::On, INDI_OK );
        return 0;
    }

    if( m_loopClosed && !onoff )
    {
        m_loopClosed = false;
        log<loop_open>();
        updateSwitchIfChanged( m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        return 0;
    }

    return 0;
}

inline bool closedLoopIndi::controllersReady() const
{
    if( m_ctrlDevices.empty() )
        return false;
    for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
    {
        auto state = m_fsmStates.find( m_ctrlDevices[n] );
        if( state == m_fsmStates.end() ||
            ( state->second != "READY" && !( m_operatingOK && state->second == "OPERATING" ) ) )
            return false;
        if( m_inputGuarded && ( n >= m_currentValid.size() || !m_currentValid[n] ) )
            return false;
    }
    return true;
}

inline int closedLoopIndi::updateLoop()
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    if( !controllersReady() )
    {
        if( m_inputGuarded )
        {
            toggleLoop( false );
            publishProcessed( m_inputSession, m_counter, 2, m_currents );
        }
        return 0;
    }

    m_delta0            = m_measurements( 0, 0 ) - m_references( 0, 0 );
    m_delta1            = m_measurements( 1, 0 ) - m_references( 1, 0 );
    m_commands.matrix() = m_intMat.matrix() * ( m_measurements - m_references ).matrix();

    std::vector<float> commands( m_measurements.rows() );
    bool               changed = false;
    for( int cc = 0; cc < m_measurements.rows(); ++cc )
    {
        commands[cc] = m_currents[cc] - m_ggain * m_gains[cc] * m_commands( cc, 0 );
        changed      = changed || commands[cc] != m_currents[cc];
        if( m_inputGuarded && !mx::math::isFinite( commands[cc] ) )
        {
            m_inputReady = false;
            toggleLoop( false );
            publishProcessed( m_inputSession, m_counter, -1, m_currents );
            return -1;
        }
    }

    int rv     = 0;
    int status = m_loopClosed ? ( changed ? 1 : 3 ) : 0;
    if( m_loopClosed && ( !m_inputGuarded || changed ) )
        rv = sendCommands( commands );

    if( m_inputGuarded )
    {
        if( rv < 0 )
        {
            status       = -2;
            m_inputReady = false;
            toggleLoop( false );
        }
        publishProcessed( m_inputSession, m_counter, status, status == 1 || status == -2 ? commands : m_currents );
    }
    updateIfChanged( m_indiP_deltas,
                     std::vector<std::string>( { "delta0", "delta1" } ),
                     std::vector<float>( { m_delta0, m_delta1 } ) );
    return rv;
}

inline bool closedLoopIndi::readNumber( double &value, const std::string &text )
{
    char *end = nullptr;
    errno     = 0;
    value     = std::strtod( text.c_str(), &end );
    return end != text.c_str() && *end == '\0' && errno != ERANGE && mx::math::isFinite( value );
}

inline bool closedLoopIndi::readNumber( double &value, const pcf::IndiProperty &property, const std::string &element )
{
    return property.find( element ) && readNumber( value, property[element].getValue() );
}

inline double closedLoopIndi::inputTime()
{
    timespec now{};
    clock_gettime( CLOCK_REALTIME, &now );
    return now.tv_sec + now.tv_nsec * 1e-9;
}

inline void
closedLoopIndi::publishProcessed( double session, double counter, int status, const std::vector<float> &targets )
{
    m_indiP_processed["session"] = static_cast<uint64_t>( session );
    m_indiP_processed["counter"] = static_cast<uint64_t>( counter );
    m_indiP_processed["target0"] = targets.size() > 0 && mx::math::isFinite( targets[0] ) ? targets[0] : 0;
    m_indiP_processed["target1"] = targets.size() > 1 && mx::math::isFinite( targets[1] ) ? targets[1] : 0;
    m_indiP_processed["applied"] = status == 1 ? 1 : 0;
    m_indiP_processed["status"]  = status;
    m_indiP_processed.setState( status < 0 ? INDI_ALERT : INDI_OK );
    if( m_indiDriver )
        m_indiDriver->sendSetProperty( m_indiP_processed );
}

inline int closedLoopIndi::guardedInput( const pcf::IndiProperty &property )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    double                                session = 0, counter = 0, timestamp = 0, valid = 0, x = 0, y = 0;
    bool identity = readNumber( session, property, "session" ) && session > 0 && session <= 9007199254740991.0 &&
                    std::floor( session ) == session && readNumber( counter, property, m_inputCounterElement ) &&
                    counter >= 0 && counter <= 9007199254740991.0 && std::floor( counter ) == counter;
    double now = inputTime();
    bool   complete =
        identity && readNumber( valid, property, "valid" ) && valid == 1 &&
        readNumber( timestamp, property, "timestamp" ) && timestamp <= now && now - timestamp <= m_inputMaxAge &&
        readNumber( x, property, m_inputElements[0] ) && readNumber( y, property, m_inputElements[1] ) &&
        std::abs( x ) <= std::numeric_limits<float>::max() && std::abs( y ) <= std::numeric_limits<float>::max();
    if( !complete || session < m_inputSession )
    {
        m_inputReady = false;
        toggleLoop( false );
        if( identity )
            publishProcessed( session, counter, -1, m_currents );
        return 0;
    }
    if( session == m_inputSession && counter <= m_counter )
    {
        if( counter == m_counter )
        {
            // Repeat the original result if an acknowledgment was lost; never resend commands.
            if( m_indiDriver && m_indiP_processed["session"].get<double>() == session &&
                m_indiP_processed["counter"].get<double>() == counter )
                m_indiDriver->sendSetProperty( m_indiP_processed );
            return 0;
        }
        m_inputReady = false;
        toggleLoop( false );
        publishProcessed( session, counter, -1, m_currents );
        return 0;
    }
    if( session == m_inputSession && timestamp <= m_inputTimestamp )
    {
        m_inputReady = false;
        toggleLoop( false );
        publishProcessed( session, counter, -1, m_currents );
        return 0;
    }
    bool newSession        = session != m_inputSession;
    m_inputSession         = session;
    m_counter              = static_cast<int64_t>( counter );
    m_inputTimestamp       = timestamp;
    m_inputReady           = true;
    m_measurements( 0, 0 ) = x;
    m_measurements( 1, 0 ) = y;
    if( newSession )
    {
        toggleLoop( false );
        publishProcessed( session, counter, 4, m_currents );
        return 0;
    }
    if( m_loopClosed && timestamp <= m_armedAfter )
    {
        publishProcessed( session, counter, 5, m_currents );
        return 0;
    }
    return updateLoop();
}

inline int closedLoopIndi::sendCommands( std::vector<float> &commands )
{
    for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );

        ip.setDevice( m_ctrlDevices[n] );
        ip.setName( m_ctrlProperties[n] );
        ip.add( pcf::IndiElement( m_ctrlTargets[n] ) );
        ip[m_ctrlTargets[n]] = commands[n];

        int rv = sendNewProperty( ip );
        if( m_inputGuarded && rv < 0 )
            return rv;
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( closedLoopIndi, m_indiP_reference0 )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_reference0 );

    float target;

    if( indiTargetUpdate( m_indiP_reference0, target, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    m_references( 0, 0 ) = target;

    updateIfChanged( m_indiP_reference0,
                     std::vector<std::string>( { "current", "target" } ),
                     std::vector<float>( { m_references( 0, 0 ), m_references( 0, 0 ) } ) );

    log<text_log>( "set reference0 to " + std::to_string( m_references( 0, 0 ) ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( closedLoopIndi, m_indiP_reference1 )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_reference1 );

    float target;

    if( indiTargetUpdate( m_indiP_reference1, target, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    m_references( 1, 0 ) = target;

    updateIfChanged( m_indiP_reference1,
                     std::vector<std::string>( { "current", "target" } ),
                     std::vector<float>( { m_references( 1, 0 ), m_references( 1, 0 ) } ) );

    log<text_log>( "set reference1 to " + std::to_string( m_references( 1, 0 ) ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_SETCALLBACK_DEFN( closedLoopIndi, m_indiP_inputs )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_inputs )

    if( m_inputGuarded )
        return guardedInput( ipRecv );

    if( !ipRecv.find( m_inputElements[0] ) )
        return -1;
    if( !ipRecv.find( m_inputElements[1] ) )
        return -1;
    if( !ipRecv.find( m_inputCounterElement ) )
        return -1;

    int counter = ipRecv[m_inputCounterElement].get<int>();

    if( counter != m_counter )
    {
        m_counter              = counter;
        m_measurements( 0, 0 ) = ipRecv[m_inputElements[0]].get<float>();
        m_measurements( 1, 0 ) = ipRecv[m_inputElements[1]].get<float>();

        return updateLoop();
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( closedLoopIndi, m_indiP_ggain )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_ggain );

    float target;

    if( indiTargetUpdate( m_indiP_ggain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_ggain = target;

    updateIfChanged( m_indiP_ggain, "current", m_ggain );
    updateIfChanged( m_indiP_ggain, "target", m_ggain );

    log<text_log>( "set global gain to " + std::to_string( m_ggain ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( closedLoopIndi, m_indiP_ctrlEnabled )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_ctrlEnabled );

    // switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        return toggleLoop( true );
    }

    // switch is toggle to off
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        return toggleLoop( false );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( closedLoopIndi, m_indiP_counterReset )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_counterReset );

    // switch is toggled to on
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        if( m_inputGuarded )
        {
            m_inputReady = false;
            toggleLoop( false );
        }
        else
            m_counter = -1;
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( closedLoopIndi, m_indiP_ctrl0_fsm )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_ctrl0_fsm );

    if( ipRecv.find( "state" ) )
    {
        m_fsmStates[ipRecv.getDevice()] = ipRecv["state"].get();
        if( m_inputGuarded && ipRecv["state"].get() != "READY" &&
            !( m_operatingOK && ipRecv["state"].get() == "OPERATING" ) )
        {
            for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
                if( m_ctrlDevices[n] == ipRecv.getDevice() )
                    m_currentValid[n] = false;
            toggleLoop( false );
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( closedLoopIndi, m_indiP_ctrl0 )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_ctrl0 );

    if( ipRecv.find( m_ctrlCurrents[0] ) )
    {
        if( m_inputGuarded )
        {
            double value      = 0;
            m_currentValid[0] = readNumber( value, ipRecv, m_ctrlCurrents[0] ) &&
                                std::abs( value ) <= std::numeric_limits<float>::max();
            if( !m_currentValid[0] )
            {
                toggleLoop( false );
                return 0;
            }
            m_currents[0] = value;
        }
        else
            m_currents[0] = ipRecv[m_ctrlCurrents[0]].get<float>();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( closedLoopIndi, m_indiP_ctrl1_fsm )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_ctrl1_fsm );

    if( ipRecv.find( "state" ) )
    {
        m_fsmStates[ipRecv.getDevice()] = ipRecv["state"].get();
        if( m_inputGuarded && ipRecv["state"].get() != "READY" &&
            !( m_operatingOK && ipRecv["state"].get() == "OPERATING" ) )
        {
            for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
                if( m_ctrlDevices[n] == ipRecv.getDevice() )
                    m_currentValid[n] = false;
            toggleLoop( false );
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( closedLoopIndi, m_indiP_ctrl1 )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_ctrl1 );

    if( ipRecv.find( m_ctrlCurrents[1] ) )
    {
        if( m_inputGuarded )
        {
            double value      = 0;
            m_currentValid[1] = readNumber( value, ipRecv, m_ctrlCurrents[1] ) &&
                                std::abs( value ) <= std::numeric_limits<float>::max();
            if( !m_currentValid[1] )
            {
                toggleLoop( false );
                return 0;
            }
            m_currents[1] = value;
        }
        else
            m_currents[1] = ipRecv[m_ctrlCurrents[1]].get<float>();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( closedLoopIndi, m_indiP_upstream )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::recursive_mutex> lock( m_loopMutex );
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_upstream );

    if( !ipRecv.find( "toggle" ) )
        return 0;

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On && m_upstreamFollowClosed && !m_inputGuarded )
    {
        std::cerr << "upstream on\n";
        return toggleLoop( true );
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        std::cerr << "upstream off\n";
        if( m_inputGuarded )
            m_inputReady = false;
        return toggleLoop( false );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // closedLoopIndi_hpp
