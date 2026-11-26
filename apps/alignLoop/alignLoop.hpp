/** \file alignLoop.hpp
 * \brief The MagAO-X alignment loop application header file.
 *
 * \ingroup alignLoop_files
 */

#ifndef alignLoop_hpp
#define alignLoop_hpp

#include <cmath>
#include <sstream>

#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup alignLoop
 * \brief The MagAO-X application for alignment-loop control.
 *
 * \ingroup apps
 */

/** \defgroup alignLoop_files
 * \ingroup alignLoop
 */

namespace MagAOX
{
namespace app
{

/// Drive two controller axes from a shared-memory measurement stream.
/**
 * The loop subtracts `offset0` and `offset1` from the two measurement axes before
 * publishing residuals and converting them to stage corrections. Each INDI offset
 * is a live setpoint in measurement units and starts at zero.
 * Binary telemetry records the loop state and its processed measurements.
 *
 * \ingroup alignLoop
 */
class alignLoop : public MagAOXApp<true>, public dev::shmimMonitor<alignLoop>, public dev::telemeter<alignLoop>
{
    /// Give the test harness access to internal loop state.
    friend class alignLoop_test;

    /// Allow the stream monitor to dispatch frames to this app.
    friend class dev::shmimMonitor<alignLoop>;

    /// Allow periodic telemetry to request a state record.
    friend class dev::telemeter<alignLoop>;

  public:
    /// Stream monitor for the alignment measurements.
    typedef dev::shmimMonitor<alignLoop> shmimMonitorT;

    /// Binary telemetry writer for alignment records.
    typedef dev::telemeter<alignLoop> telemeterT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    /// Device name for each controlled axis.
    std::vector<std::string> m_ctrlDevices;

    /// INDI property used to read and command each axis.
    std::vector<std::string> m_ctrlProperties;

    /// INDI readback element for each axis.
    std::vector<std::string> m_ctrlCurrents;

    /// INDI command element for each axis.
    std::vector<std::string> m_ctrlTargets;

    /// Latest controller readback for each axis, protected by m_indiMutex after startup.
    std::vector<float> m_currents;

    /// Registered INDI subscriptions to controller readback properties.
    std::vector<pcf::IndiProperty> m_indiP_ctrl;

    /// Configured file name reserved for loading an interaction matrix.
    std::string m_intMatFile;

    /// Startup gains, either one shared value or one value per axis.
    std::vector<float> m_defaultGains;

    /// Optional upstream loop device whose open state opens this loop.
    std::string m_upstreamDevice;

    /// INDI toggle property of the upstream loop.
    std::string m_upstreamProperty{ "loop_state" };

    /// Whether to close this loop when the upstream loop closes.
    bool m_upstreamFollowClosed{ false };

    ///@}

    /// Interaction matrix mapping measurement residuals to command deltas.
    mx::improc::eigenImage<float> m_intMat;

    /// Private copy of the current measurement frame, corrected in place.
    mx::improc::eigenImage<float> m_measurements;

    /// Latest first-axis residual.
    float m_delta0{ 0 };

    /// Latest second-axis residual.
    float m_delta1{ 0 };

    /// Magnitude limit used for offset validation and INDI number metadata.
    static constexpr float c_maxOffset{ 1e15f };

    /// Active first-axis measurement setpoint, protected by m_indiMutex.
    float m_offset0{ 0 };

    /// Active second-axis measurement setpoint, protected by m_indiMutex.
    float m_offset1{ 0 };

    /// Most recent mapped command deltas before gain and sign are applied.
    mx::improc::eigenImage<float> m_commands;

    /// Global loop gain, protected by m_indiMutex after startup.
    float m_ggain{ 0 };

    /// Per-axis loop gains initialized during startup.
    std::vector<float> m_gains;

    /// True when stage output is enabled, protected by m_indiMutex.
    bool m_ctrlEnabled{ false };

  public:
    /// Construct the app with the current repository version.
    alignLoop();

    /// Destroy the app after shutdown.
    ~alignLoop() noexcept
    {
    }

    /// Register the monitor and loop configuration keys.
    virtual void setupConfig();

    /// Load monitor and loop settings from the supplied configuration.
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] configuration from which to load values */ );

    /// Load the application configuration.
    virtual void loadConfig();

    /// Initialize the control state, INDI properties, telemetry writer, and input monitor.
    /**
     * \returns 0 on success or -1 on a startup error.
     */
    virtual int appStartup();

    /// Check monitor and telemetry health and keep the app in the operating state.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shut down the input stream monitor and telemetry interface.
    virtual int appShutdown();

    /// Change the loop state and acknowledge it over INDI.
    int toggleLoop( bool onoff /**< [in] true to close the loop and false to open it */ );

    // shmimMonitor interface:
    /// Allocate working memory for the monitored image stream.
    int allocate( const dev::shmimT &dummy /**< [in] input monitor dispatch tag */ );

    /// Publish the corrected measurement residuals and send any closed-loop correction.
    int processImage( void              *curr_src /**< [in] pointer to the current image buffer */,
                      const dev::shmimT &dummy /**< [in] input monitor dispatch tag */
    );

    /// Send each computed target to its configured controller property.
    /**
     * \returns A bit mask of successful local INDI sends for the first two controller axes.
     */
    uint8_t sendCommands( const std::vector<float> &commands /**< [in] command values to transmit */ );

    // INDI

    /// Published offset-corrected loop residuals.
    pcf::IndiProperty m_indiP_deltas;

    /// First-axis offset with current and target elements.
    pcf::IndiProperty m_indiP_offset0;

    /// Apply the first-axis offset from the incoming INDI request `ipRecv`.
    INDI_NEWCALLBACK_DECL( alignLoop, m_indiP_offset0 );

    /// Second-axis offset with current and target elements.
    pcf::IndiProperty m_indiP_offset1;

    /// Apply the second-axis offset from the incoming INDI request `ipRecv`.
    INDI_NEWCALLBACK_DECL( alignLoop, m_indiP_offset1 );

    /// Global loop gain with current and target elements.
    pcf::IndiProperty m_indiP_ggain;

    /// INDI toggle exposing whether the loop is closed.
    pcf::IndiProperty m_indiP_ctrlEnabled;

    /// Apply the requested global gain from the incoming INDI request `ipRecv`.
    INDI_NEWCALLBACK_DECL( alignLoop, m_indiP_ggain );

    /// Open or close the loop from the incoming INDI request `ipRecv`.
    INDI_NEWCALLBACK_DECL( alignLoop, m_indiP_ctrlEnabled );

    /// Route controller property updates into the instance callback.
    static int st_setCallBack_ctrl( void                    *app /**< [in] application instance pointer */,
                                    const pcf::IndiProperty &ipRecv /**< [in] controller property update */
    )
    {
        return static_cast<alignLoop *>( app )->setCallBack_ctrl( ipRecv );
    }

    /// Update cached controller current values from an INDI property callback.
    int setCallBack_ctrl( const pcf::IndiProperty &ipRecv /**< [in] controller property update */ );

    /// Storage reserved for a cached upstream state indicator.
    int m_upstreamState{ 0 };

    /// INDI subscription to the upstream loop toggle.
    pcf::IndiProperty m_indiP_upstream;

    /// Follow the upstream loop state reported by the subscribed INDI update `ipRecv`.
    INDI_SETCALLBACK_DECL( alignLoop, m_indiP_upstream );

  protected:
    /// Apply and acknowledge a valid offset request.
    int setOffset( float                   &offset, /**< [in/out] active setpoint for the requested axis */
                   pcf::IndiProperty       &prop,   /**< [in/out] local offset property */
                   const pcf::IndiProperty &ipRecv /**< [in] received offset property */ );

    /// Refresh the archived state when no frame or state change has been recorded recently.
    int checkRecordTimes();

    /// Record the current state while holding m_indiMutex.
    int recordTelem( const telem_alignloop *tag /**< [in] unused telemetry type selector */ );

    /// Queue a state or frame record while holding m_indiMutex, or during startup.
    int recordAlignment( const AlignLoopSample *sample  = nullptr, /**< [in] processed frame, if present */
                         const AlignLoopAxes   *targets = nullptr, /**< [in] attempted controller targets, if present */
                         uint8_t                sentMask = 0 /**< [in] mask of successful local sends */ );
};

alignLoop::alignLoop() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{

    return;
}

void alignLoop::setupConfig()
{
    shmimMonitorT::setupConfig( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "ctrl.devices",
                "",
                "ctrl.devices",
                argType::Required,
                "ctrl",
                "devices",
                false,
                "string",
                "Device names of the controller(s) (one per element)." );
    config.add( "ctrl.properties",
                "",
                "ctrl.properties",
                argType::Required,
                "ctrl",
                "properties",
                false,
                "string",
                "Properties of the ctrl devices to which to give the commands. One per element" );
    config.add( "ctrl.currents",
                "",
                "ctrl.currents",
                argType::Required,
                "ctrl",
                "currents",
                false,
                "vector<string>",
                "current elements of the properties on which base the commands." );
    config.add( "ctrl.targets",
                "",
                "ctrl.targets",
                argType::Required,
                "ctrl",
                "targets",
                false,
                "vector<string>",
                "target elements of the properties to which to send the commands." );

    config.add(
        "loop.gain", "", "loop.gain", argType::Required, "loop", "gain", false, "float", "default global loop gain." );
    config.add( "loop.intMat",
                "",
                "loop.intMat",
                argType::Required,
                "loop",
                "intMat",
                false,
                "string",
                "file name of the interaction matrix." );
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

int alignLoop::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::loadConfig( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_ctrlDevices, "ctrl.devices" );
    _config( m_ctrlProperties, "ctrl.properties" );
    _config( m_ctrlCurrents, "ctrl.currents" );
    _config( m_ctrlTargets, "ctrl.targets" );
    _config( m_ggain, "loop.gain" );
    _config( m_intMatFile, "loop.intMat" );
    _config( m_defaultGains, "loop.gains" );
    _config( m_upstreamDevice, "loop.upstream" );
    _config( m_upstreamProperty, "loop.upstreamProperty" );
    _config( m_upstreamFollowClosed, "loop.upstreamFollowClosed" );
    return 0;
}

void alignLoop::loadConfig()
{
    loadConfigImpl( config );
}

int alignLoop::appStartup()
{
    if( m_ctrlTargets.size() != m_ctrlDevices.size() )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "ctrl.Targets and ctrl.devices are not the same size" } );
    }

    if( m_ctrlTargets.size() != m_ctrlProperties.size() )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "ctrl.Targets and ctrl.properties are not the same size" } );
    }

    if( m_ctrlTargets.size() != m_ctrlCurrents.size() )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "ctrl.Currents and ctrl.properties are not the same size" } );
    }

    if( m_ctrlTargets.size() != m_defaultGains.size() )
    {
        if( m_defaultGains.size() == 1 )
        {
            float g = m_defaultGains[0];
            m_defaultGains.resize( m_ctrlTargets.size(), g );
            log<text_log>( "Setting loop.gains gains to be same size as ctrl.Targets", logPrio::LOG_NOTICE );
        }
        else
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "ctrl.Targets and loop.gains are not the same size" } );
        }
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_deltas, "deltas", "Deltas", "Deltas" );
    m_indiP_deltas.add( pcf::IndiElement( "delta0" ) );
    m_indiP_deltas["delta0"] = 0;
    m_indiP_deltas.add( pcf::IndiElement( "delta1" ) );
    m_indiP_deltas["delta1"] = 0;

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_offset0, "offset0", -c_maxOffset, c_maxOffset, 0, "%g", "Offset 0", "Loop" );
    m_indiP_offset0["current"] = m_offset0;
    m_indiP_offset0["target"]  = m_offset0;

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_offset1, "offset1", -c_maxOffset, c_maxOffset, 0, "%g", "Offset 1", "Loop" );
    m_indiP_offset1["current"] = m_offset1;
    m_indiP_offset1["target"]  = m_offset1;

    m_gains.resize( m_defaultGains.size() );
    for( size_t n = 0; n < m_defaultGains.size(); ++n )
    {
        m_gains[n] = m_defaultGains[n];
    }

    createStandardIndiNumber<unsigned>( m_indiP_ggain, "loop_gain", 0, 1, 0, "%0.2f" );
    m_indiP_ggain["current"] = m_ggain;
    m_indiP_ggain["target"]  = m_ggain;
    if( registerIndiPropertyNew( m_indiP_ggain, INDI_NEWCALLBACK( m_indiP_ggain ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_ctrlEnabled, "loop_state" );
    if( registerIndiPropertyNew( m_indiP_ctrlEnabled, INDI_NEWCALLBACK( m_indiP_ctrlEnabled ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_currents.resize( m_ctrlDevices.size(), -1e15 );

    m_indiP_ctrl.resize( m_ctrlDevices.size() );

    for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
    {
        registerIndiPropertySet( m_indiP_ctrl[n], m_ctrlDevices[n], m_ctrlProperties[n], &st_setCallBack_ctrl );
    }

    m_commands.resize( m_ctrlTargets.size(), m_ctrlTargets.size() );
    m_commands.setZero();

    m_intMat.resize( m_ctrlTargets.size(), m_ctrlTargets.size() );
    m_intMat.setZero();
    m_intMat( 1, 0 ) = 0.926;
    m_intMat( 1, 1 ) = -0.370;
    m_intMat( 0, 0 ) = 0.185;
    m_intMat( 0, 1 ) = 0.926;

    /*
       std::string intMatPath = m_calibDir + "/" + m_intMatFile;

       mx::fits::fitsFile<float> ff;
       try
       {
          ff.read(m_intMat, intMatPath);
       }
       catch(...)
       {
          return log<software_error, -1>({__FILE__, __LINE__, "error reading loop.intMatFile.  Does it exist?"});
       }

       if(m_intMat.rows() != m_ctrlTargets.size())
       {
          return log<software_error, -1>({__FILE__, __LINE__, "interaction matrix wrong size: rows do not match sensor
       Targets"});
       }

       if(m_intMat.cols() != m_correctorTargets.size())
       {
          return log<software_error, -1>({__FILE__, __LINE__, "interaction matrix wrong size: cols do not match
       corrector Targets"});
       }
    */

    // Get the loop state for managing offloading
    if( m_upstreamDevice != "" )
    {
        REG_INDI_SETPROP( m_indiP_upstream, m_upstreamDevice, m_upstreamProperty );
    }

    TELEMETER_APP_STARTUP;
    recordAlignment();

    if( shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    return 0;
}

int alignLoop::appLogic()
{
    if( shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    state( stateCodes::OPERATING );
    TELEMETER_APP_LOGIC;

    return 0;
}

int alignLoop::appShutdown()
{
    shmimMonitorT::appShutdown();
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int alignLoop::toggleLoop( bool onoff )
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( !m_ctrlEnabled && onoff ) // not enabled so change
    {
        m_ctrlEnabled = true;
        log<loop_closed>();
        recordAlignment();
        updateSwitchIfChanged( m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::On, INDI_OK );
        return 0;
    }

    if( m_ctrlEnabled && !onoff )
    {
        m_ctrlEnabled = false;
        log<loop_open>();
        recordAlignment();
        updateSwitchIfChanged( m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        return 0;
    }

    return 0;
}

inline int alignLoop::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    std::cerr << shmimMonitorT::m_width << shmimMonitorT::m_height << "\n";
    m_measurements.resize( shmimMonitorT::m_width, shmimMonitorT::m_height );

    return 0;
}

inline int alignLoop::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    const uint64_t frame    = shmimMonitorT::currCnt0();
    const timespec acquired = shmimMonitorT::currAtime();

    for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
    {
        m_measurements.data()[nn] = ( static_cast<float *>( curr_src ) )[nn];
    }

    // Keep the accepted offsets and stage-send decision consistent for this frame.
    std::lock_guard<std::mutex> guard( m_indiMutex );

    const AlignLoopAxes measurement{ m_measurements( 0, 0 ), m_measurements( 1, 0 ) };
    m_measurements( 0, 0 ) -= m_offset0;
    m_measurements( 1, 0 ) -= m_offset1;
    m_delta0 = m_measurements( 0, 0 );
    m_delta1 = m_measurements( 1, 0 );

    const AlignLoopSample sample{
        frame, acquired.tv_sec, static_cast<int32_t>( acquired.tv_nsec ), measurement, { m_delta0, m_delta1 } };

    updateIfChanged(
        m_indiP_deltas, std::vector<std::string>{ "delta0", "delta1" }, std::vector<float>{ m_delta0, m_delta1 } );

    std::cout << "measurements: ";
    for( int cc = 0; cc < m_measurements.rows(); ++cc )
    {
        std::cout << m_measurements( cc, 0 ) << " ";
    }
    std::cout << "\n";

    if( m_currents[0] < -1e14 )
    {
        recordAlignment( &sample );
        return 0;
    }

    m_commands.matrix() = m_intMat.matrix() * m_measurements.matrix();

    std::cout << "delta commands:    ";
    for( int cc = 0; cc < m_measurements.rows(); ++cc )
    {
        std::cout << -m_commands( cc, 0 ) << " ";
    }
    std::cout << "\n";

    std::vector<float> commands;
    commands.resize( m_measurements.rows() );

    std::cout << "commands:    ";
    for( int cc = 0; cc < m_measurements.rows(); ++cc )
    {
        commands[cc] = m_currents[cc] - m_ggain * m_gains[cc] * m_commands( cc, 0 );
        std::cout << commands[cc] << " ";
    }
    std::cout << "\n";

    if( m_ctrlEnabled && ( m_delta0 != 0 || m_delta1 != 0 ) )
    {
        const uint8_t       sentMask = sendCommands( commands );
        const AlignLoopAxes targets{ commands[0], commands[1] };
        recordAlignment( &sample, &targets, sentMask );
        return 0;
    }

    recordAlignment( &sample );
    return 0;
}

inline uint8_t alignLoop::sendCommands( const std::vector<float> &commands )
{
    uint8_t sentMask = 0;

    for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );

        ip.setDevice( m_ctrlDevices[n] );
        ip.setName( m_ctrlProperties[n] );
        ip.add( pcf::IndiElement( m_ctrlTargets[n] ) );
        ip[m_ctrlTargets[n]] = commands[n];

        if( sendNewProperty( ip ) == 0 && n < 2 )
        {
            sentMask |= static_cast<uint8_t>( 1u << n );
        }
    }

    return sentMask;
}

inline int alignLoop::checkRecordTimes()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    return telemeterT::checkRecordTimes( telem_alignloop() );
}

inline int alignLoop::recordTelem( const telem_alignloop * )
{
    return recordAlignment();
}

inline int alignLoop::recordAlignment( const AlignLoopSample *sample, const AlignLoopAxes *targets, uint8_t sentMask )
{
    return telem<telem_alignloop>(
        { m_ctrlEnabled, m_ggain, { m_gains[0], m_gains[1] }, { m_offset0, m_offset1 }, sample, targets, sentMask } );
}

inline int alignLoop::setOffset( float &offset, pcf::IndiProperty &prop, const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( prop, ipRecv );

    const char *element = ipRecv.find( "target" ) ? "target" : "current";

    if( !ipRecv.find( element ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "missing offset value" } );
    }

    std::istringstream value( ipRecv[element].get() );
    float              target;

    if( !( value >> target ) || !( value >> std::ws ).eof() || !mx::math::isFinite( target ) ||
        std::abs( target ) > c_maxOffset )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid offset value" } );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    const bool changed = offset != target;
    offset             = target;

    if( changed )
    {
        recordAlignment();
    }

    updateIfChanged( prop, std::vector<std::string>{ "current", "target" }, std::vector<float>{ target, target } );

    return 0;
}

INDI_NEWCALLBACK_DEFN( alignLoop, m_indiP_offset0 )( const pcf::IndiProperty &ipRecv )
{
    return setOffset( m_offset0, m_indiP_offset0, ipRecv );
}

INDI_NEWCALLBACK_DEFN( alignLoop, m_indiP_offset1 )( const pcf::IndiProperty &ipRecv )
{
    return setOffset( m_offset1, m_indiP_offset1, ipRecv );
}

INDI_NEWCALLBACK_DEFN( alignLoop, m_indiP_ggain )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_ggain.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_ggain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    const bool changed = m_ggain != target;
    m_ggain            = target;

    if( changed )
    {
        recordAlignment();
    }

    updateIfChanged( m_indiP_ggain, "current", m_ggain );
    updateIfChanged( m_indiP_ggain, "target", m_ggain );

    log<text_log>( "set global gain to " + std::to_string( m_ggain ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( alignLoop, m_indiP_ctrlEnabled )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_ctrlEnabled.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

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

inline int alignLoop::setCallBack_ctrl( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( size_t n = 0; n < m_ctrlDevices.size(); ++n )
    {
        if( ipRecv.getDevice() == m_ctrlDevices[n] )
        {
            if( ipRecv.getName() == m_ctrlProperties[n] && ipRecv.find( m_ctrlCurrents[n] ) )
            {
                m_currents[n] = ipRecv[m_ctrlCurrents[n]].get<float>();
            }
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( alignLoop, m_indiP_upstream )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_upstream.getName() )
    {
        return log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On && m_upstreamFollowClosed )
    {
        std::cerr << "upstream on\n";
        return toggleLoop( true );
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        std::cerr << "upstream off\n";
        return toggleLoop( false );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // alignLoop_hpp
