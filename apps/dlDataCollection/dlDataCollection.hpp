/** \file dlDataCollection.hpp
 * \brief Collect synchronized PWFS images and recorded tweeter commands.
 * \ingroup dlDataCollection_files
 */
#ifndef dlDataCollection_hpp
#define dlDataCollection_hpp

#include <atomic>
#include <sys/stat.h>

#ifndef DLDATACOLLECTION_TEST
    #include "../../libMagAOX/libMagAOX.hpp" // Included on the command line to trigger pch.
    #include "../../magaox_git_version.h"
#endif
#include "dlCollection.hpp"
#include "dlObservation.hpp"

namespace MagAOX
{
namespace app
{
/** \defgroup dlDataCollection Deep-learning data collection
 * \brief Validated command banks and synchronized four-pupil PWFS datasets.
 * \ingroup apps
 */
/** \defgroup dlDataCollection_files dlDataCollection Files
 * \ingroup dlDataCollection
 */

/// Instrument adapter for the CPU-only collection state machine.
/** The monitor uses READY while idle and attaches in OPERATING for a pending or active run.
 * Start is explicit; construction, stream attachment and bank preparation send no command.
 * \ingroup dlDataCollection
 */
class dlDataCollection : public MagAOXApp<true>, public dev::shmimMonitor<dlDataCollection>
{
    friend class dev::shmimMonitor<dlDataCollection>;

    /// Standard source-stream monitor.
    typedef dev::shmimMonitor<dlDataCollection> shmimMonitorT;

  protected:
    /** \name Collection Configuration - Data
     * @{
     */
    /// Parent for unique run directories; retains parameters.dataDirs.
    std::string m_output;

    /// Offline-prepared bank directory; retains parameters.ampsDir.
    std::string m_bank;

    /// Dedicated 50x50 float32 tweeter command channel.
    std::string m_channel;

    /// Required actuator count; only the 2500-element layout is supported.
    int m_modes{ 2500 };

    /// Expected samples per set, or zero to use the bank manifest.
    int m_samples{ 0 };

    /// Number of bank sets to acquire, or zero to use all sets.
    int m_sets{ 0 };

    /// Minimum fresh-frame counter advance after publication; zero still requires a new frame.
    int m_skip{ 8 };

    /// Padded pupil width and height, including two zero pixels on each edge.
    int m_pupil{ 64 };

    /// Left/right pupil offsets in source columns.
    std::array<int, 2> m_x{ 0, 60 };

    /// Top/bottom pupil offsets in source rows.
    std::array<int, 2> m_y{ 0, 60 };

    /// Intensity scaling applied only to accepted WFS images.
    float m_imageNorm{ 1 };

    /// Legacy unused option; only unity is accepted to prevent implicit command rescaling.
    float m_modalNorm{ 1 };

    /// Maximum time without a unique source frame during collection, in seconds.
    double m_frameTimeout{ 1 };
    ///@}

    /// Session owns all image and command buffers; its publisher checks stream ownership.
    dlCollection::Session m_session;

    /// Output stream handle, closed only under m_streamMutex.
    IMAGE m_commandStream{};

    /// Serialize output attachment, publication, release, and shutdown.
    std::mutex m_streamMutex;

    /// Whether the output mapping is valid.
    bool m_commandOpened{ false };

    /// Whether this app has issued a command on the initially zero channel.
    bool m_commandOwned{ false };

    /// Output counter after this app's last write; detect an intervening writer.
    uint64_t m_commandCounter{ 0 };

    /// File identity of the attached output mapping.
    ino_t m_commandInode{ 0 };

    /// Semaphore mapping size established during output attachment.
    uint16_t m_commandSemaphores{ 0 };

    /// Filename used to detect output deletion/recreation before a write.
    std::string m_commandFilename;

    /// Source and output attachment have passed preflight.
    std::atomic<bool> m_connected{ false };

    /// Source dimensions copied on attachment, guarded by m_indiMutex.
    std::array<size_t, 2> m_sourceSize{ 0, 0 };

    /// INDI callbacks only queue requests; appLogic performs run transitions.
    std::atomic<bool> m_startRequested{ false };

    /// Queued operator Stop.
    std::atomic<bool> m_stopRequested{ false };

    /// Ordinary-priority coefficient loader and writer.
    std::thread m_ioThread;

    /// Standard threadStart initialization handshake.
    bool m_ioThreadInit{ true };

    /// Published I/O thread identity.
    pid_t m_ioThreadID{ 0 };

    /// Standard thread status property.
    pcf::IndiProperty m_indiP_ioThread;

    /// Explicit Start control.
    pcf::IndiProperty m_indiP_start;

    /// Stop and preserve the valid partial prefix.
    pcf::IndiProperty m_indiP_stop;

    /// Idle-only output-root control.
    pcf::IndiProperty m_indiP_output;

    /// Idle-only bank-directory control.
    pcf::IndiProperty m_indiP_bank;

    /// Read-only phase, output directory, counters, rate, and failure.
    pcf::IndiProperty m_indiP_collection;

    /// Wall duration used to report effective saved-sample throughput.
    std::chrono::steady_clock::time_point m_started;

    /// Final observed run duration; negative while the run is active.
    double m_finishedElapsed{ 0 };

    /// Visible control-request failure, separate from a run's data-integrity failure.
    std::string m_controlError;

    /// Latest internal-source wheel readback, guarded by m_indiMutex.
    dlCollection::PresetReadback m_nd;

    /// Latest WFS flipper readback, guarded by m_indiMutex.
    dlCollection::PresetReadback m_flipwfsf;

    /// Latest camwfs frame rate in Hz, guarded by m_indiMutex.
    dlCollection::CameraReadback m_fps;

    /// Latest camwfs E.M. gain factor, guarded by m_indiMutex.
    dlCollection::CameraReadback m_emgain;

    /// Read-only subscription and refresh request for camwfs.fps.
    pcf::IndiProperty m_indiP_fps;

    /// Read-only subscription and refresh request for camwfs.emgain.
    pcf::IndiProperty m_indiP_emgain;

    /// Subscription and read-only refresh request for fwtelsim.filterName.
    pcf::IndiProperty m_indiP_nd;

    /// Subscription and read-only refresh request for flipwfsf.presetName.
    pcf::IndiProperty m_indiP_flipwfsf;

    /// Controller status subscription for the ND wheel.
    pcf::IndiProperty m_indiP_ndState;

    /// Controller status subscription for the WFS flipper.
    pcf::IndiProperty m_indiP_flipwfsfState;

    /// Display the current automatically read positions and camera settings in cursesINDI.
    pcf::IndiProperty m_indiP_observation;

    /// Wheel position frozen at Start, guarded by m_indiMutex.
    std::string m_runND;

    /// Flipper position frozen at Start, guarded by m_indiMutex.
    std::string m_runFlipwfsf;

    /// Frame rate frozen at Start, guarded by m_indiMutex.
    double m_runFPS{ 0 };

    /// E.M. gain frozen at Start, guarded by m_indiMutex.
    double m_runEMGain{ 0 };

    /// Whether readback changes invalidate the current run; guarded by m_indiMutex.
    bool m_observationActive{ false };

    /// Latch a change until the next Start; capture and publication check this flag.
    std::atomic<bool> m_observationChanged{ false };

    /// Start waits for new readbacks; text callbacks reject edits during this wait.
    std::atomic<bool> m_startPending{ false };

    /// Bound the main-thread wait for fresh observing readbacks after Start.
    std::chrono::steady_clock::time_point m_observationDeadline{};

    /// Refresh observing readbacks once per second, outside the image callback.
    std::chrono::steady_clock::time_point m_observationPoll{};

  public:
    /// Construct the inert instrument adapter.
    dlDataCollection();

    /// Destroy only after the normal main-thread shutdown has joined workers.
    ~dlDataCollection() noexcept;

    /// Register old configuration keys and the source watchdog.
    void setupConfig() override;

    /// Load and validate options without opening streams or issuing commands.
    int loadConfigImpl( mx::app::appConfigurator &cfg /**< [in] Configuration to consume. */ );

    /// Apply configuration, propagating validation failure to the application.
    void loadConfig() override;

    /// Publish INDI controls and start inert monitor and ordinary-priority file work.
    int appStartup() override;

    /// Process queued controls, source watchdog and user-visible progress.
    int appLogic() override;

    /// Stop capture, join both workers, preserve partial data and close output.
    int appShutdown() override;

    /// Queue Start from INDI.
    INDI_NEWCALLBACK_DECL( dlDataCollection, m_indiP_start );

    /// Queue Stop from INDI.
    INDI_NEWCALLBACK_DECL( dlDataCollection, m_indiP_stop );

    /// Update the output root while idle.
    INDI_NEWCALLBACK_DECL( dlDataCollection, m_indiP_output );

    /// Update the command bank while idle.
    INDI_NEWCALLBACK_DECL( dlDataCollection, m_indiP_bank );

    /// Track the internal-source wheel's reported named position.
    INDI_SETCALLBACK_DECL( dlDataCollection, m_indiP_nd );

    /// Track the WFS flipper's reported in/out position.
    INDI_SETCALLBACK_DECL( dlDataCollection, m_indiP_flipwfsf );

    /// Invalidate wheel positions when its controller is disconnected or unhealthy.
    INDI_SETCALLBACK_DECL( dlDataCollection, m_indiP_ndState );

    /// Invalidate flipper positions when its controller is disconnected or unhealthy.
    INDI_SETCALLBACK_DECL( dlDataCollection, m_indiP_flipwfsfState );

    /// Track camwfs frame rate and invalidate runs on a reported change.
    INDI_SETCALLBACK_DECL( dlDataCollection, m_indiP_fps );

    /// Track camwfs E.M. gain and invalidate runs on a reported change.
    INDI_SETCALLBACK_DECL( dlDataCollection, m_indiP_emgain );

  protected:
    /// Validate/attach streams; reconnection aborts rather than relabeling a partial run.
    int allocate( const dev::shmimT &dummy /**< [in] Source-monitor tag. */ );

    /// Forward a unique stable float32 source snapshot to the capture state machine.
    int processImage( void              *src /**< [in] Current source frame. */,
                      const dev::shmimT &dummy /**< [in] Source-monitor tag. */ );

    /// Build an immutable run request from current settings and attached dimensions.
    dlCollection::Settings settings();

    /// Publish/release the owned output channel and return the source counter afterwards.
    uint64_t publish( const float *command /**< [in] 2500 floats, or null to release. */ );

    /// Check mapping identity and write continuity; a new run may adopt a stable zero channel.
    /** Caller holds m_streamMutex. */
    void checkCommandStream( bool newRun /**< [in] Validate a zero channel and refresh its idle counter. */ = false );

    /// Accept an image only if its source snapshot and held command remain valid.
    bool snapshotStable( uint64_t counter /**< [in] Producer counter recorded before copying. */ );

    /// Start the standard file-thread handshake and its session worker.
    static void ioStart( dlDataCollection *app /**< [in] Owning application. */ );

    /// Refresh observations, start after readback, and abort changed-state runs.
    void checkObservation();

    /// Apply one idle-only text request using standard target/current elements.
    int textRequest( const pcf::IndiProperty &request /**< [in] Received property. */,
                     pcf::IndiProperty       &property /**< [in/out] Registered property. */,
                     std::string             &value /**< [in/out] Corresponding setting. */ );
};

inline dlDataCollection::dlDataCollection()
    : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED ), m_session(
                                                                  [this]( const float *command )
                                                                  {
                                                                      return publish( command );
                                                                  } )
{
}

inline dlDataCollection::~dlDataCollection() noexcept = default;

inline void dlDataCollection::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    const std::array<std::array<const char *, 3>, 15> entries{
        { { "dataDirs", "string", "Parent directory for unique acquisition runs." },
          { "ampsDir", "string", "Offline-prepared float32 command bank with manifest.conf." },
          { "channel", "string", "Dedicated initially-zero 50x50 float32 tweeter channel." },
          { "Nperset", "int", "Samples per set; zero uses the bank manifest." },
          { "Nset", "int", "Sets to acquire; zero uses all bank sets." },
          { "Nmodes", "int", "Full actuator count; must be 2500." },
          { "NumFrameSkip",
            "int",
            "Minimum source-counter advance after publication; zero still requires a fresh frame." },
          { "imageNorm", "float", "Scaling of saved WFS images." },
          { "modalNorm", "float", "Legacy unused command normalization; must remain 1." },
          { "m_pupPix", "int", "Padded pupil size including two zero pixels on each edge." },
          { "pup_offset1_x", "int", "Left pupil source-column offset." },
          { "pup_offset2_x", "int", "Right pupil source-column offset." },
          { "pup_offset1_y", "int", "Top pupil source-row offset." },
          { "pup_offset2_y", "int", "Bottom pupil source-row offset." },
          { "frameTimeout", "double", "Maximum seconds without a fresh frame during collection." } } };
    for( size_t i = 0; i < entries.size(); ++i )
    {
        const auto       &entry = entries[i];
        const std::string key   = std::string( "parameters." ) + entry[0];
        config.add( key, "", key, argType::Required, "parameters", entry[0], false, entry[1], entry[2] );
    }
}

inline int dlDataCollection::loadConfigImpl( mx::app::appConfigurator &cfg )
{
    SHMIMMONITOR_LOAD_CONFIG( cfg );
    cfg( m_output, "parameters.dataDirs" );
    cfg( m_bank, "parameters.ampsDir" );
    cfg( m_channel, "parameters.channel" );
    cfg( m_samples, "parameters.Nperset" );
    cfg( m_sets, "parameters.Nset" );
    cfg( m_modes, "parameters.Nmodes" );
    cfg( m_skip, "parameters.NumFrameSkip" );
    cfg( m_imageNorm, "parameters.imageNorm" );
    cfg( m_modalNorm, "parameters.modalNorm" );
    cfg( m_pupil, "parameters.m_pupPix" );
    cfg( m_x[0], "parameters.pup_offset1_x" );
    cfg( m_x[1], "parameters.pup_offset2_x" );
    cfg( m_y[0], "parameters.pup_offset1_y" );
    cfg( m_y[1], "parameters.pup_offset2_y" );
    cfg( m_frameTimeout, "parameters.frameTimeout" );
    if( m_modes != 2500 || m_samples < 0 || m_sets < 0 || m_skip < 0 || m_pupil <= 4 || m_x[0] < 0 || m_x[1] < 0 ||
        m_y[0] < 0 || m_y[1] < 0 || !dlCollection::finite( m_imageNorm ) || m_imageNorm <= 0 ||
        !dlCollection::finite( m_modalNorm ) || m_modalNorm != 1 || !dlCollection::finite( m_frameTimeout ) ||
        !( m_frameTimeout > 0 ) || m_frameTimeout > 3600 || m_bank.empty() || m_output.empty() ||
        m_channel.size() != 10 || m_channel.substr( 0, 8 ) != "dm01disp" || m_channel[8] < '0' || m_channel[8] > '9' ||
        m_channel[9] < '0' || m_channel[9] > '9' )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Invalid collection configuration" } );
    }
    return 0;
}

inline void dlDataCollection::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

inline dlCollection::Settings dlDataCollection::settings()
{
    dlCollection::Settings s;
    s.m_bank      = m_bank;
    s.m_output    = m_output;
    s.m_samples   = m_samples;
    s.m_sets      = m_sets;
    s.m_skip      = m_skip;
    s.m_pupil     = m_pupil;
    s.m_x         = { size_t( m_x[0] ), size_t( m_x[1] ) };
    s.m_y         = { size_t( m_y[0] ), size_t( m_y[1] ) };
    s.m_height    = m_sourceSize[0];
    s.m_width     = m_sourceSize[1];
    s.m_imageNorm = m_imageNorm;
    s.m_timeout   = m_frameTimeout;
    s.m_context   = "revision=" + std::string( MAGAOX_CURRENT_SHA1 ) +
                    "\nrepo_modified=" + std::to_string( MAGAOX_REPO_MODIFIED ) +
                    "\nsource=" + shmimMonitorT::m_shmimName + "\nchannel=" + m_channel + "\n";
    return s;
}

inline int dlDataCollection::appStartup()
{
    REG_INDI_SETPROP( m_indiP_nd, "fwtelsim", "filterName" );
    REG_INDI_SETPROP( m_indiP_flipwfsf, "flipwfsf", "presetName" );
    REG_INDI_SETPROP( m_indiP_ndState, "fwtelsim", "fsm" );
    REG_INDI_SETPROP( m_indiP_flipwfsfState, "flipwfsf", "fsm" );
    REG_INDI_SETPROP( m_indiP_fps, "camwfs", "fps" );
    REG_INDI_SETPROP( m_indiP_emgain, "camwfs", "emgain" );
    if( createStandardIndiRequestSw( m_indiP_start, "start" ) < 0 ||
        createStandardIndiRequestSw( m_indiP_stop, "stop" ) < 0 ||
        createStandardIndiText( m_indiP_output, "output" ) < 0 || createStandardIndiText( m_indiP_bank, "bank" ) < 0 ||
        createROIndiText( m_indiP_collection, "collection", "phase" ) < 0 ||
        createROIndiText( m_indiP_observation, "observation", "nd" ) < 0 )
    {
        return -1;
    }
    m_indiP_observation.add( pcf::IndiElement( "flipwfsf" ) );
    m_indiP_observation.add( pcf::IndiElement( "camwfs_fps" ) );
    m_indiP_observation.add( pcf::IndiElement( "camwfs_emgain" ) );
    for( const auto *name : { "directory", "error", "saved", "written", "duplicates", "torn", "samples_per_second" } )
    {
        m_indiP_collection.add( pcf::IndiElement( name ) );
    }
    m_indiP_output["current"] = m_output;
    m_indiP_output["target"]  = m_output;
    m_indiP_bank["current"]   = m_bank;
    m_indiP_bank["target"]    = m_bank;
    m_indiP_collection.setPerm( pcf::IndiProperty::ReadOnly );
    if( registerIndiPropertyNew( m_indiP_start, INDI_NEWCALLBACK( m_indiP_start ) ) < 0 ||
        registerIndiPropertyNew( m_indiP_stop, INDI_NEWCALLBACK( m_indiP_stop ) ) < 0 ||
        registerIndiPropertyNew( m_indiP_output, INDI_NEWCALLBACK( m_indiP_output ) ) < 0 ||
        registerIndiPropertyNew( m_indiP_bank, INDI_NEWCALLBACK( m_indiP_bank ) ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_collection ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_observation ) < 0 )
    {
        return -1;
    }
    if( threadStart(
            m_ioThread, m_ioThreadInit, m_ioThreadID, m_indiP_ioThread, 0, "", "collection-io", this, ioStart ) < 0 )
    {
        return -1;
    }
    SHMIMMONITOR_APP_STARTUP;
    state( stateCodes::READY );
    return 0;
}

inline void dlDataCollection::ioStart( dlDataCollection *app )
{
    app->m_ioThreadID = syscall( SYS_gettid );
    while( app->m_ioThreadInit && !app->shutdown() )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }
    app->m_session.ioLoop();
}

inline int dlDataCollection::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    const bool stop = m_stopRequested.exchange( false );
    if( stop )
    {
        m_startPending = false;
        m_session.abort();
        m_connected = false;
        state( stateCodes::READY );
    }
    if( m_startRequested.exchange( false ) && !stop )
    {
        try
        {
            if( m_startPending || !m_session.canStart() )
            {
                throw std::runtime_error( "collection is busy or retains failed-run buffers" );
            }
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_indiMutex );
                m_connected                = false;
                m_nd.m_received            = {};
                m_flipwfsf.m_received      = {};
                m_nd.m_stateReceived       = {};
                m_flipwfsf.m_stateReceived = {};
                m_fps.m_received           = {};
                m_emgain.m_received        = {};
            }
            m_startPending = true;
            state( stateCodes::OPERATING );
            m_observationDeadline = std::chrono::steady_clock::now() + std::chrono::seconds( 5 );
            m_observationPoll     = {};
            std::lock_guard<std::mutex> lock( m_indiMutex );
            m_controlError.clear();
        }
        catch( const std::exception &e )
        {
            std::lock_guard<std::mutex> lock( m_indiMutex );
            m_controlError = e.what();
            log<software_error>( { __FILE__, __LINE__, e.what() } );
        }
    }
    checkObservation();
    m_session.checkTimeout();
    const auto p = m_session.progress();
    if( !m_startPending && ( p.m_phase == "idle" || p.m_phase == "draining" || p.m_phase == "complete" ||
                             p.m_phase == "cancelled" || p.m_phase == "failed" ) )
    {
        m_connected = false;
        state( stateCodes::READY );
    }
    const double elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - m_started ).count();
    if( m_finishedElapsed < 0 && ( p.m_phase == "complete" || p.m_phase == "cancelled" || p.m_phase == "failed" ) )
    {
        m_finishedElapsed = elapsed;
    }
    const double                duration = m_finishedElapsed < 0 ? elapsed : m_finishedElapsed;
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( p.m_phase == "idle" || p.m_phase == "complete" || p.m_phase == "cancelled" || p.m_phase == "failed" )
    {
        m_observationActive = false;
    }
    SHMIMMONITOR_UPDATE_INDI;
    const auto status = m_startPending                                      ? pcf::IndiProperty::Busy
                        : ( !p.m_error.empty() || !m_controlError.empty() ) ? pcf::IndiProperty::Alert
                        : m_session.canStart()                              ? pcf::IndiProperty::Ok
                                                                            : pcf::IndiProperty::Busy;
    const std::vector<std::string> names{
        "phase", "directory", "error", "saved", "written", "duplicates", "torn", "samples_per_second" };
    const std::vector<std::string> values{ m_startPending ? "checking_observation" : p.m_phase,
                                           p.m_directory,
                                           p.m_error.empty() ? m_controlError : p.m_error,
                                           std::to_string( p.m_saved ),
                                           std::to_string( p.m_written ),
                                           std::to_string( p.m_duplicates ),
                                           std::to_string( p.m_torn ),
                                           std::to_string( duration > 0 ? p.m_saved / duration : 0 ) };
    updateIfChanged( m_indiP_collection, names, values, status );
    const auto now      = std::chrono::steady_clock::now();
    const bool observed = m_nd.valid( now ) && m_flipwfsf.valid( now ) && m_fps.valid( now ) && m_emgain.valid( now );
    updateIfChanged( m_indiP_observation,
                     std::vector<std::string>{ "nd", "flipwfsf", "camwfs_fps", "camwfs_emgain" },
                     std::vector<std::string>{ m_nd.m_value,
                                               m_flipwfsf.m_value,
                                               m_fps.valid( now ) ? std::to_string( m_fps.m_value ) : "unknown",
                                               m_emgain.valid( now ) ? std::to_string( m_emgain.m_value ) : "unknown" },
                     observed ? pcf::IndiProperty::Ok : pcf::IndiProperty::Alert );
    return 0;
}

inline void dlDataCollection::checkObservation()
{
    const auto now = std::chrono::steady_clock::now();
    if( m_indiDriver && now - m_observationPoll >= std::chrono::seconds( 1 ) )
    {
        m_observationPoll = now;
        try
        {
            m_indiDriver->sendGetProperties( m_indiP_nd );
            m_indiDriver->sendGetProperties( m_indiP_flipwfsf );
            m_indiDriver->sendGetProperties( m_indiP_ndState );
            m_indiDriver->sendGetProperties( m_indiP_flipwfsfState );
            m_indiDriver->sendGetProperties( m_indiP_fps );
            m_indiDriver->sendGetProperties( m_indiP_emgain );
        }
        catch( const std::exception &e )
        {
            log<software_error>( { __FILE__, __LINE__, e.what() } );
        }
    }
    std::lock_guard<std::mutex> lock( m_indiMutex );
    const auto                  checked = std::chrono::steady_clock::now();
    const bool                  valid =
        m_nd.valid( checked ) && m_flipwfsf.valid( checked ) && m_fps.valid( checked ) && m_emgain.valid( checked );
    if( m_observationActive && ( !valid || m_observationChanged.load() ) )
    {
        m_observationChanged = true;
        m_session.abort( "fwtelsim, flipwfsf or camwfs settings changed or have invalid/stale readback" );
    }
    if( !m_startPending )
    {
        return;
    }
    try
    {
        if( !valid )
        {
            if( checked < m_observationDeadline )
            {
                return;
            }
            std::string missing;
            if( !m_nd.valid( checked ) )
            {
                missing += " fwtelsim.filterName/fsm";
            }
            if( !m_flipwfsf.valid( checked ) )
            {
                missing += " flipwfsf.presetName/fsm";
            }
            if( !m_fps.valid( checked ) )
            {
                missing += " camwfs.fps.current";
            }
            if( !m_emgain.valid( checked ) )
            {
                missing += " camwfs.emgain.current";
            }
            throw std::runtime_error( "no fresh settled readback:" + missing );
        }
        if( !m_connected.load() )
        {
            if( checked < m_observationDeadline )
            {
                return;
            }
            throw std::runtime_error( "source/channel attachment timed out" );
        }
        auto s = settings();
        s.m_context += dlCollection::observationContext( m_nd, m_flipwfsf, m_fps, m_emgain );
        m_runND              = m_nd.m_value;
        m_runFlipwfsf        = m_flipwfsf.m_value;
        m_runFPS             = m_fps.m_value;
        m_runEMGain          = m_emgain.m_value;
        m_observationChanged = false;
        { //mutex scope
            std::lock_guard<std::mutex> streamLock( m_streamMutex );
            checkCommandStream( true );
        }
        if( m_stopRequested.load() )
        {
            throw std::runtime_error( "collection start cancelled" );
        }
        if( !m_session.start( s ) )
        {
            throw std::runtime_error( "collection is busy or retains failed-run buffers" );
        }
        m_observationActive = true;
        m_started           = checked;
        m_finishedElapsed   = -1;
        m_controlError.clear();
    }
    catch( const std::exception &e )
    {
        m_controlError = e.what();
        log<software_error>( { __FILE__, __LINE__, e.what() } );
    }
    m_startPending = false;
}

inline int dlDataCollection::appShutdown()
{
    m_shutdown = true;
    m_session.abort();
    SHMIMMONITOR_APP_SHUTDOWN;
    m_session.shutdown();
    if( m_ioThread.joinable() )
    {
        m_ioThread.join();
    }
    std::lock_guard<std::mutex> lock( m_streamMutex );
    if( m_commandOpened )
    {
        ImageStreamIO_closeIm( &m_commandStream );
    }
    m_commandOpened = false;
    m_connected     = false;
    return 0;
}

inline int dlDataCollection::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    m_connected      = false;
    const auto phase = m_session.progress().m_phase;
    if( phase != "idle" && phase != "complete" && phase != "cancelled" && phase != "failed" )
    {
        m_session.abort( "source stream reconnected; partial run aborted" );
    }
    try
    {
        if( shmimMonitorT::m_dataType != _DATATYPE_FLOAT || shmimMonitorT::m_imageStream.md[0].naxis != 2 ||
            shmimMonitorT::m_imageStream.md[0].location != -1 || !shmimMonitorT::m_imageStream.array.raw )
        {
            throw std::runtime_error( "WFS stream must be a CPU-backed 2D float32 image" );
        }
        dlCollection::Settings s;
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_indiMutex );
            m_sourceSize = { shmimMonitorT::m_height, shmimMonitorT::m_width };
            s            = settings();
        }
        dlCollection::validate( s );
        std::lock_guard<std::mutex> lock( m_streamMutex );
        if( m_commandOpened )
        {
            ImageStreamIO_closeIm( &m_commandStream );
        }
        m_commandOpened = false;
        m_commandOwned  = false;
        if( ImageStreamIO_openIm( &m_commandStream, m_channel.c_str() ) != 0 )
        {
            throw std::runtime_error( "cannot attach command channel" );
        }
        m_commandOpened = true;
        if( !m_commandStream.md || !m_commandStream.array.raw || !m_commandStream.semptr )
        {
            throw std::runtime_error( "invalid command channel mapping" );
        }
        const auto &md = m_commandStream.md[0];
        if( md.naxis != 2 || md.size[0] != 50 || md.size[1] != 50 || md.datatype != _DATATYPE_FLOAT ||
            md.location != -1 || md.shared != 1 || md.nelement != 2500 || md.imdatamemsize != 10000 ||
            md.sem < SEMAPHORE_MAXVAL )
        {
            throw std::runtime_error( "command channel must be a semaphore-backed 50x50 float32 image" );
        }
        for( uint16_t i = 0; i < md.sem; ++i )
        {
            if( !m_commandStream.semptr[i] )
            {
                throw std::runtime_error( "command channel semaphore is not attached" );
            }
        }
        m_commandSemaphores = md.sem;
        char filename[4096];
        if( ImageStreamIO_filename( filename, sizeof( filename ), m_channel.c_str() ) != 0 )
        {
            throw std::runtime_error( "cannot resolve command channel filename" );
        }
        m_commandFilename = filename;
        struct stat info;
        if( ::stat( filename, &info ) != 0 )
        {
            throw std::runtime_error( "cannot identify command channel" );
        }
        m_commandInode   = info.st_ino;
        m_commandCounter = md.cnt0;
        m_connected      = m_startPending.load();
        return 0;
    }
    catch( const std::exception &e )
    {
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_indiMutex );
            m_controlError = e.what();
            m_startPending = false;
        }
        // appLogic requests READY after observing the failed attachment.
        return log<software_error>( { __FILE__, __LINE__, e.what() } );
    }
}

inline void dlDataCollection::checkCommandStream( bool newRun )
{
    struct stat info;
    if( !m_commandOpened || ::stat( m_commandFilename.c_str(), &info ) != 0 || info.st_ino != m_commandInode )
    {
        m_commandOwned = false;
        throw std::runtime_error( "command channel deleted or recreated" );
    }
    volatile auto &md = m_commandStream.md[0];
    if( md.naxis != 2 || md.size[0] != 50 || md.size[1] != 50 || md.datatype != _DATATYPE_FLOAT || md.location != -1 ||
        md.shared != 1 || md.nelement != 2500 || md.imdatamemsize != 10000 || md.sem != m_commandSemaphores ||
        md.write || ( !newRun && md.cnt0 != m_commandCounter ) )
    {
        m_commandOwned = false;
        throw std::runtime_error( "command channel changed or another writer intervened" );
    }
    if( newRun )
    {
        if( m_commandOwned )
        {
            throw std::runtime_error( "previous command channel ownership has not been released" );
        }
        const uint64_t counter = md.cnt0;
        std::atomic_thread_fence( std::memory_order_acquire );
        for( size_t i = 0; i < 2500; ++i )
        {
            if( !dlCollection::finite( m_commandStream.array.F[i] ) || m_commandStream.array.F[i] != 0.F )
            {
                throw std::runtime_error( "dedicated command channel is not zero" );
            }
        }
        std::atomic_thread_fence( std::memory_order_acquire );
        if( md.write || md.cnt0 != counter )
        {
            throw std::runtime_error( "command channel changed during Start preflight" );
        }
        m_commandCounter = counter;
    }
}

inline uint64_t dlDataCollection::publish( const float *command )
{
    std::lock_guard<std::mutex> lock( m_streamMutex );
    if( command && m_observationChanged.load() )
    {
        throw std::runtime_error( "observing configuration changed during collection" );
    }
    if( !command && !m_commandOwned )
    {
        return 0;
    }
    checkCommandStream();
    volatile auto &md = m_commandStream.md[0];
    if( command && !m_commandOwned )
    {
        for( size_t i = 0; i < 2500; ++i )
        {
            if( !dlCollection::finite( m_commandStream.array.F[i] ) || m_commandStream.array.F[i] != 0.F )
            {
                throw std::runtime_error( "dedicated command channel is not zero" );
            }
        }
    }
    md.write = 1;
    std::atomic_thread_fence( std::memory_order_seq_cst );
    if( command )
    {
        std::memcpy( m_commandStream.array.raw, command, 2500 * sizeof( float ) );
    }
    else
    {
        std::memset( m_commandStream.array.raw, 0, 2500 * sizeof( float ) );
    }
    std::atomic_thread_fence( std::memory_order_release );
    md.cnt0          = m_commandCounter + 1;
    m_commandCounter = md.cnt0;
    md.write         = 0;
    m_commandOwned   = command != nullptr;
    if( ImageStreamIO_sempost( &m_commandStream, -1 ) != 0 )
    {
        throw std::runtime_error( "command notification failed" );
    }
    if( !command )
    {
        return 0; // Source may already be detached during timeout/shutdown.
    }
    const volatile auto &source = shmimMonitorT::m_imageStream.md[0];
    return source.cnt0;
}

inline bool dlDataCollection::snapshotStable( uint64_t counter )
{
    if( m_observationChanged.load() )
    {
        return false;
    }
    const volatile auto &md = shmimMonitorT::m_imageStream.md[0];
    std::atomic_thread_fence( std::memory_order_acquire );
    if( md.write || md.cnt0 != counter )
    {
        return false;
    }
    std::lock_guard<std::mutex> lock( m_streamMutex );
    checkCommandStream();
    if( !m_commandOwned )
    {
        throw std::runtime_error( "held command ownership was lost" );
    }
    return true;
}

inline int dlDataCollection::processImage( void *src, const dev::shmimT &dummy )
{
    if( m_startPending.load() && !m_connected.load() )
    {
        // A quick Start can reuse an attachment whose monitor never observed READY.
        // Revalidate on the monitor thread before admitting frames to the new session.
        return allocate( dummy );
    }
    if( m_observationChanged.load() )
    {
        return 0;
    }
    const volatile auto &md = shmimMonitorT::m_imageStream.md[0];
    if( md.write )
    {
        return 0;
    }
    const uint64_t counter = md.cnt0;
    const uint64_t time    = md.atime.tv_sec > 0 ? uint64_t( md.atime.tv_sec ) * 1000000000ULL + md.atime.tv_nsec : 0;
    std::atomic_thread_fence( std::memory_order_acquire );
    m_session.onFrame( counter,
                       time,
                       static_cast<const float *>( src ),
                       [this, counter]
                       {
                           return snapshotStable( counter );
                       } );
    return 0;
}

inline int
dlDataCollection::textRequest( const pcf::IndiProperty &request, pcf::IndiProperty &property, std::string &value )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( !m_session.canStart() || m_startRequested.load() || m_startPending || !request.find( "target" ) )
    {
        return -1;
    }
    const auto target = request["target"].get<std::string>();
    if( target.empty() || target.find_first_of( "\r\n" ) != std::string::npos )
    {
        return -1;
    }
    value = target;
    updateIfChanged( property, "target", value );
    updateIfChanged( property, "current", value );
    return 0;
}

INDI_NEWCALLBACK_DEFN( dlDataCollection, m_indiP_start )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_start, ipRecv );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        m_startRequested = true;
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( dlDataCollection, m_indiP_stop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stop, ipRecv );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        m_stopRequested = true;
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( dlDataCollection, m_indiP_output )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_output, ipRecv );
    return textRequest( ipRecv, m_indiP_output, m_output );
}

INDI_NEWCALLBACK_DEFN( dlDataCollection, m_indiP_bank )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_bank, ipRecv );
    return textRequest( ipRecv, m_indiP_bank, m_bank );
}

INDI_SETCALLBACK_DEFN( dlDataCollection, m_indiP_nd )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nd, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_nd.update( ipRecv );
    if( m_observationActive && m_nd.m_value != m_runND )
    {
        m_observationChanged = true;
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( dlDataCollection, m_indiP_flipwfsf )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_flipwfsf, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_flipwfsf.update( ipRecv );
    if( m_flipwfsf.m_value != "in" && m_flipwfsf.m_value != "out" )
    {
        m_flipwfsf.m_value.clear();
    }
    if( m_observationActive && m_flipwfsf.m_value != m_runFlipwfsf )
    {
        m_observationChanged = true;
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( dlDataCollection, m_indiP_ndState )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ndState, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_nd.updateState( ipRecv );
    if( m_observationActive && !m_nd.valid( std::chrono::steady_clock::now() ) )
    {
        m_observationChanged = true;
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( dlDataCollection, m_indiP_flipwfsfState )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_flipwfsfState, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_flipwfsf.updateState( ipRecv );
    if( m_observationActive && !m_flipwfsf.valid( std::chrono::steady_clock::now() ) )
    {
        m_observationChanged = true;
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( dlDataCollection, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fps, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_fps.update( ipRecv );
    if( m_observationActive && m_fps.m_value != m_runFPS )
    {
        m_observationChanged = true;
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( dlDataCollection, m_indiP_emgain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_emgain, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_emgain.update( ipRecv );
    if( m_observationActive && m_emgain.m_value != m_runEMGain )
    {
        m_observationChanged = true;
    }
    return 0;
}

} // namespace app
} // namespace MagAOX
#endif // dlDataCollection_hpp
