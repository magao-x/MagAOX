/** \file accelSim.hpp
 * \brief Continuous raw-ADC accelerometer simulator with the mcp3208Ctrl interface.
 * \ingroup accelSim_files
 */
#ifndef accelSim_hpp
#define accelSim_hpp

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <sys/stat.h>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "accelSimModel.hpp"

/** \defgroup accelSim Accelerometer Simulator
 * \brief Continuous physical vibrations published as raw X/Y ADC counts.
 * \ingroup apps
 */
/** \defgroup accelSim_files Accelerometer Simulator Files
 * \ingroup accelSim
 */

namespace MagAOX
{
namespace app
{

/// Simulated accelerometer acquisition; frameGrabber exclusively owns output publication.
class accelSim : public MagAOXApp<true>, public dev::frameGrabber<accelSim>, public dev::telemeter<accelSim>
{
    friend class dev::frameGrabber<accelSim>;
    friend class dev::telemeter<accelSim>;
    typedef dev::frameGrabber<accelSim> frameGrabberT;
    typedef dev::telemeter<accelSim>    telemeterT;
    static constexpr bool               c_frameGrabber_flippable = false; ///< Preserve fixed X/Y ordering.

  protected:
    accelSimDetail::parameters m_parameters;                      ///< Startup-only physics/calibration configuration.
    accelSimDetail::model      m_model;                           ///< Acquisition-thread-owned physical state and RNG.
    accelSimDetail::schedule   m_schedule;                        ///< Absolute monotonic timer schedule.
    std::atomic<double>        m_requestedFps{ 6000 };            ///< Validated callback-to-worker rate request.
    std::atomic<double>        m_activeFps{ 6000 };               ///< Rate currently used by the acquisition worker.
    std::atomic<uint64_t> m_rateRequest{ 0 }, m_rateApplied{ 0 }; ///< Requested and worker-acknowledged FPS revisions.
    std::atomic<double>   m_alpha{ .01 };      ///< Timing EMA coefficient, independently updated by INDI.
    std::atomic<int>      m_postDelay{ 0 };    ///< Signed synchronized phase offset in microseconds.
    double                m_maxFrequency{ 0 }; ///< Immutable maximum mode frequency for callback validation.
    std::string           m_fpsDevice, m_fpsProperty{ "fps" }, m_fpsElement{ "current" }; ///< External FPS source.
    std::string           m_channelsDevice, m_channelsProperty{ "numChannels" },
        m_channelsElement{ "current" };            ///< Channel source.
    double      m_fpsTol{ 0 }, m_channelsTol{ 0 }; ///< Compatibility keys, unused by the hardware callbacks as well.
    std::string m_syncName;                        ///< Optional semaphore trigger stream; empty selects timer mode.
    double      m_dtTransfer{ 3000 }, m_wfsProcess{ 51500 }, m_dtF{ 10000 },
        m_wfsRead{ 276100 };                                                ///< Delay-model terms in ns.
    double m_lockAbsolute{ 50000 }, m_lockFraction{ .1 }, m_guard{ 20000 }; ///< Delay lock criteria and cadence margin.
    IMAGE  m_syncImage{};                  ///< Synchronization input only; never the output image.
    bool   m_syncOpen{ false };            ///< Whether the source mapping is owned by this app.
    int    m_syncIndex{ 5 };               ///< Claimed source semaphore index, initially preferring index five.
    ino_t  m_syncInode{ 0 };               ///< Backing-file inode used to detect source replacement.
    accelSimDetail::syncClock m_syncClock; ///< Committed source and nominal-time accounting.
    accelSimDetail::syncTick  m_syncTick;  ///< Notification awaiting acquisition or explicit cancellation.
    bool                      m_syncPending{ false }; ///< True until the consumed notification has been finalized.
    int64_t                   m_lastArrival{ 0 }, m_lastAcquisition{ 0 }; ///< Monotonic timestamps for observed timing.
    int64_t                   m_triggerPrediction{ 0 };                   ///< Last predicted synchronized trigger time.
    int64_t                   m_arrival{ 0 };   ///< Current synchronized semaphore-arrival timestamp.
    bool                      m_fresh{ false }; ///< True only for an acquired sample not yet copied.

    using enum accelSimDetail::timingIndex;
    /// Hardware-compatible element names in timingIndex order.
    inline static constexpr std::array<const char *, timingCount> c_timingNames{ "avg_read_latency_us",
                                                                                 "synchro_delay_us",
                                                                                 "synchro_delay_target_us",
                                                                                 "delay_applied_us",
                                                                                 "delay_model_us",
                                                                                 "delay_phase_error_us",
                                                                                 "delay_lock",
                                                                                 "delay_budget_us",
                                                                                 "avg_non_delay_service_us",
                                                                                 "delay_capped",
                                                                                 "read_latency_error_us",
                                                                                 "avg_semaphore_period_us",
                                                                                 "channel_readout_us",
                                                                                 "trigger_interval_us",
                                                                                 "trigger_time_us",
                                                                                 "local_frame_seq",
                                                                                 "sync_frames_received",
                                                                                 "sync_frames_written",
                                                                                 "sync_frames_dropped",
                                                                                 "sync_frame_id_gap_count",
                                                                                 "sync_producer_frame_id",
                                                                                 "sync_producer_frame_delta",
                                                                                 "mode_code" };

    using diagnostics = accelSimDetail::diagnostics;
    diagnostics m_diagnostics, m_snapshot; ///< Worker state and synchronized app-loop snapshot.
    std::mutex  m_snapshotMutex;           ///< Protects only snapshot copies, never acquisition work.
    std::mutex  m_controlMutex;            ///< Serializes app-owned INDI properties and their callbacks.
    /// Compatible accelerometer controls, serialized by m_controlMutex.
    pcf::IndiProperty m_indiP_fps, m_indiP_numChannels, m_indiP_alpha, m_indiP_synchroDelay;
    /// Diagnostic properties and configured external subscriptions.
    pcf::IndiProperty m_indiP_timingDiag, m_indiP_simDiag, m_indiP_fpsSource, m_indiP_numChannelsSource;

    /// Define the accelerometer-owned properties, separated from thread startup for testing.
    int setupProperties();
    /// Send coherent app-owned controls and diagnostics at ordinary app-loop cadence.
    void updateINDI( const diagnostics &snapshot /**< [in] coherent worker snapshot */ );
    /// Open and claim the synchronization source, returning positive to retry.
    int openSync();
    /// Release the claimed source semaphore and mapping.
    void closeSync();
    /// Detect a source mapping that vanished or was replaced.
    bool syncStale();
    /// Wait for a synchronization trigger, retaining tentative accounting through the phase wait.
    int waitSync();
    /// Finalize any pending notification once, including explicit cancellation.
    void finishSync( bool acquired /**< [in] whether a fresh model sample was generated */ );
    /// Clear synchronization status and publish it even when no frames can be acquired.
    void invalidateSync();
    /// Transfer diagnostics; forced transfers wait only for the app-loop snapshot copy.
    void snapshotDiagnostics( bool force /**< [in] require delivery for lifecycle events */ );
    /// Compute hardware-compatible synchronized phase, budget, and lock diagnostics.
    void syncTiming( int64_t arrival /**< [in] monotonic semaphore arrival */ );
    /// Wait interruptibly for an absolute monotonic deadline.
    int waitUntil( int64_t deadline /**< [in] monotonic deadline in ns */ );
    /// Get monotonic nanoseconds without conversion of epoch time to double.
    static int64_t monotonicNow();
    /// Send a property state, leaving its last accepted values intact on error.
    int propertyState( pcf::IndiProperty &property /**< [in,out] property to acknowledge */,
                       bool               valid /**< [in] whether the update was accepted */ );
    /// Validate and enqueue an FPS update without touching model state.
    int requestRate( double value /**< [in] requested FPS */ );
    /// Validate the fixed two-channel interface.
    int requestChannels( double value /**< [in] requested channel count */ );
    /// Extract a finite numeric target, falling back to current for compatible clients.
    static double target( const pcf::IndiProperty &property /**< [in] incoming update */ );

  public:
    /// Construct without opening any hardware or shared-memory stream.
    accelSim();
    /// Register app and helper configuration keys.
    void setupConfig() override;
    /// Load and validate startup configuration.
    void loadConfig() override;
    /// Load configuration without starting threads, for focused tests.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] parsed configuration */ );
    /// Register properties and start the existing helper threads.
    int appStartup() override;
    /// Publish diagnostics at ordinary application cadence.
    int appLogic() override;
    /// Join acquisition before releasing synchronization resources.
    int appShutdown() override;
    /// Set the two-channel stream geometry and connect any synchronization source.
    int configureAcquisition();
    /// Return the nominal active sampling rate to frameGrabber.
    float fps();
    /// Initialize acquisition pacing and source counters.
    int startAcquisition();
    /// Acquire one fresh sample; positive returns never publish an old pair.
    int acquireAndCheckValid();
    /// Copy precisely one X/Y pair; framework owns metadata and semaphore posts.
    int loadImageIntoStream( void *destination /**< [out] frameGrabber destination plane */ );
    /// Transfer a nonblocking diagnostic snapshot after successful publication.
    int frameGrabberPostPublish( IMAGE *stream /**< [in] framework-owned output, not modified here */ );
    /// Release a stale synchronization source before framework reconfiguration.
    int reconfig();
    /// Check the existing frame-grabber telemetry cadence.
    int checkRecordTimes();
    /// Record the existing timing telemetry schema.
    int recordTelem( const telem_fgtimings *tag /**< [in] telemetry dispatch tag */ );
    /// Handle a local nominal-rate update.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_fps );
    /// Handle the compatible fixed-channel control.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_numChannels );
    /// Handle timing EMA changes, independent of AR damping.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_alpha );
    /// Handle signed synchronization phase offsets.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_synchroDelay );
    /// Follow configured external nominal-rate metadata.
    INDI_SETCALLBACK_DECL( accelSim, m_indiP_fpsSource );
    /// Follow configured external channel metadata, accepting only two.
    INDI_SETCALLBACK_DECL( accelSim, m_indiP_numChannelsSource );
};

inline accelSim::accelSim() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_shmimName = "acc000";
}

inline void accelSim::setupConfig()
{
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );
    const auto add = [this]( const std::string &section,
                             const std::string &key,
                             const std::string &type,
                             const std::string &description )
    {
        const std::string name = section + "." + key;
        config.add( name, "", name, argType::Required, section, key, false, type, description );
    };
    add( "sim", "fps", "double", "Initial nominal sampling rate [Hz], default 6000." );
    add( "sim", "seed", "uint64_t", "Numerical random seed, default 42." );
    add( "sim", "modeSeed", "uint64_t", "High-frequency parameter seed, default 123." );
    add( "sim", "leverArm", "double", "Effective lever arm [m], default 10." );
    add( "sensor", "noisePSD", "double", "One-sided acceleration noise PSD, default 1e-8." );
    add( "adc", "zero", "double", "ADC zero point [counts], default 2048." );
    add( "adc", "voltsPerBit", "double", "Empirical volts per count, default 0.0083799." );
    add( "adc", "sensitivity", "double", "Empirical sensitivity, default 1.02." );
    add( "accel", "numChannels", "int", "Compatibility channel count; must be 2." );
    for( const std::string section : { "fps", "numChannels" } )
    {
        add( section, "device", "string", "External INDI source; empty disables subscription." );
        add( section, "property", "string", "External property name." );
        add( section, "element", "string", "External element name, default current." );
        add( section, "tol", "double", "Compatibility tolerance key; hardware callback does not apply it." );
    }
    add( "synchro", "shmimName", "string", "Trigger input stream; empty selects timer acquisition." );
    add( "synchro", "postDelay", "int", "Signed phase offset [us], default 0." );
    add( "synchro", "alpha", "double", "Timing EMA coefficient, default 0.01." );
    for( const std::string key : { "dtTransfer_ns",
                                   "wfsProcess_ns",
                                   "dtF_ns",
                                   "wfsRead_ns",
                                   "delayLockAbsThreshold_ns",
                                   "delayLockFracThreshold",
                                   "cadenceGuard_ns" } )
        add( "synchro", key, "double", "mcp3208Ctrl synchronized timing-model parameter." );
}

inline int accelSim::loadConfigImpl( mx::app::appConfigurator &config )
{
    FRAMEGRABBER_LOAD_CONFIG( config );
    TELEMETER_LOAD_CONFIG( config );
    config( m_parameters.m_fps, "sim.fps" );
    config( m_parameters.m_seed, "sim.seed" );
    config( m_parameters.m_modeSeed, "sim.modeSeed" );
    config( m_parameters.m_leverArm, "sim.leverArm" );
    config( m_parameters.m_noisePSD, "sensor.noisePSD" );
    config( m_parameters.m_zero, "adc.zero" );
    config( m_parameters.m_voltsPerBit, "adc.voltsPerBit" );
    config( m_parameters.m_sensitivity, "adc.sensitivity" );
    config( m_fpsDevice, "fps.device" );
    config( m_fpsProperty, "fps.property" );
    config( m_fpsElement, "fps.element" );
    config( m_fpsTol, "fps.tol" );
    config( m_channelsDevice, "numChannels.device" );
    config( m_channelsProperty, "numChannels.property" );
    config( m_channelsElement, "numChannels.element" );
    config( m_channelsTol, "numChannels.tol" );
    config( m_syncName, "synchro.shmimName" );
    config( m_dtTransfer, "synchro.dtTransfer_ns" );
    config( m_wfsProcess, "synchro.wfsProcess_ns" );
    config( m_dtF, "synchro.dtF_ns" );
    config( m_wfsRead, "synchro.wfsRead_ns" );
    config( m_lockAbsolute, "synchro.delayLockAbsThreshold_ns" );
    config( m_lockFraction, "synchro.delayLockFracThreshold" );
    config( m_guard, "synchro.cadenceGuard_ns" );
    double alpha     = .01;
    int    postDelay = 0;
    double channels  = 2;
    config( alpha, "synchro.alpha" );
    config( postDelay, "synchro.postDelay" );
    config( channels, "accel.numChannels" );
    try
    {
        if( channels != 2 || !std::isfinite( alpha ) || postDelay < -1000000 || postDelay > 1000000 ||
            ( !m_syncName.empty() && m_syncName == m_shmimName ) )
            throw std::invalid_argument( "invalid channels, alpha, phase offset, or self-trigger stream" );
        for( double *value :
             { &m_dtTransfer, &m_wfsProcess, &m_dtF, &m_wfsRead, &m_lockAbsolute, &m_lockFraction, &m_guard } )
        {
            if( !std::isfinite( *value ) )
                throw std::invalid_argument( "nonfinite timing configuration" );
            *value = std::max( 0.0, *value );
        }
        if( !std::isfinite( m_dtTransfer + m_wfsProcess + m_dtF + m_wfsRead ) )
            throw std::invalid_argument( "timing-model overflow" );
        m_model.configure( m_parameters, accelSimDetail::referenceModes( m_parameters.m_modeSeed ) );
        m_maxFrequency = m_model.m_modes.back().m_spec.m_frequency;
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    m_alpha        = std::clamp( alpha, 0.0, 1.0 );
    m_postDelay    = postDelay;
    m_activeFps    = m_parameters.m_fps;
    m_requestedFps = m_parameters.m_fps;
    return 0;
}

inline void accelSim::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
        m_shutdown = true;
}

inline int accelSim::setupProperties()
{
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_fps, "fps", 0, 10000, 1, "%d", "", "" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_numChannels, "numChannels", 0, 8, 1, "%d", "", "" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_alpha, "alpha", 0., 1., .00001, "%.5f", "", "" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_synchroDelay, "synchroDelay", -1000000, 1000000, 1, "%d", "us", "" );
    for( const char *element : { "current", "target" } )
    {
        m_indiP_fps[element]          = m_activeFps.load();
        m_indiP_numChannels[element]  = 2;
        m_indiP_alpha[element]        = m_alpha.load();
        m_indiP_synchroDelay[element] = m_postDelay.load();
    }
    CREATE_REG_INDI_RO_NUMBER( m_indiP_timingDiag, "timingDiag", "Timing Diagnostics", "Diagnostics" );
    for( const char *name : c_timingNames )
        m_indiP_timingDiag.add( pcf::IndiElement( name ) );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_simDiag, "sim_diag", "Simulation Diagnostics", "Diagnostics" );
    for( const char *name : { "skipped_publications",
                              "discontinuities",
                              "late_samples",
                              "max_lateness_us",
                              "source_period_ratio",
                              "clip_x_low",
                              "clip_x_high",
                              "clip_y_low",
                              "clip_y_high" } )
        m_indiP_simDiag.add( pcf::IndiElement( name ) );
    if( !m_fpsDevice.empty() )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );
    }
    if( !m_channelsDevice.empty() )
    {
        REG_INDI_SETPROP( m_indiP_numChannelsSource, m_channelsDevice, m_channelsProperty );
    }
    return 0;
}

inline int accelSim::appStartup()
{
    if( setupProperties() < 0 )
        return -1;
    TELEMETER_APP_STARTUP;
    FRAMEGRABBER_APP_STARTUP;
    state( stateCodes::OPERATING );
    return 0;
}

inline int accelSim::appLogic()
{
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    FRAMEGRABBER_UPDATE_INDI;
    diagnostics snapshot;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_snapshotMutex );
        snapshot = m_snapshot;
    }
    updateINDI( snapshot );
    return 0;
}

inline void accelSim::updateINDI( const diagnostics &snapshot )
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_controlMutex );
        const auto                  fpsState = m_indiP_fps.getState() == INDI_ALERT           ? INDI_ALERT
                                               : m_rateApplied.load() == m_rateRequest.load() ? INDI_OK
                                                                                              : INDI_BUSY;
        updatesIfChanged<double>(
            m_indiP_fps, { "current", "target" }, { m_activeFps.load(), m_requestedFps.load() }, fpsState );
        updatesIfChanged<double>(
            m_indiP_alpha, { "current", "target" }, { m_alpha.load(), m_alpha.load() }, m_indiP_alpha.getState() );
        updatesIfChanged<int>( m_indiP_synchroDelay,
                               { "current", "target" },
                               { m_postDelay.load(), m_postDelay.load() },
                               m_indiP_synchroDelay.getState() );
        indi::updatesIfChanged<double>( m_indiP_timingDiag,
                                        c_timingNames,
                                        std::vector<double>( snapshot.m_timing.begin(), snapshot.m_timing.end() ),
                                        m_indiDriver );
        updatesIfChanged<double>( m_indiP_simDiag,
                                  { "skipped_publications",
                                    "discontinuities",
                                    "late_samples",
                                    "max_lateness_us",
                                    "source_period_ratio",
                                    "clip_x_low",
                                    "clip_x_high",
                                    "clip_y_low",
                                    "clip_y_high" },
                                  { double( snapshot.m_skipped ),
                                    double( snapshot.m_discontinuities ),
                                    double( snapshot.m_late ),
                                    snapshot.m_maxLateUs,
                                    snapshot.m_cadenceRatio,
                                    double( snapshot.m_low[0] ),
                                    double( snapshot.m_high[0] ),
                                    double( snapshot.m_low[1] ),
                                    double( snapshot.m_high[1] ) } );
    }
}

inline int accelSim::appShutdown()
{
    FRAMEGRABBER_APP_SHUTDOWN;
    closeSync();
    log<text_log>( "accelSim skipped=" + std::to_string( m_diagnostics.m_skipped ) +
                   " discontinuities=" + std::to_string( m_diagnostics.m_discontinuities ) +
                   " late_samples=" + std::to_string( m_diagnostics.m_late ) +
                   " max_lateness_us=" + std::to_string( m_diagnostics.m_maxLateUs ) + " clip_x_low=" +
                   std::to_string( m_model.m_clipLow[0] ) + " clip_x_high=" + std::to_string( m_model.m_clipHigh[0] ) +
                   " clip_y_low=" + std::to_string( m_model.m_clipLow[1] ) +
                   " clip_y_high=" + std::to_string( m_model.m_clipHigh[1] ) );
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline int64_t accelSim::monotonicNow()
{
    timespec now;
    if( clock_gettime( CLOCK_MONOTONIC, &now ) != 0 )
        throw std::runtime_error( "CLOCK_MONOTONIC failed" );
    return int64_t( now.tv_sec ) * 1000000000LL + now.tv_nsec;
}

inline int accelSim::waitUntil( int64_t deadline )
{
    while( !shutdown() && !m_reconfig )
    {
        if( m_requestedFps.load() != m_activeFps.load() )
            return 1;
        const int64_t remaining = deadline - monotonicNow();
        if( remaining <= 0 )
            return 0;
        // Absolute schedule, bounded sleeps for shutdown/rate changes; no busy wait.
        const int64_t interval = std::min<int64_t>( remaining, 1000000 );
        timespec      delay{ 0, static_cast<long>( interval ) };
        if( nanosleep( &delay, nullptr ) < 0 && errno != EINTR )
            return -1;
    }
    return 1;
}

inline int accelSim::configureAcquisition()
{
    m_width    = 2;
    m_height   = 1;
    m_dataType = _DATATYPE_UINT16;
    return m_syncName.empty() ? 0 : openSync();
}

inline float accelSim::fps()
{
    return m_activeFps.load();
}

inline int accelSim::startAcquisition()
{
    m_schedule.reset( monotonicNow(), m_activeFps.load() );
    m_lastArrival = m_lastAcquisition = m_triggerPrediction = 0;
    m_syncClock                                             = {};
    m_syncPending                                           = false;
    m_fresh                                                 = false;
    m_diagnostics.m_timing                                  = {};
    m_diagnostics.m_timing[modeCode]                        = m_syncName.empty() ? 0 : 1;
    m_diagnostics.m_timing[delayTarget]                     = m_postDelay.load();
    if( m_syncOpen )
        ImageStreamIO_semflush( &m_syncImage, m_syncIndex );
    snapshotDiagnostics( true );
    return 0;
}

inline int accelSim::openSync()
{
    if( m_syncOpen )
        return 0;
    if( ImageStreamIO_openIm( &m_syncImage, m_syncName.c_str() ) != 0 )
        return 1;
    m_syncOpen = true;
    char        filename[1024];
    struct stat status;
    ImageStreamIO_filename( filename, sizeof( filename ), m_syncName.c_str() );
    if( m_syncImage.md[0].sem < SEMAPHORE_MAXVAL || stat( filename, &status ) != 0 )
    {
        m_syncIndex = -1;
        closeSync();
        return 1;
    }
    m_syncInode = status.st_ino;
    m_syncIndex = ImageStreamIO_getsemwaitindex( &m_syncImage, 5 );
    if( m_syncIndex < 0 )
    {
        closeSync();
        return 1;
    }
    return 0;
}

inline void accelSim::closeSync()
{
    if( m_syncOpen )
    {
        if( m_syncIndex >= 0 )
            m_syncImage.semReadPID[m_syncIndex] = 0;
        ImageStreamIO_closeIm( &m_syncImage );
    }
    m_syncImage = {};
    m_syncOpen  = false;
    m_syncIndex = -1;
    m_syncInode = 0;
}

inline bool accelSim::syncStale()
{
    if( !m_syncOpen || m_syncImage.md[0].sem <= 0 )
        return true;
    char        filename[1024];
    struct stat status;
    ImageStreamIO_filename( filename, sizeof( filename ), m_syncName.c_str() );
    return stat( filename, &status ) != 0 || status.st_ino != m_syncInode;
}

inline void accelSim::syncTiming( int64_t arrival )
{
    auto        &t     = m_diagnostics.m_timing;
    const double alpha = m_alpha.load();
    if( m_lastArrival )
        t[semaphorePeriod] = alpha * ( arrival - m_lastArrival ) / 1000.0 + ( 1 - alpha ) * t[semaphorePeriod];
    m_lastArrival       = arrival;
    const double period = t[semaphorePeriod];
    t[delayModel]       = 0;
    t[delayTarget]      = m_postDelay.load();
    if( period > 0 )
    {
        const auto wrap = [period]( double value )
        {
            const double v = std::fmod( value, period );
            return v < 0 ? v + period : v;
        };
        t[delayModel]  = wrap( .5 * period - ( m_dtTransfer + m_wfsProcess + m_dtF + m_wfsRead ) / 1000 );
        t[delayTarget] = wrap( t[delayModel] + m_postDelay.load() );
    }
    t[delayBudget]       = std::max( 0.0, period - t[service] - m_guard / 1000 );
    const double desired = std::max( 0.0, t[delayTarget] );
    t[delayApplied]      = std::min( desired, t[delayBudget] );
    t[capped]            = desired > t[delayBudget] ? 1 : 0;
    t[delay]             = t[delayTarget];
    m_diagnostics.updatePhaseError();
    t[lock] = period > 0 && std::abs( t[phaseError] ) <= m_lockAbsolute / 1000 &&
                      std::abs( t[phaseError] ) <= m_lockFraction * period
                  ? 1
                  : 0;
    m_diagnostics.updateTrigger( arrival, m_triggerPrediction );
    m_diagnostics.m_cadenceRatio = period * m_activeFps.load() / 1e6;
}

inline void accelSim::finishSync( bool acquired )
{
    if( !m_syncPending )
        return;
    m_diagnostics.finishSync( m_syncClock, m_syncTick, acquired );
    m_syncPending = false;
}

inline void accelSim::invalidateSync()
{
    finishSync( false );
    m_diagnostics.invalidateSync( m_postDelay.load() );
    m_lastArrival = m_lastAcquisition = m_triggerPrediction = 0;
    m_fresh                                                 = false;
    snapshotDiagnostics( true );
}

inline int accelSim::waitSync()
{
    if( !m_syncOpen || m_syncIndex < 0 )
    {
        m_reconfig = true;
        invalidateSync();
        return 1;
    }
    timespec timeout;
    if( clock_gettime( CLOCK_REALTIME, &timeout ) < 0 )
    {
        invalidateSync();
        return -1;
    }
    ++timeout.tv_sec;
    if( sem_timedwait( m_syncImage.semptr[m_syncIndex], &timeout ) < 0 )
    {
        const int waitError = errno;
        if( waitError != EINTR )
        {
            if( waitError != ETIMEDOUT || syncStale() )
                m_reconfig = true;
            invalidateSync(); // Also invalidate a stalled but still mapped source.
        }
        return 1;
    }
    ++m_diagnostics.m_timing[received];
    m_arrival     = monotonicNow();
    m_syncTick    = m_syncClock.prepare( m_syncImage.md[0].cnt0, m_arrival, m_activeFps.load() );
    m_syncPending = m_syncTick.m_newFrame;
    if( !m_syncPending )
        return 1; // Drain repeated notifications without replay or duplicate accounting.
    if( m_syncTick.m_reset )
    {
        m_model.restart();
        ++m_diagnostics.m_discontinuities;
        m_diagnostics.invalidateSync( m_postDelay.load() );
        m_lastArrival = m_lastAcquisition = m_triggerPrediction = 0;
    }
    if( !m_syncTick.m_due )
    {
        finishSync( false ); // Source progressed within an already-accounted nominal opportunity.
        return 1;
    }
    syncTiming( m_arrival );
    const int result = waitUntil( m_arrival + int64_t( std::llround( m_diagnostics.m_timing[delayApplied] * 1000 ) ) );
    if( result != 0 )
    {
        if( result < 0 )
            m_reconfig = true;
        invalidateSync(); // Counts the canceled current opportunity as well as older missed ticks.
    }
    return result;
}

inline int accelSim::acquireAndCheckValid()
{
    m_fresh = false;
    if( shutdown() || m_reconfig )
        return 1;
    try
    {
        if( m_rateRequest.load() != m_rateApplied.load() )
        {
            double   requested;
            uint64_t revision;
            { //mutex scope
                std::lock_guard<std::mutex> guard( m_controlMutex );
                requested = m_requestedFps.load();
                revision  = m_rateRequest.load();
            }
            if( requested != m_activeFps.load() )
            {
                m_model.setRate( requested );
                m_activeFps = requested;
                ++m_diagnostics.m_discontinuities;
                m_schedule.reset( monotonicNow(), requested );
                m_lastArrival = m_lastAcquisition = m_triggerPrediction = 0;
                m_syncClock.resetTime();
                if( !m_syncName.empty() )
                    invalidateSync();
            }
            m_rateApplied = revision; // A newer callback remains Busy until its own boundary.
        }
        uint64_t skipped = 0;
        int64_t  deadline;
        if( m_syncName.empty() )
        {
            deadline         = m_schedule.deadline();
            const int result = waitUntil( deadline );
            if( result != 0 )
                return result;
            const int64_t now = monotonicNow();
            skipped           = m_schedule.consume( now );
            // Lateness is measured against the original deadline, including skipped periods.
        }
        else
        {
            const int result = waitSync();
            if( result != 0 )
                return result;
            skipped  = m_syncTick.m_advance;
            deadline = m_arrival + int64_t( std::llround( m_diagnostics.m_timing[delayApplied] * 1000 ) );
        }
        if( shutdown() || m_reconfig || ( !m_syncName.empty() && m_requestedFps.load() != m_activeFps.load() ) )
        {
            if( !m_syncName.empty() )
                invalidateSync();
            return 1;
        }
        const int64_t acquired = monotonicNow();
        if( clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp ) < 0 )
        {
            if( !m_syncName.empty() )
                invalidateSync();
            return -1;
        }
        const double late = std::max( 0.0, ( acquired - deadline ) / 1000.0 );
        if( late > 0 )
            ++m_diagnostics.m_late;
        m_diagnostics.m_maxLateUs = std::max( m_diagnostics.m_maxLateUs, late );
        if( m_syncName.empty() )
            m_diagnostics.m_skipped += skipped;
        if( m_model.advance( skipped ) )
            ++m_diagnostics.m_discontinuities;
        const int64_t completed = monotonicNow();
        auto         &t         = m_diagnostics.m_timing;
        t[channelReadout]       = ( completed - acquired ) / 1000.0;
        if( m_syncName.empty() )
        {
            t[triggerInterval] = m_lastAcquisition ? ( acquired - m_lastAcquisition ) / 1000.0 : 0;
            t[delayTarget]     = m_postDelay.load();
        }
        else
        {
            const double latency = ( acquired - m_arrival ) / 1000.0;
            const double alpha   = m_alpha.load();
            t[readLatency]       = m_lastAcquisition ? alpha * latency + ( 1 - alpha ) * t[readLatency] : latency;
            t[latencyError]      = t[readLatency] - t[delayTarget];
            // Same bounded feedback diagnostic as mcp3208Ctrl; the next trigger recalculates its target.
            double updated = std::max( 0.0, t[delayTarget] - .1 * t[latencyError] );
            if( t[capped] && updated > t[delayTarget] )
                updated = t[delayApplied];
            t[delay]             = std::min( updated, t[delayBudget] );
            const double elapsed = std::max( 0.0, ( completed - m_arrival ) / 1000.0 - t[delayApplied] );
            t[service]           = m_lastAcquisition ? alpha * elapsed + ( 1 - alpha ) * t[service] : elapsed;
        }
        finishSync( true );
        m_lastAcquisition = acquired;
        m_fresh           = true;
        return 0;
    }
    catch( const std::exception &error )
    {
        m_shutdown = true;
        if( !m_syncName.empty() )
            invalidateSync();
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
}

inline int accelSim::loadImageIntoStream( void *destination )
{
    if( !m_fresh || destination == nullptr )
        return -1;
    std::memcpy( destination, m_model.m_sample.m_adc.data(), 2 * sizeof( uint16_t ) );
    m_fresh = false;
    ++m_diagnostics.m_timing[frameSequence];
    if( !m_syncName.empty() )
        ++m_diagnostics.m_timing[written];
    return 0;
}

inline void accelSim::snapshotDiagnostics( bool force )
{
    m_diagnostics.m_low  = m_model.m_clipLow;
    m_diagnostics.m_high = m_model.m_clipHigh;
    { //mutex scope
        std::unique_lock<std::mutex> guard( m_snapshotMutex, std::defer_lock );
        if( force )
            guard.lock();
        else if( !guard.try_lock() )
            return;
        m_snapshot = m_diagnostics;
    }
}

inline int accelSim::frameGrabberPostPublish( IMAGE *stream )
{
    static_cast<void>( stream );
    snapshotDiagnostics( false );
    return 0;
}

inline int accelSim::reconfig()
{
    finishSync( false );
    closeSync();
    m_model.restart();
    ++m_diagnostics.m_discontinuities;
    if( !m_syncName.empty() )
        invalidateSync();
    return 0;
}

inline int accelSim::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int accelSim::recordTelem( const telem_fgtimings *tag )
{
    static_cast<void>( tag );
    return recordFGTimings( true );
}

inline int accelSim::propertyState( pcf::IndiProperty &property, bool valid )
{
    property.setState( valid ? pcf::IndiProperty::Ok : pcf::IndiProperty::Alert );
    if( m_indiDriver )
        m_indiDriver->sendSetProperty( property );
    return valid ? 0 : -1;
}

inline double accelSim::target( const pcf::IndiProperty &property )
{
    try
    {
        if( property.find( "target" ) )
            return property["target"].get<double>();
        if( property.find( "current" ) )
            return property["current"].get<double>();
    }
    catch( const std::exception & )
    {
    }
    return std::numeric_limits<double>::quiet_NaN();
}

inline int accelSim::requestRate( double value )
{
    const bool valid = std::isfinite( value ) && value > 2 * m_maxFrequency && value <= 10000;
    if( !valid )
        return propertyState( m_indiP_fps, false );
    m_requestedFps = value;
    ++m_rateRequest;
    updateIfChanged( m_indiP_fps, "target", value, INDI_BUSY );
    m_indiP_fps["target"] = value;
    m_indiP_fps.setState( INDI_BUSY );
    return 0;
}

inline int accelSim::requestChannels( double value )
{
    return propertyState( m_indiP_numChannels, value == 2 );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "fps" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    return requestRate( target( ipRecv ) );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_numChannels )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "numChannels" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    return requestChannels( target( ipRecv ) );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_alpha )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "alpha" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const double                value = target( ipRecv );
    if( !std::isfinite( value ) )
        return propertyState( m_indiP_alpha, false );
    m_alpha                  = std::clamp( value, 0.0, 1.0 );
    m_indiP_alpha["current"] = m_alpha.load();
    m_indiP_alpha["target"]  = m_alpha.load();
    return propertyState( m_indiP_alpha, true );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_synchroDelay )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "synchroDelay" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const double                value = target( ipRecv );
    if( !std::isfinite( value ) || value < -1000000 || value > 1000000 )
        return propertyState( m_indiP_synchroDelay, false );
    m_postDelay                     = static_cast<int>( value );
    m_indiP_synchroDelay["current"] = m_postDelay.load();
    m_indiP_synchroDelay["target"]  = m_postDelay.load();
    return propertyState( m_indiP_synchroDelay, true );
}

inline INDI_SETCALLBACK_DEFN( accelSim, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( !ipRecv.find( m_fpsElement ) )
        return propertyState( m_indiP_fps, false );
    try
    {
        return requestRate( ipRecv[m_fpsElement].get<double>() );
    }
    catch( const std::exception & )
    {
        return propertyState( m_indiP_fps, false );
    }
}

inline INDI_SETCALLBACK_DEFN( accelSim, m_indiP_numChannelsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numChannelsSource, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( !ipRecv.find( m_channelsElement ) )
        return propertyState( m_indiP_numChannels, false );
    try
    {
        return requestChannels( ipRecv[m_channelsElement].get<double>() );
    }
    catch( const std::exception & )
    {
        return propertyState( m_indiP_numChannels, false );
    }
}

} // namespace app
} // namespace MagAOX
#endif
