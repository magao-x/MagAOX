/** \file accelFFCtrl.hpp
 * \brief Live non-actuating MagAO-X accelerometer pointing observer.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrl_hpp
#define accelFFCtrl_hpp

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <unistd.h>
#ifdef __linux__
    #include <sys/syscall.h>
#endif

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "accelFFCtrlStream.hpp"

namespace MagAOX
{
namespace app
{

/// Continuous ADC observation; one worker owns input mapping, retained-frame cursor and estimator.
class accelFFCtrl : public MagAOXApp<true>
{
  protected:
    accelFFCtrlDetail::parameters m_requested; ///< Coherent callback-owned candidate; copied only under m_controlMutex.
    uint64_t   m_revision{ 0 };      ///< Parameter/reset revision; applied identity is reported in the worker snapshot.
    uint64_t   m_resetRevision{ 0 }; ///< Last explicit reset request, acknowledged independently of parameter edits.
    std::mutex m_controlMutex, m_snapshotMutex; ///< Controls and snapshots; worker only uses try_lock.
    accelFFCtrlDetail::inputDiagnostics
        m_snapshot; ///< Last complete owner snapshot, delivered at ordinary app cadence.
    std::optional<accelFFCtrlDetail::inputProcessor>
                m_runtime; ///< Prepared before thread startup; worker-only thereafter.
    std::string m_inputName{ "acc000" }, m_inputPath, m_cpuset; ///< Startup-only source/path and worker CPU placement.
    int         m_priority{ 2 };   ///< Normal MagAO-X scheduling priority; use zero for unprivileged tests.
    unsigned    m_maxBatch{ 256 }; ///< Preallocated retained-frame copy bound per owner iteration.
    double      m_predictionFps{ 2000 }, m_telemetryFps{ 5 },
        m_clockToleranceUs{ 1000 };   ///< Startup prediction/report/clock policy.
    std::thread m_thread;             ///< Joined before runtime and property members are destroyed.
    bool        m_threadInit{ true }; ///< Passed to MagAOXApp threadStart; atomic m_run gates worker access instead.
    pid_t       m_threadId{ 0 };      ///< Framework scheduling identity, set at entry before the run barrier.
    std::atomic<bool> m_run{ false }, m_stop{ false }, m_failed{ false },
        m_invalidate{ false };                         ///< Lifecycle/probe handoff.
    std::atomic<uint64_t> m_inode{ 0 }, m_device{ 0 }; ///< Mapping identity for supervision; zero means disconnected.
    pcf::IndiProperty     m_indiP_thread, m_indiP_reset, m_indiP_diagnostics,
        m_indiP_input;                                    ///< Thread/reset/read-only status.
    std::array<pcf::IndiProperty, 11> m_indiP_parameters; ///< Standard current/target numeric controls.

    /// Define controls/status separately from starting the input worker, for tests.
    int setupProperties();

    /// Apply the newest coherent request without blocking the worker on callbacks.
    void applyRequested();

    /// Sample the local clock relation with a monotonic bracket; invalid reads return false.
    static bool readClock( accelFFCtrlDetail::inputClock &value /**< [out] midpoint clock sample */ );

    /// Framework thread entry, forwarding to the sole input/estimator owner.
    static void workerStart( accelFFCtrl *self /**< [in] application owner */ );

    /// Drain source records and predict on bounded monotonic waits; no INDI or routine logging.
    void worker();

  public:
    /// Construct without opening any stream.
    accelFFCtrl();

    /// Stop and join the input owner even when the application exits before normal shutdown.
    ~accelFFCtrl() noexcept override;

    /// Register startup input/numerical settings using normal MagAO-X configuration.
    void setupConfig() override;

    /// Load configuration and request shutdown on invalid settings.
    void loadConfig() override;

    /// Validate configuration without threads/stream attachment, for tests.
    int loadConfigImpl( mx::app::appConfigurator &configuration /**< [in] parsed configuration */ );

    /// Register INDI and start the existing-only non-actuating input worker.
    int appStartup() override;

    /// Probe backing identity and publish bounded-rate diagnostic snapshots.
    int appLogic() override;

    /// Stop and join without creating, modifying or zeroing any output stream.
    int appShutdown() override;

    /// Validate and enqueue one scalar control update, preserving other candidate fields.
    INDI_NEWCALLBACK_DECL( accelFFCtrl, m_indiP_parameters );

    /// Request a fit reset; observation automatically warms on subsequent source samples.
    INDI_NEWCALLBACK_DECL( accelFFCtrl, m_indiP_reset );
};

inline accelFFCtrl::accelFFCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_loopPause = 200000000;
}

inline accelFFCtrl::~accelFFCtrl() noexcept
{
    m_stop = true;
    m_run  = true;
    if( m_thread.joinable() )
        m_thread.join();
}

inline void accelFFCtrl::setupConfig()
{
    for( const char *key : accelFFCtrlDetail::parameterNames )
        config.add( std::string( "parameters." ) + key,
                    "",
                    std::string( "parameters." ) + key,
                    argType::Required,
                    "parameters",
                    key,
                    false,
                    "double",
                    "Numerical control; units match its INDI property." );
    const auto add = [this]( const std::string &key, const std::string &type, const std::string &description )
    { config.add( "input." + key, "", "input." + key, argType::Required, "input", key, false, type, description ); };
    add( "shmimName", "string", "Existing raw UINT16 pair stream; default acc000. DM-prefixed names are prohibited." );
    add( "threadPrio", "int", "Input/estimator worker priority, default 2." );
    add( "cpuset", "string", "Worker cpuset, empty by default." );
    add( "maxBatch", "unsigned", "Bounded startup copy capacity, default 256." );
    add(
        "sampleRate", "double", "Nominal ADC Hz for numerical initialization; actual intervals come from timestamps." );
    add( "predictionFps", "double", "Prediction query cadence, default 2000 Hz." );
    add( "telemetryFps", "double", "INDI/supervision cadence, default 5 Hz." );
    add( "clockTolerance_us", "double", "Maximum local clock bracket uncertainty/step, default 1000 us." );
    add( "maxAge_us", "double", "Maximum retained acquisition age, default 2000 us." );
    add( "maxInterval_us", "double", "Maximum consecutive acquisition interval, default 2000 us." );
    add( "maxPrediction_ms", "double", "Numerical forward prediction bound, default 50 ms." );
}

inline int accelFFCtrl::loadConfigImpl( mx::app::appConfigurator &configuration )
{
    using namespace accelFFCtrlDetail;
    configuration( m_inputName, "input.shmimName" );
    configuration( m_priority, "input.threadPrio" );
    configuration( m_cpuset, "input.cpuset" );
    configuration( m_maxBatch, "input.maxBatch" );
    configuration( m_requested.m_sampleRate, "input.sampleRate" );
    configuration( m_predictionFps, "input.predictionFps" );
    configuration( m_telemetryFps, "input.telemetryFps" );
    configuration( m_clockToleranceUs, "input.clockTolerance_us" );
    double age = m_requested.m_maxAgeNs * 1e-3, interval = m_requested.m_maxIntervalNs * 1e-3,
           maximum = m_requested.m_maxPredictionNs * 1e-6;
    configuration( age, "input.maxAge_us" );
    configuration( interval, "input.maxInterval_us" );
    configuration( maximum, "input.maxPrediction_ms" );
    try
    {
        if( !inputNameValid( m_inputName ) || m_priority < 0 || m_priority > 99 || m_maxBatch == 0 ||
            m_maxBatch > 65536 || !std::isfinite( m_predictionFps ) || m_predictionFps < 1 || m_predictionFps > 10000 ||
            !std::isfinite( m_telemetryFps ) || m_telemetryFps < .1 || m_telemetryFps > 20 ||
            !std::isfinite( m_clockToleranceUs ) || m_clockToleranceUs <= 0 || m_clockToleranceUs > 1e6 ||
            !std::isfinite( age ) || age <= 0 || age > 1e6 || !std::isfinite( interval ) || interval <= 0 ||
            interval > 1e6 || !std::isfinite( maximum ) || maximum <= 0 || maximum > 1000 )
            throw std::invalid_argument( "invalid input/cadence/time settings" );
        m_requested.m_maxAgeNs        = std::llround( age * 1e3 );
        m_requested.m_maxIntervalNs   = std::llround( interval * 1e3 );
        m_requested.m_maxPredictionNs = std::llround( maximum * 1e6 );
        for( size_t n = 0; n < parameterNames.size(); ++n )
        {
            const std::string key = std::string( "parameters." ) + parameterNames[n];
            if( !configuration.isSet( key ) )
                continue;
            std::string value;
            configuration( value, key );
            if( !parameterUpdate( m_requested, n, value ) )
                throw std::invalid_argument( "invalid numerical control" );
        }
        harmonicEstimator validate( m_requested );
        m_loopPause         = static_cast<unsigned long>( 1e9 / m_telemetryFps );
        m_snapshot.m_config = m_requested;
    }
    catch( const std::exception &error )
    {
        log<text_log>( std::string( "accelFFCtrl configuration: " ) + error.what(), logPrio::LOG_ERROR );
        return -1;
    }
    return 0;
}

inline void accelFFCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
        m_shutdown = true;
}

inline int accelFFCtrl::setupProperties()
{
    using namespace accelFFCtrlDetail;
    constexpr std::array<const char *, 11> labels{ "Frequency [Hz]",
                                                   "Memory [s]",
                                                   "Lever arm [m]",
                                                   "Rotation [deg]",
                                                   "Prediction horizon [ms]",
                                                   "ADC X zero [count]",
                                                   "ADC Y zero [count]",
                                                   "ADC X volts/count",
                                                   "ADC Y volts/count",
                                                   "ADC X sensitivity",
                                                   "ADC Y sensitivity" };
    for( size_t n = 0; n < parameterNames.size(); ++n )
    {
        auto        &property = m_indiP_parameters[n];
        const double minimum  = n == 3 || n == 5 || n == 6 ? -std::numeric_limits<double>::max() : 0;
        const double maximum  = n == 4 ? m_requested.m_maxPredictionNs * 1e-6 : std::numeric_limits<double>::max();
        if( createStandardIndiNumber<double>( property,
                                              parameterNames[n],
                                              minimum,
                                              maximum,
                                              0.,
                                              "%.9g",
                                              labels[n],
                                              n >= 5 ? "ADC calibration" : "Estimator" ) < 0 ||
            registerIndiPropertyNew( property, INDI_NEWCALLBACK( m_indiP_parameters ) ) < 0 )
            return -1;
        property["current"] = parameterValue( m_requested, n );
        property["target"]  = parameterValue( m_requested, n );
    }
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_reset, "estimator_reset" );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_diagnostics, "diagnostics", "Observation diagnostics", "Diagnostics" );
    for( const char *name : { "tip_x_mas",
                              "tilt_y_mas",
                              "amplitude_x_ms2",
                              "amplitude_y_ms2",
                              "phase_x_deg",
                              "phase_y_deg",
                              "ready",
                              "fit_state",
                              "stream_state",
                              "last_event",
                              "input_fps",
                              "processed",
                              "dropped",
                              "invalid_frames",
                              "restarts",
                              "resets",
                              "counter_rollbacks",
                              "empty_polls",
                              "busy_polls",
                              "copy_races",
                              "retention_frames",
                              "sample_age_us",
                              "effective_horizon_us",
                              "producer_delay_us",
                              "service_us",
                              "applied_revision",
                              "acquisition_fps" } )
        m_indiP_diagnostics.add( pcf::IndiElement( name ) );
    if( createROIndiText( m_indiP_input, "input", "stream", "ADC input", "Diagnostics" ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_input ) < 0 )
        return -1;
    for( const char *name : { "counter", "generation", "mode", "sample_time_ns", "target_time_ns" } )
        m_indiP_input.add( pcf::IndiElement( name ) );
    m_indiP_input["stream"] = m_inputName;
    m_indiP_input["mode"]   = "observation only";
    return 0;
}

inline int accelFFCtrl::appStartup()
{
    if( setupProperties() < 0 )
        return -1;
    char path[1024];
    if( ImageStreamIO_filename( path, sizeof( path ), m_inputName.c_str() ) != 0 )
        return -1;
    m_inputPath = path;
    m_runtime.emplace( m_requested, m_maxBatch, std::llround( m_clockToleranceUs * 1e3 ) );
    state( stateCodes::OPERATING );
    if( threadStart( m_thread,
                     m_threadInit,
                     m_threadId,
                     m_indiP_thread,
                     m_priority,
                     m_cpuset,
                     "accelFFCtrl-input",
                     this,
                     workerStart ) < 0 )
    {
        m_stop = true;
        m_run  = true;
        if( m_thread.joinable() )
            m_thread.join();
        return -1;
    }
    m_run.store( true, std::memory_order_release );
    return 0;
}

inline bool accelFFCtrl::readClock( accelFFCtrlDetail::inputClock &value )
{
    timespec first, real, last;
    if( clock_gettime( CLOCK_MONOTONIC, &first ) || clock_gettime( CLOCK_REALTIME, &real ) ||
        clock_gettime( CLOCK_MONOTONIC, &last ) )
        return false;
    int64_t a, b, r;
    if( !accelFFCtrlDetail::inputProcessor::timeNs( first, a ) ||
        !accelFFCtrlDetail::inputProcessor::timeNs( last, b ) ||
        !accelFFCtrlDetail::inputProcessor::timeNs( real, r ) || b < a )
        return false;
    value = { r, a + ( b - a ) / 2, ( b - a ) / 2 };
    return true;
}

inline void accelFFCtrl::applyRequested()
{
    accelFFCtrlDetail::parameters candidate;
    uint64_t                      revision;
    { //mutex scope
        std::unique_lock<std::mutex> guard( m_controlMutex, std::try_to_lock );
        if( !guard || m_runtime->diagnostics().m_revision == m_revision )
            return;
        candidate = m_requested;
        revision  = m_revision;
    }
    m_runtime->configure( candidate, revision );
}

inline void accelFFCtrl::workerStart( accelFFCtrl *self )
{
    self->worker();
}

inline void accelFFCtrl::worker()
{
#ifdef __linux__
    m_threadId = syscall( SYS_gettid );
#else
    m_threadId = getpid();
#endif
    while( !m_run.load( std::memory_order_acquire ) && !m_stop )
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    using namespace accelFFCtrlDetail;
    try
    {
        adcStream     source;
        bool          connected = false;
        int64_t       retry = 0, nextPrediction = 0;
        const int64_t period = std::llround( 1e9 / m_predictionFps );
        while( !m_stop )
        {
            inputClock clock;
            if( !readClock( clock ) )
            {
                m_failed = true;
                break;
            }
            const int64_t begin = clock.m_mono;
            applyRequested();
            if( m_invalidate.exchange( false ) )
            {
                m_inode = 0;
                source.close();
                connected = false;
                m_runtime->disconnected();
                retry = 0;
            }
            if( !connected && clock.m_mono >= retry )
            {
                connected = source.open( m_inputName );
                retry     = clock.m_mono + 1000000000;
                if( connected )
                {
                    m_device = source.device();
                    m_inode  = source.inode();
                    m_runtime->attached();
                }
            }
            if( connected )
                m_runtime->drain( source,
                                  []
                                  {
                                      inputClock received;
                                      readClock( received );
                                      return received;
                                  } );
            if( !readClock( clock ) )
            {
                m_failed = true;
                break;
            }
            if( clock.m_mono >= nextPrediction )
            {
                m_runtime->predict( clock.m_mono );
                inputClock completed;
                if( !readClock( completed ) )
                {
                    m_failed = true;
                    break;
                }
                auto snapshot        = m_runtime->diagnostics();
                snapshot.m_serviceUs = ( completed.m_mono - begin ) * 1e-3;
                { //mutex scope
                    std::unique_lock<std::mutex> guard( m_snapshotMutex, std::try_to_lock );
                    if( guard )
                        m_snapshot = snapshot;
                }
                nextPrediction = nextPrediction
                                     ? nextPrediction + ( ( clock.m_mono - nextPrediction ) / period + 1 ) * period
                                     : clock.m_mono + period;
            }
            if( connected )
            {
                if( !source.wait( nextPrediction ) )
                    m_invalidate = true;
            }
            else
                std::this_thread::sleep_for(
                    std::chrono::nanoseconds( std::max<int64_t>( 1, nextPrediction - clock.m_mono ) ) );
        }
        m_inode = 0;
        m_runtime->disconnected();
    }
    catch( const std::exception & )
    {
        m_inode  = 0;
        m_failed = true;
    }
}

inline int accelFFCtrl::appLogic()
{
    using namespace accelFFCtrlDetail;
    if( m_failed )
        return log<software_error, -1>( { __FILE__, __LINE__, "input worker failed" } );
    const uint64_t inode = m_inode.load();
    if( inode )
    {
        struct stat current;
        if( stat( m_inputPath.c_str(), &current ) != 0 || uint64_t( current.st_ino ) != inode ||
            uint64_t( current.st_dev ) != m_device.load() )
            m_invalidate = true;
    }
    inputDiagnostics snapshot;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_snapshotMutex );
        snapshot = m_snapshot;
    }
    inputClock clock;
    if( readClock( clock ) && snapshot.m_prediction.m_state == estimatorState::ready &&
        clock.m_mono - snapshot.m_prediction.m_sampleTimeNs > snapshot.m_config.m_maxAgeNs )
    {
        snapshot.m_ageUs              = ( clock.m_mono - snapshot.m_prediction.m_sampleTimeNs ) * 1e-3;
        snapshot.m_horizonUs          = 0;
        snapshot.m_prediction         = {};
        snapshot.m_prediction.m_state = estimatorState::stale;
        snapshot.m_state              = inputState::stale;
    }
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const bool                  pending = snapshot.m_revision != m_revision;
    for( size_t n = 0; n < parameterNames.size(); ++n )
    {
        auto &property      = m_indiP_parameters[n];
        property["current"] = parameterValue( snapshot.m_config, n );
        property["target"]  = parameterValue( m_requested, n );
        property.setState( property.getState() == INDI_ALERT ? INDI_ALERT : pending ? INDI_BUSY : INDI_OK );
        if( m_indiDriver )
            m_indiDriver->sendSetProperty( property );
    }
    const bool resetPending  = snapshot.m_revision < m_resetRevision;
    m_indiP_reset["request"] = resetPending ? pcf::IndiElement::On : pcf::IndiElement::Off;
    m_indiP_reset.setState( resetPending ? INDI_BUSY : INDI_OK );
    constexpr double                       toMas = 206264806.24709636, toDegrees = 57.29577951308232;
    const auto                            &p = snapshot.m_prediction;
    const std::array<double, 27>           values{ p.m_pointing[0] * toMas,
                                                   p.m_pointing[1] * toMas,
                                                   p.m_amplitude[0],
                                                   p.m_amplitude[1],
                                                   p.m_phase[0] * toDegrees,
                                                   p.m_phase[1] * toDegrees,
                                                   double( p.m_state == estimatorState::ready ),
                                                   double( p.m_state ),
                                                   double( snapshot.m_state ),
                                                   double( snapshot.m_lastEvent ),
                                                   snapshot.m_rate,
                                                   double( snapshot.m_processed ),
                                                   double( snapshot.m_dropped ),
                                                   double( snapshot.m_invalid ),
                                                   double( snapshot.m_restarts ),
                                                   double( snapshot.m_resets ),
                                                   double( snapshot.m_rollbacks ),
                                                   double( snapshot.m_duplicates ),
                                                   double( snapshot.m_busy ),
                                                   double( snapshot.m_races ),
                                                   double( snapshot.m_depth ),
                                                   snapshot.m_ageUs,
                                                   snapshot.m_horizonUs,
                                                   snapshot.m_writeDelayUs,
                                                   snapshot.m_serviceUs,
                                                   double( snapshot.m_revision ),
                                                   snapshot.m_sampleRate };
    constexpr std::array<const char *, 27> names{ "tip_x_mas",
                                                  "tilt_y_mas",
                                                  "amplitude_x_ms2",
                                                  "amplitude_y_ms2",
                                                  "phase_x_deg",
                                                  "phase_y_deg",
                                                  "ready",
                                                  "fit_state",
                                                  "stream_state",
                                                  "last_event",
                                                  "input_fps",
                                                  "processed",
                                                  "dropped",
                                                  "invalid_frames",
                                                  "restarts",
                                                  "resets",
                                                  "counter_rollbacks",
                                                  "empty_polls",
                                                  "busy_polls",
                                                  "copy_races",
                                                  "retention_frames",
                                                  "sample_age_us",
                                                  "effective_horizon_us",
                                                  "producer_delay_us",
                                                  "service_us",
                                                  "applied_revision",
                                                  "acquisition_fps" };
    for( size_t n = 0; n < names.size(); ++n )
        m_indiP_diagnostics[names[n]] = values[n];
    const bool fault    = snapshot.m_state != inputState::streaming && snapshot.m_state != inputState::waiting &&
                          snapshot.m_state != inputState::disconnected;
    const bool fitFault = p.m_state != estimatorState::warming && p.m_state != estimatorState::ready;
    m_indiP_diagnostics.setState( fault || fitFault                    ? INDI_ALERT
                                  : p.m_state == estimatorState::ready ? INDI_OK
                                                                       : INDI_BUSY );
    m_indiP_input["counter"]        = std::to_string( snapshot.m_counter );
    m_indiP_input["generation"]     = std::to_string( snapshot.m_generation );
    m_indiP_input["sample_time_ns"] = std::to_string( p.m_sampleTimeNs );
    m_indiP_input["target_time_ns"] = std::to_string( p.m_targetTimeNs );
    if( m_indiDriver )
    {
        m_indiDriver->sendSetProperty( m_indiP_reset );
        m_indiDriver->sendSetProperty( m_indiP_diagnostics );
        m_indiDriver->sendSetProperty( m_indiP_input );
    }
    return 0;
}

inline int accelFFCtrl::appShutdown()
{
    m_stop = true;
    m_run  = true;
    if( m_thread.joinable() )
        m_thread.join();
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( accelFFCtrl, m_indiP_parameters )( const pcf::IndiProperty &ipRecv )
{
    using namespace accelFFCtrlDetail;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( ipRecv.getDevice() != m_configName )
        return -1;
    for( size_t n = 0; n < parameterNames.size(); ++n )
    {
        auto &property = m_indiP_parameters[n];
        if( ipRecv.getName() != property.getName() )
            continue;
        const std::string text = ipRecv.find( "target" )    ? ipRecv["target"].get()
                                 : ipRecv.find( "current" ) ? ipRecv["current"].get()
                                                            : "";
        if( !parameterUpdate( m_requested, n, text ) )
        {
            property.setState( INDI_ALERT );
            if( m_indiDriver )
                m_indiDriver->sendSetProperty( property );
            return -1;
        }
        ++m_revision;
        property["target"] = parameterValue( m_requested, n );
        property.setState( INDI_BUSY );
        return 0;
    }
    return -1;
}

inline INDI_NEWCALLBACK_DEFN( accelFFCtrl, m_indiP_reset )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reset, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( !ipRecv.find( "request" ) )
        return -1;
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        ++m_revision;
        m_resetRevision          = m_revision;
        m_indiP_reset["request"] = pcf::IndiElement::On;
        m_indiP_reset.setState( INDI_BUSY );
    }
    return 0;
}

} // namespace app
} // namespace MagAOX
#endif
