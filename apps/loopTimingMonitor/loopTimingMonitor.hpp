/** \file loopTimingMonitor.hpp
 * \brief Semaphore-free, operating-state-qualified camera-to-DM timing monitor.
 */
#ifndef loopTimingMonitor_hpp
#define loopTimingMonitor_hpp

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "../../libMagAOX/common/publicationClock.hpp"
#include "../../libMagAOX/common/spscQueue.hpp"
#include "../../libMagAOX/logger/types/telem_fgtimings.hpp"
#include "loopTiming.hpp"
#include "loopTimingGate.hpp"
#include "loopTimingStream.hpp"

namespace MagAOX
{
namespace app
{

/** \defgroup loopTimingMonitor Loop Timing Monitor
 * \brief Observed camera acquisition to DM SDK-entry latency.
 * \ingroup apps
 */

/// Observe software endpoint timestamps only while all configured devices permit collection.
/** \ingroup loopTimingMonitor
 * No command is sent to the camera, DM or loop. Timestamp association assumes
 * one successful DM submission per camera frame and latency below one period.
 */
class loopTimingMonitor : public MagAOXApp<>, public dev::telemeter<loopTimingMonitor>
{
    friend class dev::telemeter<loopTimingMonitor>;

  protected:
    /// Standard persistent telemetry lifecycle.
    typedef dev::telemeter<loopTimingMonitor> telemeterT;

    /** \name Configuration - Data
     * @{ */
    std::string m_cameraStream{ "camwfs" }; ///< Camera timestamp source, independent of its INDI device name.

    std::string m_sdkStream{ "dmtweeter_sdk_start" }; ///< Post-processing SDK-start timestamp source.

    std::string m_cameraDevice{ "camwfs" }; ///< Owner of the camera fsm.state subscription.

    std::string m_dmDevice{ "dmtweeter" }; ///< Owner of the DM fsm.state subscription.

    std::string m_loopDevice{ "holoop" }; ///< Owner of the loop_state.toggle subscription.

    std::string m_cameraClock{ "realtime" }; ///< Explicit external clock used by camera atime.

    double m_maxFPS{ 3600 }; ///< Declared fastest frame rate; its reciprocal limits accepted latency.

    double m_pollSleepUs{ 10 }; ///< Requested delay after each scan; zero explicitly selects busy polling.

    double m_windowSec{ 2 }; ///< Trailing event-time statistics window.

    size_t m_capacity{ 32768 }; ///< Maximum retained samples and unread observation records.

    double m_clockBracketUs{ 1 }; ///< Maximum MONOTONIC/external/MONOTONIC capture width.

    double m_clockMaxAgeMs{ 100 }; ///< Maximum camera timestamp age at clock mapping.

    double m_cameraQuantizationUs{ 1 }; ///< Camera timestamp truncation uncertainty, ordinary EDT default.

    int m_threadPrio{ 0 }; ///< Optional polling-worker scheduling priority.

    std::string m_cpuset; ///< Optional polling-worker CPU set.
    ///@}

    /// One immutable observation; generations prevent joining data across invalidation.
    struct Observation
    {
        bool     camera{ false }; ///< Select the camera timestamp source, otherwise SDK entry.
        uint64_t gate{ 0 };       ///< Operating-state generation at capture.
        uint64_t epoch{ 0 };      ///< Stream/clock/queue continuity generation at capture.
        uint64_t counter{ 0 };    ///< Source publication counter.
        int64_t  stampNs{ 0 };    ///< Endpoint in the host MONOTONIC clock.
        int64_t  lowerNs{ 0 };    ///< Earliest camera endpoint allowed by clock uncertainty.
        int64_t  upperNs{ 0 };    ///< Latest camera endpoint including truncation uncertainty.
        int64_t  rawNs{ 0 };      ///< Original camera clock stamp for cadence without mapping noise.
    };

    std::mutex m_gateMutex; ///< Serializes INDI state updates and gate publication.

    loopTiming::Gate m_gate; ///< Last received source states, protected by m_gateMutex.

    std::atomic<uint64_t> m_gateGeneration{ 0 }; ///< Published gate generation for the polling worker.

    std::atomic<bool> m_gateEnabled{ false }; ///< Published operating gate for the polling worker.

    std::atomic<uint64_t> m_captureEpoch{ 1 }; ///< Polling-owner continuity generation.

    std::atomic<uint64_t> m_invalidSnapshots{ 0 }; ///< Invalid new metadata observations.

    std::atomic<uint64_t> m_clockRejected{ 0 }; ///< Camera clock mappings rejected before handoff.

    std::atomic<uint64_t> m_cameraMissed{ 0 }; ///< Publications overwritten before camera polling.

    std::atomic<uint64_t> m_sdkMissed{ 0 }; ///< Publications overwritten before SDK polling.

    std::atomic<uint64_t> m_counterRegressions{ 0 }; ///< Source counter resets/wraps detected before handoff.

    std::atomic<bool> m_streamsReady{ false }; ///< Both expected stream incarnations are attached.

    std::atomic<bool> m_stop{ false }; ///< Cooperative worker shutdown request.

    std::atomic<bool> m_workerReady{ false }; ///< Scheduler/affinity initialization completed.

    std::atomic<bool> m_workerFailed{ false }; ///< An unexpected polling exception requires application shutdown.

    spscQueue<Observation> m_queue; ///< Fixed-capacity handoff from polling to ordinary app logic.

    loopTiming::Monitor m_monitor; ///< Consumer-owned pairing and statistics.

    uint64_t m_summaryGate{ 0 }; ///< Gate generation represented by retained statistics.

    uint64_t m_summaryEpoch{ 0 }; ///< Capture generation represented by retained statistics.

    uint64_t m_summarySerial{ 0 }; ///< Ordinary-thread statistics publication sequence.

    uint64_t m_lastRecord{ 0 }; ///< Summary sequence last written to fgtimes.

    std::thread m_thread; ///< Sole owner of both attached streams and clock mapper.

    bool m_threadInit{ true }; ///< Standard threadStart initialization flag.

    pid_t m_threadID{ 0 }; ///< Polling-worker kernel thread ID.

    pcf::IndiProperty m_indiP_thread; ///< Standard polling thread scheduling report.

    pcf::IndiProperty m_indiP_camera; ///< Camera FSM subscription storage.

    pcf::IndiProperty m_indiP_dm; ///< DM FSM subscription storage.

    pcf::IndiProperty m_indiP_loop; ///< Loop toggle subscription storage.

    pcf::IndiProperty m_indiP_timing; ///< Read-only mean/deviation, count and freshness.

    pcf::IndiProperty m_indiP_quality; ///< Read-only observation coverage and generation counters.

    pcf::IndiProperty m_indiP_status; ///< Read-only availability and endpoint/clock descriptions.

  public:
    /// Construct a passive monitor without power control.
    loopTimingMonitor();

    /// Stop the polling owner before destroying its shared state.
    ~loopTimingMonitor() noexcept;

    /// Register stream, clock, gate, polling and telemetry options.
    void setupConfig() override;

    /// Validate immutable startup configuration.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] Configured startup values. */ );

    /// Load configuration and request shutdown on rejection.
    void loadConfig() override;

    /// Register subscriptions/output properties and start the polling owner.
    int appStartup() override;

    /// Consume observations and publish statistics/telemetry under the subscribed source states.
    int appLogic() override;

    /// Stop polling and the telemetry helper without touching source devices.
    int appShutdown() override;

    /// Route a received read-only source property to this instance.
    static int st_setCallBack_source( void                    *app /**< [in] Owning monitor. */,
                                      const pcf::IndiProperty &property /**< [in] Observed source property. */ );

    /// Update the appropriate device gate from an exact FSM or toggle property.
    int setCallBack_source( const pcf::IndiProperty &property /**< [in] Observed source property. */ );

    /// Request forced telemetry when its standard maximum record interval expires.
    int checkRecordTimes();

    /// Record the current valid gated summary when requested by telemeter.
    int recordTelem( const logger::telem_fgtimings *telem /**< [in] Telemetry overload selector. */ );

  protected:
    /// Read the host monotonic clock, returning zero on failure.
    static int64_t monotonicNow();

    /// Publish the subscribed gate state and generation while holding m_gateMutex.
    bool publishGate();

    /// Consume at most one captured queue population and update its trailing statistics.
    void consume( int64_t nowNs /**< [in] Current monotonic evaluation time. */ );

    /// Copy all six fgtimes values only when current evidence still qualifies the summary.
    bool timingValues( std::array<double, 6> &values /**< [out] Cadence/cadence/latency mean-deviation pairs. */ );

    /// Qualify and copy telemetry values while the caller holds m_gateMutex.
    bool timingValuesLocked( std::array<double, 6> &values /**< [out] Current fgtimes populations. */ );

    /// Write one normal or forced fgtimes record from a qualified summary.
    int recordTimings( bool force = false /**< [in] Allow re-emitting the current valid summary. */ );

    /// Standard worker entrypoint.
    static void threadStart( loopTimingMonitor *app /**< [in] Owning application. */ );

    /// Poll both timestamp streams without semaphore synchronization.
    void threadExec() noexcept;

    /// Sleep in bounded slices so configured long pauses do not delay shutdown.
    void sleepFor( int64_t ns /**< [in] Requested relative sleep, zero means no sleep. */ );

    /// Stop and join the worker; safe before startup and after shutdown.
    void stopWorker();
};

inline loopTimingMonitor::loopTimingMonitor() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = false;
}

inline loopTimingMonitor::~loopTimingMonitor() noexcept
{
    stopWorker();
}

inline void loopTimingMonitor::setupConfig()
{
    auto add = [this]( const std::string &key,
                       const std::string &section,
                       const std::string &name,
                       const std::string &type,
                       const std::string &description )
    {
        config.add( key, "", key, argType::Required, section, name, false, type, description );
    };

    add( "camera.shmimName", "camera", "shmimName", "string", "Camera timestamp stream; default camwfs." );
    add( "camera.device", "camera", "device", "string", "Camera FSM device; default camwfs." );
    add( "camera.clock", "camera", "clock", "string", "Camera atime clock: realtime or tai." );
    add( "camera.quantizationUs",
         "camera",
         "quantizationUs",
         "real",
         "Timestamp truncation uncertainty; default 1 us." );
    add( "dm.shmimName", "dm", "shmimName", "string", "Post-processing SDK timestamp stream." );
    add( "dm.device", "dm", "device", "string", "DM FSM device; default dmtweeter." );
    add( "loop.device", "loop", "device", "string", "Device with loop_state.toggle; default holoop." );
    add( "monitor.maxFPS",
         "monitor",
         "maxFPS",
         "real",
         "Fastest frame rate; default 3600 Hz; reciprocal bounds latency." );
    add( "monitor.pollSleepUs",
         "monitor",
         "pollSleepUs",
         "real",
         "Requested poll sleep; default 10 us, zero busy polls." );
    add( "monitor.windowSec", "monitor", "windowSec", "real", "Trailing statistics window; default 2 seconds." );
    add( "monitor.capacity", "monitor", "capacity", "integer", "Bounded observation/sample capacity; default 32768." );
    add( "monitor.clockBracketUs", "monitor", "clockBracketUs", "real", "Maximum clock-read bracket; default 1 us." );
    add( "monitor.clockMaxAgeMs", "monitor", "clockMaxAgeMs", "real", "Maximum mapped timestamp age; default 100 ms." );
    add( "monitor.threadPrio", "monitor", "threadPrio", "integer", "Polling thread priority 0-99; default 0." );
    add( "monitor.cpuset", "monitor", "cpuset", "string", "Optional polling thread CPU set." );
    TELEMETER_SETUP_CONFIG( config );
}

inline int loopTimingMonitor::loadConfigImpl( mx::app::appConfigurator &config )
{
    config( m_cameraStream, "camera.shmimName" );
    config( m_cameraDevice, "camera.device" );
    config( m_cameraClock, "camera.clock" );
    config( m_cameraQuantizationUs, "camera.quantizationUs" );
    config( m_sdkStream, "dm.shmimName" );
    config( m_dmDevice, "dm.device" );
    config( m_loopDevice, "loop.device" );
    config( m_maxFPS, "monitor.maxFPS" );
    config( m_pollSleepUs, "monitor.pollSleepUs" );
    config( m_windowSec, "monitor.windowSec" );
    config( m_capacity, "monitor.capacity" );
    config( m_clockBracketUs, "monitor.clockBracketUs" );
    config( m_clockMaxAgeMs, "monitor.clockMaxAgeMs" );
    config( m_threadPrio, "monitor.threadPrio" );
    config( m_cpuset, "monitor.cpuset" );

    const auto bounded = []( double value, double low, double high )
    {
        return mx::math::isFinite( value ) && value >= low && value <= high;
    };

    if( m_cameraStream.empty() || m_sdkStream.empty() || m_cameraStream == m_sdkStream || m_cameraDevice.empty() ||
        m_dmDevice.empty() || m_loopDevice.empty() || m_cameraDevice == m_dmDevice ||
        ( m_cameraClock != "realtime" && m_cameraClock != "tai" ) || !bounded( m_maxFPS, 1, 1000000 ) ||
        !bounded( m_pollSleepUs, 0, 1000000 ) || !bounded( m_windowSec, 0.001, 3600 ) || m_capacity < 4 ||
        m_capacity > 10000000 || !bounded( m_clockBracketUs, 0.001, 1000000 ) ||
        !bounded( m_clockMaxAgeMs, 0.001, 1000 ) || !bounded( m_cameraQuantizationUs, 0, 1000 ) || m_threadPrio < 0 ||
        m_threadPrio > 99 )
    {
        return log<software_error, -1>( { "invalid loop timing configuration" } );
    }

#ifndef CLOCK_TAI
    if( m_cameraClock == "tai" )
    {
        return log<software_error, -1>( { "CLOCK_TAI is unavailable" } );
    }
#endif

    TELEMETER_LOAD_CONFIG( config );
    return 0;
}

inline void loopTimingMonitor::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

inline int loopTimingMonitor::appStartup()
{
    if( !m_monitor.configure( m_maxFPS, m_windowSec, m_capacity ) )
    {
        return -1;
    }

    m_queue.configure( m_capacity );

    if( registerIndiPropertySet( m_indiP_camera, m_cameraDevice, "fsm", st_setCallBack_source ) < 0 ||
        registerIndiPropertySet( m_indiP_dm, m_dmDevice, "fsm", st_setCallBack_source ) < 0 ||
        registerIndiPropertySet( m_indiP_loop, m_loopDevice, "loop_state", st_setCallBack_source ) < 0 )
    {
        return -1;
    }

    createROIndiNumber( m_indiP_timing, "loop_timing" );

    for( const auto *name :
         { "mean", "stddev", "count", "cameraMean", "cameraStddev", "sdkMean", "sdkStddev", "span", "age", "valid" } )
    {
        indi::addNumberElement( m_indiP_timing, name, -1.0, std::numeric_limits<double>::max(), 0.0, "%g" );
    }

    createROIndiNumber( m_indiP_quality, "loop_timing_quality" );

    for( const auto *name : { "cameraMissed",
                              "sdkMissed",
                              "invalid",
                              "ambiguous",
                              "unmatched",
                              "queueDropped",
                              "clockRejected",
                              "counterRegressions",
                              "overflow",
                              "epoch",
                              "gateGeneration" } )
    {
        indi::addNumberElement( m_indiP_quality, name, 0.0, std::numeric_limits<double>::max(), 0.0, "%g" );
    }

    createROIndiText( m_indiP_status, "loop_timing_state", "state" );
    indi::addTextElement( m_indiP_status, "endpoints" );
    indi::addTextElement( m_indiP_status, "cameraClock" );
    m_indiP_status["state"]       = "waiting";
    m_indiP_status["endpoints"]   = m_sdkStream + ".atime - " + m_cameraStream + ".atime (clock mapped)";
    m_indiP_status["cameraClock"] = m_cameraClock;

    if( registerIndiPropertyReadOnly( m_indiP_timing ) < 0 || registerIndiPropertyReadOnly( m_indiP_quality ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_status ) < 0 )
    {
        return -1;
    }

    TELEMETER_APP_STARTUP;
    m_stop         = false;
    m_workerReady  = false;
    m_workerFailed = false;

    if( MagAOXApp<>::threadStart( m_thread,
                                  m_threadInit,
                                  m_threadID,
                                  m_indiP_thread,
                                  m_threadPrio,
                                  m_cpuset,
                                  "loopTiming",
                                  this,
                                  threadStart ) < 0 )
    {
        stopWorker();
        return -1;
    }

    int         policy = -1;
    sched_param scheduling{};
    const auto  result = pthread_getschedparam( m_thread.native_handle(), &policy, &scheduling );

    if( result != 0 || scheduling.sched_priority != m_threadPrio ||
        policy != ( m_threadPrio > 0 ? MAGAOX_RT_SCHED_POLICY : SCHED_OTHER ) )
    {
        stopWorker();
        return log<software_error, -1>( { "loop timing worker did not acquire the requested scheduling policy" } );
    }

    m_workerReady.store( true, std::memory_order_release );
    state( stateCodes::READY );
    return 0;
}

inline int64_t loopTimingMonitor::monotonicNow()
{
    timespec stamp{};
    return clock_gettime( CLOCK_MONOTONIC, &stamp ) == 0 ? int64_t( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec : 0;
}

inline bool loopTimingMonitor::publishGate()
{
    const bool enabled = m_gate.enabled();
    m_gateEnabled.store( enabled, std::memory_order_release );
    m_gateGeneration.store( m_gate.generation(), std::memory_order_release );
    return enabled;
}

inline int loopTimingMonitor::st_setCallBack_source( void *app, const pcf::IndiProperty &property )
{
    return static_cast<loopTimingMonitor *>( app )->setCallBack_source( property );
}

inline int loopTimingMonitor::setCallBack_source( const pcf::IndiProperty &property )
{
    loopTiming::Gate::Source source;
    bool                     value = false;

    if( property.getName() == "fsm" &&
        ( property.getDevice() == m_cameraDevice || property.getDevice() == m_dmDevice ) )
    {
        source =
            property.getDevice() == m_cameraDevice ? loopTiming::Gate::Source::camera : loopTiming::Gate::Source::dm;
        value = property.getType() == pcf::IndiProperty::Text && property.find( "state" ) &&
                property["state"].get<std::string>() == "OPERATING";
    }
    else if( property.getName() == "loop_state" && property.getDevice() == m_loopDevice )
    {
        source = loopTiming::Gate::Source::loop;
        value  = property.getType() == pcf::IndiProperty::Switch && property.find( "toggle" ) &&
                property["toggle"].getSwitchState() == pcf::IndiElement::On;
    }
    else
    {
        return -1;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_gateMutex );
        m_gate.update( source, value );
        publishGate();
    }

    return 0;
}

inline void loopTimingMonitor::consume( int64_t nowNs )
{
    uint64_t generation;
    bool     enabled;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_gateMutex );
        enabled    = publishGate();
        generation = m_gate.generation();
    }

    const auto epoch = m_captureEpoch.load( std::memory_order_acquire );

    if( !enabled || generation != m_summaryGate || epoch != m_summaryEpoch )
    {
        m_monitor.reset();
        m_summaryGate  = generation;
        m_summaryEpoch = epoch;
    }

    Observation observation;
    const auto  queued = m_queue.queued();

    for( size_t index = 0; index < queued && m_queue.pop( observation ); ++index )
    {
        if( !enabled || observation.gate != generation || observation.epoch != epoch )
        {
            continue;
        }

        if( observation.camera )
        {
            m_monitor.camera(
                observation.counter, observation.stampNs, observation.lowerNs, observation.upperNs, observation.rawNs );
        }
        else
        {
            m_monitor.sdk( observation.counter, observation.stampNs );
        }
    }

    m_monitor.update( nowNs );
    ++m_summarySerial;
}

inline bool loopTimingMonitor::timingValues( std::array<double, 6> &values )
{
    std::lock_guard<std::mutex> lock( m_gateMutex );
    return timingValuesLocked( values );
}

inline bool loopTimingMonitor::timingValuesLocked( std::array<double, 6> &values )
{
    const auto  now     = monotonicNow();
    const auto &summary = m_monitor.summary();

    if( !publishGate() || m_summaryGate != m_gate.generation() ||
        m_summaryEpoch != m_captureEpoch.load( std::memory_order_acquire ) || !m_streamsReady.load() ||
        !summary.cameraIntervals.valid || !summary.sdkIntervals.valid || !summary.latency.valid ||
        summary.newestNs <= 0 || now < summary.newestNs || now - summary.newestNs >= int64_t( m_windowSec * 1e9 ) )
    {
        return false;
    }

    values = { summary.cameraIntervals.mean,
               summary.cameraIntervals.stddev,
               summary.sdkIntervals.mean,
               summary.sdkIntervals.stddev,
               summary.latency.mean,
               summary.latency.stddev };
    return true;
}

inline int loopTimingMonitor::appLogic()
{
    if( m_workerFailed.load() )
    {
        return -1;
    }

    consume( monotonicNow() );
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_gateMutex );
        const auto                 &summary = m_monitor.summary();
        std::array<double, 6>       values{};
        const bool                  enabled   = publishGate();
        const bool                  connected = m_streamsReady.load();
        const bool                  active    = enabled && connected;
        const bool current = active && m_summaryGate == m_gate.generation() && m_summaryEpoch == m_captureEpoch.load();
        const bool valid   = current && timingValuesLocked( values );
        const std::string availability = !enabled                                ? "gated"
                                         : !connected                            ? "disconnected"
                                         : valid                                 ? "valid"
                                         : current && summary.age >= m_windowSec ? "stale"
                                                                                 : "warming";
        state( active ? stateCodes::OPERATING : stateCodes::READY );
        updateIfChanged( m_indiP_status, "state", availability );
        updateIfChanged( m_indiP_timing,
                         std::vector<std::string>{ "mean",
                                                   "stddev",
                                                   "count",
                                                   "cameraMean",
                                                   "cameraStddev",
                                                   "sdkMean",
                                                   "sdkStddev",
                                                   "span",
                                                   "age",
                                                   "valid" },
                         std::vector<double>{ current && summary.latency.count ? summary.latency.mean : -1,
                                              valid ? values[5] : -1,
                                              current ? double( summary.latency.count ) : 0,
                                              valid ? values[0] : -1,
                                              valid ? values[1] : -1,
                                              valid ? values[2] : -1,
                                              valid ? values[3] : -1,
                                              current ? summary.latency.span : 0,
                                              current ? summary.age : -1,
                                              double( valid ) } );
        const auto &quality = summary.quality;
        updateIfChanged( m_indiP_quality,
                         std::vector<std::string>{ "cameraMissed",
                                                   "sdkMissed",
                                                   "invalid",
                                                   "ambiguous",
                                                   "unmatched",
                                                   "queueDropped",
                                                   "clockRejected",
                                                   "counterRegressions",
                                                   "overflow",
                                                   "epoch",
                                                   "gateGeneration" },
                         std::vector<double>{ double( m_cameraMissed.load() ),
                                              double( m_sdkMissed.load() ),
                                              double( quality.invalid + m_invalidSnapshots.load() ),
                                              double( quality.ambiguous ),
                                              double( quality.unmatched ),
                                              double( m_queue.dropped() ),
                                              double( m_clockRejected.load() ),
                                              double( quality.counterRegressions + m_counterRegressions.load() ),
                                              double( quality.overflow ),
                                              double( m_summaryEpoch ),
                                              double( m_summaryGate ) } );
    }

    recordTimings();
    TELEMETER_APP_LOGIC;
    return 0;
}

inline int loopTimingMonitor::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( logger::telem_fgtimings() );
}

inline int loopTimingMonitor::recordTelem( const logger::telem_fgtimings * )
{
    return recordTimings( true );
}

inline int loopTimingMonitor::recordTimings( bool force )
{
    std::lock_guard<std::mutex> lock( m_gateMutex );
    std::array<double, 6>       values{};

    if( ( force || m_lastRecord != m_summarySerial ) && timingValuesLocked( values ) )
    {
        telem<logger::telem_fgtimings>( { values[0], values[1], values[2], values[3], values[4], values[5] } );
        m_lastRecord = m_summarySerial;
    }

    return 0;
}

inline void loopTimingMonitor::threadStart( loopTimingMonitor *app )
{
    app->m_threadID = syscall( SYS_gettid );
    app->threadExec();
}

inline void loopTimingMonitor::sleepFor( int64_t ns )
{
    while( ns > 0 && !m_stop.load( std::memory_order_acquire ) )
    {
        const auto slice = std::min<int64_t>( ns, 10000000 );
        timespec   delay{ 0, static_cast<long>( slice ) };
        nanosleep( &delay, nullptr );
        ns -= slice;
    }
}

inline void loopTimingMonitor::threadExec() noexcept
{
    loopTiming::Stream       camera;
    loopTiming::Stream       sdk;
    timing::PublicationClock clock;
    clock.configure( int64_t( m_clockBracketUs * 1000 ), 1000, int64_t( m_clockMaxAgeMs * 1000000 ) );
    clockid_t externalClock = CLOCK_REALTIME;

#ifdef CLOCK_TAI
    if( m_cameraClock == "tai" )
    {
        externalClock = CLOCK_TAI;
    }
#endif

    uint64_t generation = 0;
    int64_t  nextAttach = 0;
    bool     wasEnabled = false;

    try
    {
        while( !m_stop.load( std::memory_order_acquire ) )
        {
            if( !m_workerReady.load( std::memory_order_acquire ) )
            {
                sleepFor( 1000000 );
                continue;
            }

            const auto currentGeneration = m_gateGeneration.load( std::memory_order_acquire );
            const bool enabled           = m_gateEnabled.load( std::memory_order_acquire );
            const auto now               = monotonicNow();

            if( currentGeneration != generation || enabled != wasEnabled )
            {
                camera.close();
                sdk.close();
                clock.reset();
                m_captureEpoch.fetch_add( 1, std::memory_order_release );
                m_streamsReady = false;
                generation     = currentGeneration;
                wasEnabled     = enabled;
                nextAttach     = 0;
            }

            if( !enabled )
            {
                sleepFor( 10000000 );
                continue;
            }

            if( now >= nextAttach )
            {
                const bool replaced =
                    ( camera.isOpen() && !camera.checkIdentity() ) || ( sdk.isOpen() && !sdk.checkIdentity() );

                if( replaced )
                {
                    camera.close();
                    sdk.close();
                    clock.reset();
                    m_captureEpoch.fetch_add( 1, std::memory_order_release );
                }

                if( !camera.isOpen() )
                {
                    camera.open( m_cameraStream );
                }

                if( !sdk.isOpen() )
                {
                    sdk.open( m_sdkStream, true );
                }

                m_streamsReady = camera.isOpen() && sdk.isOpen();
                nextAttach     = now + 100000000;
            }

            if( !m_streamsReady.load() )
            {
                sleepFor( 10000000 );
                continue;
            }

            bool reset = false;

            for( int source = 0; source < 2; ++source )
            {
                loopTiming::Stream::Snapshot snapshot;
                const auto                   result = source == 0 ? camera.poll( snapshot ) : sdk.poll( snapshot );

                if( result == loopTiming::Stream::Result::unchanged || result == loopTiming::Stream::Result::busy )
                {
                    continue;
                }

                if( result == loopTiming::Stream::Result::gap )
                {
                    auto &missed = source == 0 ? m_cameraMissed : m_sdkMissed;
                    missed.fetch_add( snapshot.missed );

                    reset = true;
                    break;
                }

                if( result != loopTiming::Stream::Result::updated )
                {
                    if( result == loopTiming::Stream::Result::reset )
                    {
                        ++m_counterRegressions;
                    }

                    ++m_invalidSnapshots;
                    reset = true;
                    break;
                }

                Observation observation;
                observation.camera  = source == 0;
                observation.counter = snapshot.counter;
                observation.gate    = generation;
                observation.epoch   = m_captureEpoch.load( std::memory_order_relaxed );
                observation.stampNs = observation.rawNs = snapshot.atimeNs;

                if( source == 0 )
                {
                    const auto before = monotonicNow();
                    timespec   external{};
                    const int  rv         = clock_gettime( externalClock, &external );
                    const auto after      = monotonicNow();
                    const auto externalNs = rv == 0 ? int64_t( external.tv_sec ) * 1000000000 + external.tv_nsec : 0;

                    if( !clock.sample( before, externalNs, after ) ||
                        !( observation.stampNs = clock.map( snapshot.atimeNs ) ) )
                    {
                        ++m_clockRejected;
                        reset = true;
                        break;
                    }

                    observation.lowerNs = clock.status().publicationLowerNs;
                    observation.upperNs = clock.status().publicationUpperNs + int64_t( m_cameraQuantizationUs * 1000 );
                }

                if( !m_queue.push( observation ) )
                {
                    reset = true;
                    break;
                }
            }

            if( reset )
            {
                camera.close();
                sdk.close();
                clock.reset();
                m_captureEpoch.fetch_add( 1, std::memory_order_release );
                m_streamsReady = false;
                nextAttach     = 0;
            }

            sleepFor( int64_t( m_pollSleepUs * 1000 ) );
        }
    }
    catch( const std::exception &error )
    {
        m_workerFailed = true;
        log<software_error>( { std::string( "loop timing polling: " ) + error.what() } );
    }
    catch( ... )
    {
        m_workerFailed = true;
        log<software_error>( { "unknown loop timing polling failure" } );
    }

    m_streamsReady = false;
    m_captureEpoch.fetch_add( 1, std::memory_order_release );
}

inline void loopTimingMonitor::stopWorker()
{
    m_stop.store( true, std::memory_order_release );

    if( m_thread.joinable() )
    {
        m_thread.join();
    }
}

inline int loopTimingMonitor::appShutdown()
{
    stopWorker();
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // loopTimingMonitor_hpp
