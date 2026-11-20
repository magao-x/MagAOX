/** \file autoJay.hpp
 * \brief Dual-DM differential registration for the automated J-test.
 * \ingroup autoJay_files
 */
#ifndef autoJay_hpp
#define autoJay_hpp

#include <chrono>
#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <sys/stat.h>

#include <mx/improc/milkImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "autoJayAcquisition.hpp"
#include "autoJayCalibration.hpp"
#include "autoJayRegistration.hpp"

/** \defgroup autoJay Automated J-test
 * \brief Registers NCPC and tweeter pokes against aligned reference images.
 * \ingroup apps
 */
/** \defgroup autoJay_files Automated J-test files
 * \ingroup autoJay
 */

namespace MagAOX
{
namespace app
{

/// A DM stream with coherent snapshots of its pixels and publication counter.
class autoJayDmStream : public mx::improc::milkImage<float>
{
  public:
    /// Open an existing stream without creating it.
    explicit autoJayDmStream( const std::string &name /**< [in] Stream name. */ );

    /// Close on fallback destruction without throwing during exception unwinding.
    ~autoJayDmStream() noexcept;

    /// Retain the checked Eigen-image write overloads.
    using mx::improc::milkImage<float>::operator=;

    /// Require a two-dimensional DM command map.
    bool plane() const;

    /// Copy only a stable publication, rejecting a concurrent writer.
    bool snapshot( autoJayDetail::imageT &image, /**< [out] Copied combined command. */
                   uint64_t              &counter /**< [out] Counter belonging to the copy. */ ) const;
};

inline autoJayDmStream::autoJayDmStream( const std::string &name ) : mx::improc::milkImage<float>( name )
{
}
inline autoJayDmStream::~autoJayDmStream() noexcept
{
    try
    {
        close();
    }
    catch( ... )
    {
    }
}
inline bool autoJayDmStream::plane() const
{
    return valid() && m_image->md->naxis == 2;
}
inline bool autoJayDmStream::snapshot( autoJayDetail::imageT &image, uint64_t &counter ) const
{
    if( !valid() || m_image->md->write )
        return false;
    counter = m_image->md->cnt0;
    image   = operator()();
    return !m_image->md->write && m_image->md->cnt0 == counter;
}

/// Measures registration; only the separate closedLoopIndi process commands ttmperi.
/** The main thread owns DM streams and disk I/O. The image monitor and INDI callbacks
 * share m_mutex with the main thread. No image lock is held during registration or disk I/O.
 * \ingroup autoJay
 */
class autoJay : public MagAOXApp<true>, public dev::shmimMonitor<autoJay>, public dev::telemeter<autoJay>
{
    friend class dev::shmimMonitor<autoJay>;
    friend class dev::telemeter<autoJay>;
    using shmimMonitorT = dev::shmimMonitor<autoJay>;
    using telemeterT    = dev::telemeter<autoJay>;
    using calibration   = autoJayDetail::calibration;
    using geometry      = autoJayDetail::geometry;
    using imageT        = autoJayDetail::imageT;

  protected:
    /** \name Configuration - Data
     * @{ */
    calibration                         m_captureConfig; ///< Bootstrap pattern identities and required setup.
    std::array<std::string, 2>                     m_patternFiles; ///< Optional FITS source maps for reference capture.
    std::array<std::vector<std::array<int, 2>>, 2> m_pokeActuators; ///< Zero-based (x, y) pairs for each DM.
    autoJayDetail::registrationSettings m_quality;       ///< Registration acceptance thresholds.
    std::string                         m_controller{ "autoJay-ctrl" }; ///< Guarded closedLoopIndi device.
    std::string                         m_aoLoop{ "holoop" };           ///< Main AO loop which must remain open.
    std::string m_alignmentLoop{ "twAlign-camwfs-ctrl" }; ///< Tweeter/WFS alignment loop which must remain open.
    std::string m_actuator{ "ttmperi" };                  ///< Read-only correction voltage source.
    bool        m_controllerEnabled{
        false };             ///< Require guarded correction acknowledgments; false permits initial calibration.
    double      m_frameMaxAge{ 5 };          ///< Maximum time since a coherent camera frame was received, in seconds.
    double      m_controlSettle{ 0.2 };      ///< Quiet interval after acknowledged voltage readback.
    double      m_controlTimeout{ 5 };       ///< Bounded correction acknowledgment/readback wait.
    double      m_voltageTolerance{ 0.005 }; ///< Covers one 150/32767 V controller quantization step.
    double      m_dmTolerance{ 1e-5 };       ///< Combined-command tolerance at pixels changed by a poke transition.
    std::string m_diagnosticsDir;            ///< Directory for rejected/requested diagnostic images.
    ///@}

    /// One subscribed property and its most recent complete update.
    struct observed
    {
        pcf::IndiProperty                subscription; ///< Stable registration object.
        std::optional<pcf::IndiProperty> value;        ///< Empty until the publisher supplies a value.
        std::string                      signature;    ///< Values/state excluding metadata such as labels.
    };

    std::mutex                      m_mutex; ///< Protects frame accumulation, external state, and requested operations.
    std::map<std::string, observed> m_observed;       ///< Subscriptions indexed by device.property.
    std::shared_ptr<const calibration> m_calibration; ///< Installed complete calibration.
    calibration                        m_cycle;       ///< Frozen calibration/configuration for the active transaction.
    geometry                           m_geometry;    ///< Frozen live geometry for the transaction.
    autoJayDetail::acquisition         m_acquisition; ///< Bounded frame/command state machine.
    uint64_t          m_generation{ 0 }; ///< Changes whenever relevant external state or stream allocation changes.
    uint64_t          m_cycleGeneration{ 0 }; ///< Generation sampled before acquisition.
    uint64_t          m_counter{ 0 };         ///< Accepted measurement sequence number.
    double            m_session{ 0 };         ///< Epoch-millisecond identity preserved by INDI's numeric serialization.
    bool              m_active{ false };      ///< A three-state transaction is in progress.
    bool              m_capture{ false };     ///< Active run records references instead of control samples.
    std::atomic<bool> m_continuous{ false };  ///< Repeat accepted measurement transactions.
    bool              m_startRequested{ false };   ///< Queued measurement request.
    bool              m_captureRequested{ false }; ///< Queued reference capture request.
    bool              m_reloadRequested{ false };  ///< Queued latest-calibration recall request.
    bool              m_stopRequested{ false };    ///< Queued cancellation request.
    bool              m_saveDiagnostics{ false };  ///< Save diagnostics for the next completed sequence.
    bool              m_waitingControl{ false };   ///< Awaiting this sample's controller acknowledgment and settling.
    double            m_controlDeadline{ 0 };      ///< Monotonic correction deadline.
    double            m_quietSince{ 0 };           ///< Start of matching command/current readback interval.
    double            m_lastFrameTime{ 0 };        ///< Last coherent camera frame received.
    double            m_pixelTime{ 0 };            ///< Last camera acquisition timestamp, for diagnostics.
    uint64_t          m_cameraCounter{ 0 };        ///< Last coherent camera counter.
    uint32_t          m_streamWidth{ 0 }, m_streamHeight{ 0 };  ///< Camera dimensions copied under the state lock.
    void ( *m_copyFrame )( float *, const void *, size_t ){ nullptr }; ///< Copies and converts one frame to float.
    std::array<std::unique_ptr<autoJayDmStream>, 2> m_channels; ///< Exclusively owned additive channels.
    std::array<std::unique_ptr<autoJayDmStream>, 2> m_triggers; ///< Optional combined streams monitored for feedback.
    std::array<imageT, 2>      m_lastCommands;    ///< Last commanded channel values, used to detect another writer.
    std::array<imageT, 2>      m_combinedBefore;  ///< Coherent combined commands preceding a channel write.
    std::array<imageT, 2>      m_expectedChange;  ///< Change which must appear in the combined command before settling.
    std::array<timespec, 4>    m_creationTimes{}; ///< Stream identities at open time.
    std::array<std::string, 4> m_streamPaths;     ///< Files backing the opened DM streams.
    std::array<ino_t, 4>       m_streamInodes{};  ///< File identities, detecting replacement of a mapped stream.
    std::array<off_t, 4>       m_streamBytes{};   ///< File lengths, detecting truncation before access.
    float                      m_deltaX{ 0 }, m_deltaY{ 0 }; ///< Latest accepted detector-pixel residual.
    std::string                m_status{ "idle; calibration and setup required" }; ///< Operator-facing status.

    pcf::IndiProperty m_indiP_measurement; ///< Coherent guarded controller input.
    pcf::IndiProperty m_indiP_ready;       ///< Opens the correction loop when unusable.
    pcf::IndiProperty m_indiP_status;      ///< Calibration identity and status/rejection reason.
    pcf::IndiProperty m_indiP_quality;     ///< Per-DM displacement and confidence metrics.
    pcf::IndiProperty m_indiP_continuous;  ///< Continuous acquisition toggle.
    pcf::IndiProperty m_indiP_measure;     ///< Single measurement request.
    pcf::IndiProperty m_indiP_capture;     ///< Reference capture request.
    pcf::IndiProperty m_indiP_reload;      ///< Recall newest calibration request.
    pcf::IndiProperty m_indiP_stop;        ///< Cancel acquisition request.
    pcf::IndiProperty m_indiP_diagnostics; ///< Save the next completed sequence request.

    /// Return monotonic seconds for deadlines.
    static double monotonic();

    /// Return epoch seconds for the guarded controller protocol.
    static double realtime();

    /// Parse a complete finite numeric element without accepting trailing text.
    static bool number( const pcf::IndiProperty &prop, /**< [in] Numeric property. */
                        const std::string       &name, /**< [in] Element name. */
                        double                  &value /**< [out] Parsed number. */ );

    /// Register a property once, retaining a stable callback object.
    int subscribe( const std::string &device, /**< [in] Publisher. */
                   const std::string &property /**< [in] Property name. */ );

    /// Dispatch subscribed values without issuing hardware commands.
    static int receive( void                    *object, /**< [in] Application pointer. */
                        const pcf::IndiProperty &property /**< [in] Received value. */ );

    /// Cache a received property and invalidate changed transaction state.
    int receive( const pcf::IndiProperty &property /**< [in] Received value. */ );

    /// Return the last published value, if known; caller holds m_mutex.
    const pcf::IndiProperty *property( const std::string &device, /**< [in] Publisher. */
                                       const std::string &name /**< [in] Property name. */ ) const;

    /// Validate external conditions and snapshot geometry; caller holds m_mutex.
    bool checkSetup( calibration &cal,     /**< [in/out] Capture rules may resolve wildcard values. */
                     geometry    &geom,    /**< [out] Applied camera geometry. */
                     bool         capture, /**< [in] Resolve wildcard values for a new reference. */
                     std::string &error,   /**< [out] Failure reason. */
                     bool settling = false /**< [in] Permit the acknowledged correction to be moving. */ ) const;

    /// Load the capture input for one DM from FITS or paired actuator coordinates.
    int loadPokeConfig( mx::app::appConfigurator &cfg, /**< [in] Parsed application settings. */
                        size_t                    dm /**< [in] Index of the NCPC or tweeter. */ );

    /// Check a saved map or expand configured coordinates using the actual stream dimensions.
    bool resolvePattern( calibration &cal,    /**< [in,out] Transaction with loaded or pending maps. */
                         size_t       dm,     /**< [in] Index of the NCPC or tweeter. */
                         size_t       width,  /**< [in] DM stream width. */
                         size_t       height, /**< [in] DM stream height. */
                         std::string &error /**< [out] Failure reason. */ ) const;

    /// Open only reserved, zeroed channels with compatible streams and maps.
    virtual bool openChannels( calibration &cal, /**< [in,out] Channel identities and effective maps. */
                               std::string &error /**< [out] Failure reason. */ );

    /// Verify streams and channel contents before reading or writing them.
    virtual bool channelsValid( std::string &error /**< [out] Failure reason. */ ) const;

    /// Read monitored combined-stream counters after checking identities.
    virtual std::array<uint64_t, 2> triggerCounters() const;

    /// Verify the commanded contribution on each monitored combined stream.
    virtual bool
    contributionApplied( std::array<uint64_t, 2> &counters /**< [out] Verified combined counters. */ ) const;

    /// Apply scales on the two owned channels using active writes.
    virtual bool writeChannels( const std::array<float, 2> &scales, /**< [in] Per-DM pattern scales. */
                                std::string                &error /**< [out] Failure reason. */ );

    /// Best-effort clear and close of both channels, without clearing another writer's data.
    virtual bool clearChannels( std::string &error /**< [out] Cleanup failure reason. */ );

    /// Start a measurement or reference capture after validating its snapshot.
    bool begin( bool capture /**< [in] Capture references instead of publishing. */ );

    /// Finish image processing outside image locks and publish only the same generation.
    void finish();

    /// Stop acquisition, clear owned channels, and invalidate the guarded input.
    void stop( const std::string &reason /**< [in] Operator-facing stop reason. */ );

    /// Publish measurement invalidity without advancing the accepted counter.
    void invalidate();

    /// Wait for the controller's acknowledgment, voltage readback, and quiet interval.
    bool controlSettled( std::string &error /**< [out] Failure reason; empty while waiting. */ );

    /// Load and atomically install the newest valid calibration in this profile.
    bool recall();

    /// Persist a completed sequence and processing context for offline inspection.
    void diagnostics( const std::array<autoJayDetail::frameStack, 3> &stacks, /**< [in] Completed images. */
                      const std::string                              &reason /**< [in] Why images are retained. */ );

    /// Publish status and running state from the main thread.
    void report();

    /// Record periodic standard poke-loop telemetry.
    int checkRecordTimes();

    /// Record the latest accepted residual and acquisition state.
    int recordTelem( const telem_pokeloop *type /**< [in] Telemetry dispatch tag. */ );

  public:
    /// Construct an idle sensor with no hardware connections.
    autoJay();

    /// Preserve the application's nonthrowing destructor contract.
    ~autoJay() noexcept override;

    /// Register configuration and base-class options.
    void setupConfig();

    /// Load and validate application configuration.
    void loadConfig();

    /// Load settings independently of startup for offline tests.
    int loadConfigImpl( mx::app::appConfigurator &cfg /**< [in] Parsed application settings. */ );

    /// Register subscriptions and start only the camera monitor.
    int appStartup();

    /// Drive the acquisition, persistence, and controller handshake.
    int appLogic();

    /// Clear owned channels and stop the image monitor before releasing resources.
    int appShutdown();

    /// Invalidate transactions when the monitored stream is allocated or recreated.
    int allocate( const dev::shmimT &tag /**< [in] Camera monitor tag. */ );

    /// Copy a coherent source frame into the active accumulator.
    int processImage( void              *source, /**< [in] Source frame pixels. */
                      const dev::shmimT &tag /**< [in] Camera monitor tag. */ );

    /// Queue continuous acquisition changes; ipRecv contains toggle.
    INDI_NEWCALLBACK_DECL( autoJay, m_indiP_continuous );

    /// Queue a single measurement; ipRecv contains request.
    INDI_NEWCALLBACK_DECL( autoJay, m_indiP_measure );

    /// Queue aligned-reference capture; ipRecv contains request.
    INDI_NEWCALLBACK_DECL( autoJay, m_indiP_capture );

    /// Queue calibration recall; ipRecv contains request.
    INDI_NEWCALLBACK_DECL( autoJay, m_indiP_reload );

    /// Queue cancellation; ipRecv contains request.
    INDI_NEWCALLBACK_DECL( autoJay, m_indiP_stop );

    /// Request diagnostic retention; ipRecv contains request.
    INDI_NEWCALLBACK_DECL( autoJay, m_indiP_diagnostics );
};

inline double autoJay::monotonic()
{
    return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}
inline double autoJay::realtime()
{
    return std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count();
}
inline bool autoJay::number( const pcf::IndiProperty &prop, const std::string &name, double &value )
{
    return prop.getType() == pcf::IndiProperty::Number && prop.find( name ) &&
           autoJayDetail::calibrationIO::number( value, prop[name].get() );
}
inline autoJay::autoJay() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_loopPause                                = 10000000;
    m_captureConfig.camera                     = "camflowfs";
    m_captureConfig.profile                    = "camflowfs";
    m_captureConfig.cameraGeometry.orientation = "native";
    m_captureConfig.dm[0].name                 = "dmncpc";
    m_captureConfig.dm[0].channel              = "dm02disp02";
    m_captureConfig.dm[0].trigger              = "dm02disp";
    m_captureConfig.dm[1].name                 = "dmtweeter";
    m_captureConfig.dm[1].channel              = "dm01disp02";
    m_captureConfig.dm[1].trigger              = "dm01disp";
}

inline autoJay::~autoJay() noexcept
{
    try
    {
        std::string error;
        clearChannels( error );
    }
    catch( ... )
    {
    }
}

inline void autoJay::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );
    auto add = [&]( const std::string &key, const std::string &type, const std::string &description )
    {
        const auto dot = key.find( '.' );
        config.add( key,
                    "",
                    key,
                    mx::app::argType::Required,
                    key.substr( 0, dot ),
                    key.substr( dot + 1 ),
                    false,
                    type,
                    description );
    };
    for( const auto &key : { "autoJay.camera",
                             "autoJay.profile",
                             "autoJay.orientation",
                             "autoJay.controller",
                             "autoJay.aoLoop",
                             "autoJay.alignmentLoop",
                             "autoJay.actuator",
                             "autoJay.diagnosticsDir" } )
        add( key, "string", "Camera/profile/control identity or diagnostic path." );
    for( const auto &dm : { "ncpc", "tweeter" } )
    {
        for( const auto &name : { "channel", "pattern" } )
        {
            add( std::string( dm ) + "." + name, "string", "Owned additive channel or FITS pattern." );
        }

        add( std::string( dm ) + ".trigger", "string", "Combined command stream; empty to time from the local write." );
        add( std::string( dm ) + ".amplitude", "float", "Signed nonzero command-map amplitude for reference capture." );
        add( std::string( dm ) + ".pokeX", "vector<int>", "Zero-based actuator x coordinates." );
        add( std::string( dm ) + ".pokeY", "vector<int>", "Zero-based actuator y coordinates paired with pokeX." );
        add( std::string( dm ) + ".pokeAmp", "float", "Signed command amplitude applied to each listed actuator." );
    }
    for( const auto &name : { "devices", "properties", "elements", "values", "types", "tolerances" } )
        add( std::string( "setup." ) + name,
             "vector<string>",
             "Parallel setup rules; types 0 numeric, 1 text, 2 switch. '*' captures a value." );
    for( const auto &name : { "frames", "settleFrames" } )
        add( std::string( "acquisition." ) + name,
             "unsigned",
             "Distinct averaged or discarded frame count; frames >= 2." );
    add( "control.enabled", "bool", "Enable guarded correction handshake only after measuring the controller matrix." );
    for( const auto &key : { "acquisition.dmSleep",
                             "acquisition.phaseTimeout",
                             "quality.maxShift",
                             "quality.minCorrelation",
                             "quality.minSnr",
                             "quality.minOverlap",
                             "quality.minReferenceEnergy",
                             "quality.maxSecondaryRatio",
                             "quality.peakExclusion",
                             "quality.saturation",
                             "control.settle",
                             "control.timeout",
                             "control.voltageTolerance",
                             "acquisition.dmTolerance" } )
    {
        add( key,
             "double",
             "Acquisition, acceptance, or settling threshold; times are seconds and shifts detector pixels." );
    }

    add( "monitor.maxAge", "double", "Maximum time since a coherent camera frame was received, in seconds." );
}

inline int autoJay::loadPokeConfig( mx::app::appConfigurator &cfg, size_t dm )
{
    const std::string prefix = dm == 0 ? "ncpc." : "tweeter.";
    cfg( m_captureConfig.dm[dm].channel, prefix + "channel" );
    cfg( m_captureConfig.dm[dm].trigger, prefix + "trigger" );
    cfg( m_patternFiles[dm], prefix + "pattern" );
    cfg( m_captureConfig.dm[dm].amplitude, prefix + "amplitude" );

    std::array<std::vector<std::string>, 2> coordinates;
    std::string                             amplitude;
    cfg( coordinates[0], prefix + "pokeX" );
    cfg( coordinates[1], prefix + "pokeY" );
    cfg( amplitude, prefix + "pokeAmp" );

    auto &actuators = m_pokeActuators[dm];
    actuators.clear();
    if( !cfg.isSet( prefix + "pokeX" ) && !cfg.isSet( prefix + "pokeY" ) && !cfg.isSet( prefix + "pokeAmp" ) )
    {
        return 0;
    }

    if( cfg.isSet( prefix + "pattern" ) || cfg.isSet( prefix + "amplitude" ) || coordinates[0].empty() ||
        coordinates[0].size() != coordinates[1].size() ||
        !autoJayDetail::calibrationIO::number( m_captureConfig.dm[dm].amplitude, amplitude ) )
    {
        log<text_log>( prefix + "configure paired pokeX/pokeY and a finite pokeAmp, or pattern and amplitude",
                       logPrio::LOG_ERROR );
        return -1;
    }

    for( size_t i = 0; i < coordinates[0].size(); ++i )
    {
        std::array<int, 2> point;
        for( size_t axis = 0; axis < point.size(); ++axis )
        {
            auto &value = coordinates[axis][i];
            auto  last  = value.find_last_not_of( " \t" );
            value.erase( last == std::string::npos ? 0 : last + 1 );

            if( !autoJayDetail::calibrationIO::number( point[axis], value ) || point[axis] < 0 )
            {
                log<text_log>( prefix + "pokeX and pokeY must contain nonnegative integer coordinates",
                               logPrio::LOG_ERROR );
                return -1;
            }
        }

        actuators.push_back( point );
    }

    return 0;
}

inline int autoJay::loadConfigImpl( mx::app::appConfigurator &cfg )
{
    SHMIMMONITOR_LOAD_CONFIG( cfg );
    TELEMETER_LOAD_CONFIG( cfg );
    cfg( m_captureConfig.camera, "autoJay.camera" );
    cfg( m_captureConfig.profile, "autoJay.profile" );
    cfg( m_captureConfig.cameraGeometry.orientation, "autoJay.orientation" );
    cfg( m_controller, "autoJay.controller" );
    cfg( m_aoLoop, "autoJay.aoLoop" );
    cfg( m_alignmentLoop, "autoJay.alignmentLoop" );
    cfg( m_actuator, "autoJay.actuator" );
    cfg( m_diagnosticsDir, "autoJay.diagnosticsDir" );
    for( size_t i = 0; i < 2; ++i )
    {
        if( loadPokeConfig( cfg, i ) < 0 )
        {
            return -1;
        }
    }
    cfg( m_captureConfig.frames, "acquisition.frames" );
    cfg( m_captureConfig.settleFrames, "acquisition.settleFrames" );
    cfg( m_captureConfig.dmSleep, "acquisition.dmSleep" );
    cfg( m_captureConfig.phaseTimeout, "acquisition.phaseTimeout" );
    cfg( m_quality.maxShift, "quality.maxShift" );
    m_quality.maxDifferential = m_quality.maxShift;
    cfg( m_quality.minCorrelation, "quality.minCorrelation" );
    cfg( m_quality.minSnr, "quality.minSnr" );
    cfg( m_quality.minOverlap, "quality.minOverlap" );
    cfg( m_quality.minReferenceEnergy, "quality.minReferenceEnergy" );
    cfg( m_quality.maxSecondaryRatio, "quality.maxSecondaryRatio" );
    cfg( m_quality.peakExclusion, "quality.peakExclusion" );
    cfg( m_quality.saturation, "quality.saturation" );
    cfg( m_controlSettle, "control.settle" );
    cfg( m_controllerEnabled, "control.enabled" );
    cfg( m_controlTimeout, "control.timeout" );
    cfg( m_voltageTolerance, "control.voltageTolerance" );
    cfg( m_dmTolerance, "acquisition.dmTolerance" );
    cfg( m_frameMaxAge, "monitor.maxAge" );
    std::array<std::vector<std::string>, 6> fields;
    const std::array<std::string, 6> keys{ "devices", "properties", "elements", "values", "types", "tolerances" };
    for( size_t i = 0; i < keys.size(); ++i )
        cfg( fields[i], "setup." + keys[i] );
    for( const auto &values : fields )
        if( values.size() != fields[0].size() )
            return -1;
    m_captureConfig.setup.clear();
    try
    {
        for( size_t i = 0; i < fields[0].size(); ++i )
        {
            size_t used = 0;
            int    type = std::stoi( fields[4][i], &used );
            if( used != fields[4][i].size() || type < 0 || type > 2 )
                return -1;
            double tolerance = std::stod( fields[5][i], &used );
            if( used != fields[5][i].size() || !mx::math::isFinite( tolerance ) || tolerance < 0 )
                return -1;
            m_captureConfig.setup.push_back(
                { fields[0][i], fields[1][i], fields[2][i], fields[3][i], type, tolerance } );
        }
    }
    catch( ... )
    {
        return -1;
    }
    if( m_captureConfig.setup.empty() )
    {
        if( m_captureConfig.camera != "camflowfs" )
            return -1;
        m_captureConfig.setup = { { "stagelosel", "presetName", "pupil", "1", 2, 0 },
                                  { "stageflowfs", "position", "current", "32.0", 0, 0.01 } };
    }
    std::set<std::string> ruleKeys;
    for( const auto &r : m_captureConfig.setup )
    {
        double value;
        if( r.device.empty() || r.property.empty() || r.element.empty() || r.value.empty() ||
            !ruleKeys.insert( r.device + "." + r.property + "." + r.element ).second ||
            ( r.type == 0 && r.value != "*" && !autoJayDetail::calibrationIO::number( value, r.value ) ) ||
            ( r.type == 2 && r.value != "*" && r.value != "0" && r.value != "1" ) )
            return -1;
    }
    if( m_captureConfig.camera == "camflowfs" )
    {
        for( const auto &required :
             std::vector<autoJayDetail::setupRule>{ { "stagelosel", "presetName", "pupil", "1", 2, 0 },
                                                    { "stagelosel", "position", "current", "*", 0, 0.01 },
                                                    { "stageflowfs", "position", "current", "32.0", 0, 0.01 } } )
        {
            const auto key = required.device + "." + required.property + "." + required.element;
            if( ruleKeys.insert( key ).second )
                m_captureConfig.setup.push_back( required );
            else
                for( const auto &r : m_captureConfig.setup )
                    if( r.device + "." + r.property + "." + r.element == key &&
                        ( r.type != required.type || ( required.type == 2 && r.value != "1" ) ) )
                        return -1;
        }
    }
    if( m_captureConfig.camera == "camsci1" )
    {
        bool filter = false, beamsplitter = false, focus = false;
        for( const auto &r : m_captureConfig.setup )
        {
            filter |= r.device == "fwscind" && r.property == "filterName" && r.type == 2;
            beamsplitter |= r.device == "stagebs" && r.property == "position" && r.element == "current" && r.type == 0;
            focus |= r.device == "stagesci1" && r.property == "position" && r.element == "current" && r.type == 0;
        }
        if( !filter || !beamsplitter || !focus )
            return -1;
    }
    // Loop-off rules are mandatory even when the optical requirements are customized.
    for( const auto &device : { m_aoLoop, m_alignmentLoop } )
    {
        const auto key = device + ".loop_state.toggle";
        if( ruleKeys.insert( key ).second )
            m_captureConfig.setup.push_back( { device, "loop_state", "toggle", "0", 2, 0 } );
        else
            for( const auto &r : m_captureConfig.setup )
                if( r.device + "." + r.property + "." + r.element == key && ( r.type != 2 || r.value != "0" ) )
                    return -1;
    }
    if( ruleKeys.insert( m_captureConfig.camera + ".exptime.current" ).second )
        m_captureConfig.setup.push_back( { m_captureConfig.camera, "exptime", "current", "*", 0, 1e-9 } );
    for( double value : { m_controlSettle, m_voltageTolerance, m_dmTolerance } )
        if( !mx::math::isFinite( value ) || value < 0 )
            return -1;
    for( double value : { m_controlTimeout, m_frameMaxAge, double( m_quality.maxShift ) } )
    {
        if( !mx::math::isFinite( value ) || value <= 0 )
        {
            return -1;
        }
    }

    if( m_controlSettle >= m_controlTimeout || m_controller.empty() || m_aoLoop.empty() || m_alignmentLoop.empty() ||
        m_actuator.empty() || m_captureConfig.profile.empty() )
    {
        return -1;
    }

    autoJayDetail::acquisition check;
    return check.start( m_captureConfig, 0 ) && autoJayDetail::registrationDetail::settingsValid( m_quality ) ? 0 : -1;
}

inline void autoJay::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<text_log>( "invalid autoJay configuration", logPrio::LOG_CRITICAL );
        m_shutdown = true;
    }
    m_captureConfig.stream = shmimMonitorT::shmimName();
    if( m_diagnosticsDir.empty() )
        m_diagnosticsDir = m_basePath + "/rawimages/" + m_configName;
}

inline int autoJay::subscribe( const std::string &device, const std::string &name )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    const auto                  key = device + "." + name;
    if( m_observed.count( key ) )
        return 0;
    return registerIndiPropertySet( m_observed[key].subscription,
                                    device,
                                    name,
                                    static_cast<int ( * )( void *, const pcf::IndiProperty & )>( &autoJay::receive ) );
}

inline int autoJay::receive( void *object, const pcf::IndiProperty &prop )
{
    return static_cast<autoJay *>( object )->receive( prop );
}

inline int autoJay::receive( const pcf::IndiProperty &prop )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    auto                        it = m_observed.find( prop.getDevice() + "." + prop.getName() );
    if( it == m_observed.end() )
        return -1;
    std::ostringstream signature;
    signature << prop.getState() << ':' << prop.getType();
    for( const auto &element : prop.getElements() )
        signature << ':' << element.first << '='
                  << ( prop.getType() == pcf::IndiProperty::Switch ? std::to_string( element.second.getSwitchState() )
                                                                   : element.second.get() );
    if( prop.getDevice() != m_controller && signature.str() != it->second.signature )
        ++m_generation;
    it->second.value     = prop;
    it->second.signature = signature.str();
    return 0;
}

inline const pcf::IndiProperty *autoJay::property( const std::string &device, const std::string &name ) const
{
    auto it = m_observed.find( device + "." + name );
    return it != m_observed.end() && it->second.value ? &*it->second.value : nullptr;
}

inline bool
autoJay::checkSetup( calibration &cal, geometry &geom, bool capture, std::string &error, bool settling ) const
{
    const double now    = monotonic();
    auto         reject = [&]( const std::string &why )
    {
        error = why;
        return false;
    };
    if( now - m_lastFrameTime > m_frameMaxAge || m_lastFrameTime == 0 )
    {
        return reject( "camera stream is stale" );
    }

    const auto *region = property( cal.camera, "roi_region_state" );
    if( !region || region->getState() != pcf::IndiProperty::Ok )
        return reject( "camera ROI is unavailable" );
    for( const auto &name : { "x", "y", "w", "h", "bin_x", "bin_y", "units" } )
        if( !region->find( name ) )
            return reject( "incomplete camera ROI" );
    try
    {
        std::array<double, 7>            r{};
        const std::array<std::string, 7> names{ "x", "y", "w", "h", "bin_x", "bin_y", "units" };
        for( size_t i = 0; i < r.size(); ++i )
            if( !number( *region, names[i], r[i] ) || std::abs( r[i] ) > 10000000 )
                return reject( "invalid ROI value" );
        for( size_t i = 2; i < 6; ++i )
            if( r[i] < 1 || r[i] != std::floor( r[i] ) )
                return reject( "invalid ROI extent/binning" );
        if( r[6] != 0 && r[6] != 1 )
            return reject( "unsupported ROI units" );
        geom.region = { float( r[0] ), float( r[1] ), int( r[2] ), int( r[3] ), int( r[4] ), int( r[5] ) };
        int units   = int( r[6] );
        if( units < 0 || units > 1 || !roi::valid( geom.region ) )
            return reject( "unsupported camera ROI" );
        geom.units       = static_cast<roi::units>( units );
        geom.orientation = m_captureConfig.cameraGeometry.orientation;
        if( roi::imageSize( geom.region.w, geom.region.bin_x, geom.units ) != m_streamWidth ||
            roi::imageSize( geom.region.h, geom.region.bin_y, geom.units ) != m_streamHeight )
            return reject( "camera metadata and stream dimensions differ" );
        if( !capture &&
            ( geom.units != cal.cameraGeometry.units || geom.orientation != cal.cameraGeometry.orientation ||
              geom.region.bin_x != cal.cameraGeometry.region.bin_x ||
              geom.region.bin_y != cal.cameraGeometry.region.bin_y ) )
            return reject( "calibration camera sampling/orientation differs" );
        for( auto &rule : cal.setup )
        {
            const auto *p   = property( rule.device, rule.property );
            const auto *fsm = property( rule.device, "fsm" );
            if( !p || !p->find( rule.element ) || p->getState() == pcf::IndiProperty::Busy ||
                p->getState() == pcf::IndiProperty::Alert )
                return reject( "missing, moving, or invalid setup: " + rule.device + "." + rule.property );
            if( !fsm || !fsm->find( "state" ) )
                return reject( "missing device state: " + rule.device );
            const auto state = ( *fsm )["state"].get();
            if( state != "READY" && state != "OPERATING" )
                return reject( "device not ready: " + rule.device );
            if( rule.type == 2 && ( p->getType() != pcf::IndiProperty::Switch ||
                                    ( ( *p )[rule.element].getSwitchState() != pcf::IndiElement::On &&
                                      ( *p )[rule.element].getSwitchState() != pcf::IndiElement::Off ) ) )
                return reject( "unknown setup switch: " + rule.device + "." + rule.property );
            std::string value = rule.type == 2
                                    ? ( ( *p )[rule.element].getSwitchState() == pcf::IndiElement::On ? "1" : "0" )
                                    : ( *p )[rule.element].get();
            if( rule.value == "*" && capture )
                rule.value = value;
            if( rule.type == 0 )
            {
                double actual, expected;
                if( !number( *p, rule.element, actual ) ||
                    !autoJayDetail::calibrationIO::number( expected, rule.value ) ||
                    std::abs( actual - expected ) > rule.tolerance )
                    return reject( "numeric setup mismatch: " + rule.device + "." + rule.property );
            }
            else if( value != rule.value )
                return reject( "setup mismatch: " + rule.device + "." + rule.property );
        }
        // These remain mandatory even if a loaded bundle has different optical setup rules.
        for( const auto &device : { m_aoLoop, m_alignmentLoop } )
        {
            const auto *loop = property( device, "loop_state" );
            if( !loop || !loop->find( "toggle" ) || ( *loop )["toggle"].getSwitchState() != pcf::IndiElement::Off )
                return reject( "loop must be open: " + device );
        }
        if( m_controllerEnabled )
        {
            const auto *ctrl = property( m_controller, "loop_state" );
            if( !ctrl || !ctrl->find( "toggle" ) )
                return reject( "correction controller unavailable" );
            if( capture && ( *ctrl )["toggle"].getSwitchState() != pcf::IndiElement::Off )
                return reject( "open the correction loop before capturing references" );
            const auto *actuator = property( m_actuator, "fsm" );
            if( !actuator || !actuator->find( "state" ) || ( *actuator )["state"].get() != "OPERATING" )
                return reject( "both correction axes must be enabled" );
        }
        for( const auto &axis : { "axis1_voltage", "axis2_voltage" } )
        {
            const auto *p = property( m_actuator, axis );
            double      current, target;
            if( !p || !number( *p, "current", current ) || !number( *p, "target", target ) ||
                p->getState() == pcf::IndiProperty::Alert ||
                ( !settling &&
                  ( p->getState() == pcf::IndiProperty::Busy || std::abs( current - target ) > m_voltageTolerance ) ) )
                return reject( "correction voltage is unavailable or unsettled" );
        }
    }
    catch( ... )
    {
        return reject( "invalid external setup value" );
    }
    return true;
}

inline bool
autoJay::resolvePattern( calibration &cal, size_t dm, size_t width, size_t height, std::string &error ) const
{
    auto &pattern = cal.dm[dm].pattern;
    if( pattern.size() != 0 )
    {
        if( size_t( pattern.rows() ) == width && size_t( pattern.cols() ) == height )
        {
            return true;
        }

        error = "DM pattern geometry mismatch: " + cal.dm[dm].channel;
        return false;
    }

    if( width == 0 || height == 0 || width > autoJayDetail::calibrationIO::maxPixels / height ||
        m_pokeActuators[dm].empty() )
    {
        error = "invalid DM geometry or missing actuator coordinates: " + cal.dm[dm].channel;
        return false;
    }

    for( const auto &point : m_pokeActuators[dm] )
    {
        if( size_t( point[0] ) >= width || size_t( point[1] ) >= height )
        {
            error = "poke coordinate outside DM stream geometry: " + cal.dm[dm].channel;
            return false;
        }
    }

    pattern.setZero( width, height );
    for( const auto &point : m_pokeActuators[dm] )
    {
        pattern( point[0], point[1] ) = 1;
    }

    return true;
}

inline bool autoJay::openChannels( calibration &cal, std::string &error )
{
    decltype( m_channels ) channels;
    decltype( m_triggers ) triggers;

    for( size_t i = 0; i < 2; ++i )
    {
        if( cal.dm[i].channel.empty() || cal.dm[i].channel == cal.dm[0].trigger ||
            cal.dm[i].channel == cal.dm[1].trigger || cal.dm[0].channel == cal.dm[1].channel )
        {
            error = "invalid dedicated DM channels";
            return false;
        }
        auto channel = std::make_unique<autoJayDmStream>( cal.dm[i].channel );
        if( !channel->plane() )
        {
            error = "DM command stream is unavailable or is not a plane: " + cal.dm[i].channel;
            return false;
        }

        std::unique_ptr<autoJayDmStream> trigger;
        if( !cal.dm[i].trigger.empty() )
        {
            trigger = std::make_unique<autoJayDmStream>( cal.dm[i].trigger );
            if( !trigger->plane() || trigger->rows() != channel->rows() || trigger->cols() != channel->cols() )
            {
                error = "DM combined stream geometry mismatch: " + cal.dm[i].trigger;
                return false;
            }
        }

        if( !resolvePattern( cal, i, channel->rows(), channel->cols(), error ) )
        {
            return false;
        }

        for( size_t p = 0; p < channel->rows() * channel->cols(); ++p )
        {
            if( ( *channel )[p] != 0 || !mx::math::isFinite( ( *channel )[p] ) )
            {
                error = "reserved channel is not zero: " + cal.dm[i].channel;
                return false;
            }
        }

        channel->passive( false ); // Notify the combiner or relay even when AO is off.
        for( size_t j = 0; j < 2; ++j )
        {
            const auto &stream = j == 0 ? channel : trigger;
            if( !stream )
            {
                continue;
            }

            char        path[4096];
            struct stat info;
            const auto &name = j == 0 ? cal.dm[i].channel : cal.dm[i].trigger;
            if( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) != IMAGESTREAMIO_SUCCESS ||
                stat( path, &info ) )
            {
                error = "DM stream file unavailable";
                return false;
            }
            m_streamPaths[2 * i + j]  = path;
            m_streamInodes[2 * i + j] = info.st_ino;
            m_streamBytes[2 * i + j]  = info.st_size;
            m_creationTimes[2 * i + j] = stream->creationtime();
        }

        m_lastCommands[i] = imageT::Zero( channel->rows(), channel->cols() );
        channels[i]       = std::move( channel );
        triggers[i]       = std::move( trigger );
    }

    m_channels = std::move( channels );
    m_triggers = std::move( triggers );
    return true;
}

inline bool autoJay::channelsValid( std::string &error ) const
{
    for( size_t i = 0; i < 2; ++i )
    {
        for( size_t j = 0; j < 2; ++j )
        {
            if( j == 1 && m_cycle.dm[i].trigger.empty() )
            {
                continue;
            }

            const auto &stream = j == 0 ? m_channels[i] : m_triggers[i];
            if( !stream )
            {
                error = "DM stream unavailable";
                return false;
            }

            struct stat info;
            if( stat( m_streamPaths[2 * i + j].c_str(), &info ) || info.st_ino != m_streamInodes[2 * i + j] ||
                info.st_size != m_streamBytes[2 * i + j] )
            {
                error = "DM stream file replaced or truncated";
                return false;
            }

            if( !stream->plane() )
            {
                error = "DM stream unavailable";
                return false;
            }

            const auto &t   = stream->creationtime();
            const auto &old = m_creationTimes[2 * i + j];
            if( t.tv_sec != old.tv_sec || t.tv_nsec != old.tv_nsec )
            {
                error = "DM stream recreated";
                return false;
            }
        }

        if( !( m_channels[i]->operator()() == m_lastCommands[i] ).all() )
        {
            error = "another writer changed an owned poke channel";
            return false;
        }
    }
    return true;
}

inline std::array<uint64_t, 2> autoJay::triggerCounters() const
{
    return { m_triggers[0] ? m_triggers[0]->cnt0() : 0, m_triggers[1] ? m_triggers[1]->cnt0() : 0 };
}

inline bool autoJay::contributionApplied( std::array<uint64_t, 2> &counters ) const
{
    for( size_t i = 0; i < 2; ++i )
    {
        if( !m_triggers[i] )
        {
            continue;
        }

        imageT combined;
        if( !m_triggers[i]->snapshot( combined, counters[i] ) )
            return false;
        if( combined.rows() != m_combinedBefore[i].rows() || combined.cols() != m_combinedBefore[i].cols() )
            return false;
        for( Eigen::Index n = 0; n < combined.size(); ++n )
        {
            const float delta = m_expectedChange[i].data()[n];
            if( delta == 0 )
                continue;
            if( !mx::math::isFinite( combined.data()[n] ) ||
                std::abs( combined.data()[n] - m_combinedBefore[i].data()[n] - delta ) > m_dmTolerance )
                return false;
        }
    }
    return true;
}

inline bool autoJay::writeChannels( const std::array<float, 2> &scales, std::string &error )
{
    if( !channelsValid( error ) )
    {
        return false;
    }

    for( size_t i = 0; i < 2; ++i )
    {
        if( !m_triggers[i] )
        {
            continue;
        }

        uint64_t counter;
        if( !m_triggers[i]->snapshot( m_combinedBefore[i], counter ) )
        {
            error = "combined DM command changed during snapshot";
            return false;
        }
        m_expectedChange[i] = m_cycle.dm[i].pattern * ( scales[i] * m_cycle.dm[i].amplitude ) - m_lastCommands[i];
    }
    for( size_t i = 0; i < 2; ++i )
    {
        m_lastCommands[i] = m_cycle.dm[i].pattern * ( scales[i] * m_cycle.dm[i].amplitude );
        *m_channels[i]    = m_lastCommands[i];
    }
    return true;
}

inline bool autoJay::clearChannels( std::string &error )
{
    bool success = true;
    for( size_t i = 0; i < 2; ++i )
    {
        try
        {
            if( m_channels[i] )
            {
                struct stat info;
                if( stat( m_streamPaths[2 * i].c_str(), &info ) || info.st_ino != m_streamInodes[2 * i] ||
                    info.st_size != m_streamBytes[2 * i] )
                    throw std::runtime_error( "channel file replaced or truncated" );
                const auto &t   = m_channels[i]->creationtime();
                const auto &old = m_creationTimes[2 * i];
                if( !m_channels[i]->valid() || t.tv_sec != old.tv_sec || t.tv_nsec != old.tv_nsec ||
                    !( m_channels[i]->operator()() == m_lastCommands[i] ).all() )
                    throw std::runtime_error( "channel identity or ownership changed" );
                m_lastCommands[i].setZero();
                *m_channels[i] = m_lastCommands[i];
            }
        }
        catch( const std::exception &e )
        {
            success = false;
            error += std::string( " cleanup: " ) + e.what();
        }
        try
        {
            if( m_channels[i] )
                m_channels[i]->close();
        }
        catch( const std::exception &e )
        {
            success = false;
            error += std::string( " channel close: " ) + e.what();
        }
        try
        {
            if( m_triggers[i] )
                m_triggers[i]->close();
        }
        catch( const std::exception &e )
        {
            success = false;
            error += std::string( " trigger close: " ) + e.what();
        }
        m_channels[i].reset();
        m_triggers[i].reset();
    }
    return success;
}

inline bool autoJay::begin( bool capture )
{
    calibration cycle;
    geometry    geom;
    std::string error;
    uint64_t    generation;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( !capture && !m_calibration )
        {
            m_status = "no valid calibration";
            return false;
        }
        cycle = capture ? m_captureConfig : *m_calibration;
        if( !checkSetup( cycle, geom, capture, error ) )
        {
            m_status = error;
            return false;
        }
        generation = m_generation;
    }
    if( capture )
    {
        for( size_t i = 0; i < 2; ++i )
        {
            auto &dm = cycle.dm[i];
            if( !mx::math::isFinite( dm.amplitude ) || dm.amplitude == 0 )
            {
                m_status = "capture requires a finite nonzero poke amplitude: " + dm.channel;
                return false;
            }

            if( !m_pokeActuators[i].empty() )
            {
                dm.pattern.resize( 0, 0 );
            }
            else if( !autoJayDetail::loadPattern( dm.pattern, m_patternFiles[i], error ) )
            {
                m_status = "capture pattern invalid for " + dm.channel + ": " + error;
                return false;
            }
        }

        cycle.cameraGeometry = geom;
    }
    for( const auto &dm : cycle.dm )
        for( Eigen::Index n = 0; n < dm.pattern.size(); ++n )
            if( !mx::math::isFinite( dm.pattern.data()[n] * dm.amplitude ) )
            {
                m_status = "scaled pattern is nonfinite";
                return false;
            }
    if( !openChannels( cycle, error ) )
    {
        stop( error );
        return false;
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( generation != m_generation || m_stopRequested )
            error = "setup changed before acquisition";
        else
        {
            m_cycle           = std::move( cycle );
            m_geometry        = geom;
            m_cycleGeneration = generation;
            m_capture         = capture;
            if( capture )
                m_continuous = false;
            m_active = m_acquisition.start( m_cycle, monotonic(), m_quality.saturation );
            if( !m_active )
                error = m_acquisition.error();
            m_status = capture ? "capturing aligned references" : "measuring";
        }
    }
    if( !error.empty() )
    {
        stop( error );
        return false;
    }
    updateSwitchIfChanged(
        m_indiP_ready, "toggle", capture || !m_controllerEnabled ? pcf::IndiElement::Off : pcf::IndiElement::On );
    return true;
}

inline void autoJay::invalidate()
{
    updateSwitchIfChanged( m_indiP_ready, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    updateIfChanged( m_indiP_measurement,
                     std::vector<std::string>{ "delta_x", "delta_y", "counter", "valid", "session", "timestamp" },
                     std::vector<double>{ m_deltaX, m_deltaY, double( m_counter ), 0, m_session, realtime() },
                     INDI_IDLE );
}

inline void autoJay::stop( const std::string &reason )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_active         = false;
        m_continuous     = false;
        m_waitingControl = false;
        m_acquisition.fail( reason );
        m_status = reason;
    }
    std::string error;
    if( !clearChannels( error ) )
    {
        m_status += error;
        log<text_log>( error, logPrio::LOG_ERROR );
    }
    invalidate();
}

inline bool autoJay::recall()
{
    uint64_t generation;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_calibration.reset();
        generation = ++m_generation;
    }
    calibration              candidate;
    std::string              error;
    std::vector<std::string> skipped;
    if( !autoJayDetail::latestCalibration(
            candidate, m_calibDir + "/" + m_configName, m_captureConfig.profile, error, &skipped ) )
    {
        m_status = "calibration unavailable: " + error;
        return false;
    }
    for( const auto &why : skipped )
        log<text_log>( why, logPrio::LOG_WARNING );
    if( candidate.camera != m_captureConfig.camera || candidate.stream != m_captureConfig.stream )
    {
        m_status = "calibration camera/stream identity mismatch";
        return false;
    }
    for( size_t i = 0; i < 2; ++i )
        if( candidate.dm[i].channel != m_captureConfig.dm[i].channel ||
            candidate.dm[i].trigger != m_captureConfig.dm[i].trigger ||
            candidate.dm[i].name != m_captureConfig.dm[i].name )
        {
            m_status = "calibration DM channel identity mismatch";
            return false;
        }
    for( const auto &required : m_captureConfig.setup )
    {
        auto it = std::find_if( candidate.setup.begin(),
                                candidate.setup.end(),
                                [&]( const auto &r )
                                {
                                    return r.device == required.device && r.property == required.property &&
                                           r.element == required.element && r.type == required.type;
                                } );
        if( it == candidate.setup.end() || ( required.device == "stagelosel" && required.property == "presetName" &&
                                             required.element == "pupil" && it->value != "1" ) )
        {
            m_status = "calibration is missing a required setup binding";
            return false;
        }
    }
    for( const auto &rule : candidate.setup )
        if( subscribe( rule.device, rule.property ) < 0 || subscribe( rule.device, "fsm" ) < 0 )
            return false;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( generation != m_generation )
        {
            m_status = "setup changed during recall; retry reload";
            return false;
        }
        m_calibration = std::make_shared<const calibration>( std::move( candidate ) );
        ++m_generation;
        m_status = "calibration loaded; verify setup before measuring";
    }
    return true;
}

inline bool autoJay::controlSettled( std::string &error )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    const double                now = monotonic();
    if( now > m_controlDeadline )
    {
        error = "correction acknowledgment/readback timed out";
        return false;
    }
    const auto *ack = property( m_controller, "processed" );
    if( !ack )
        return false;
    for( const auto &name : { "counter", "session", "target0", "target1", "applied", "status" } )
        if( !ack->find( name ) )
            return false;
    std::array<double, 6>            values{};
    const std::array<std::string, 6> names{ "counter", "session", "target0", "target1", "applied", "status" };
    for( size_t i = 0; i < values.size(); ++i )
        if( !number( *ack, names[i], values[i] ) )
        {
            error = "invalid controller acknowledgment number";
            return false;
        }
    if( values[1] != m_session || values[0] != double( m_counter ) )
        return false;
    double statusValue = values[5];
    if( statusValue < -2 || statusValue > 5 || statusValue != std::floor( statusValue ) ||
        values[4] != ( statusValue == 1 ? 1 : 0 ) )
    {
        error = "invalid controller acknowledgment status";
        return false;
    }
    int status = static_cast<int>( statusValue );
    if( status < 0 || status == 2 )
    {
        error = "controller rejected sample or actuator not ready";
        return false;
    }
    if( status == 4 || status == 5 || status == 0 )
    {
        m_quietSince = 0;
        return true;
    }
    const auto *fsm = property( m_actuator, "fsm" );
    if( !fsm || !fsm->find( "state" ) || ( *fsm )["state"].get() != "OPERATING" )
    {
        error = "correction actuator is not enabled";
        return false;
    }
    for( size_t i = 0; i < 2; ++i )
    {
        const auto  *axis   = property( m_actuator, i == 0 ? "axis1_voltage" : "axis2_voltage" );
        const double target = values[2 + i];
        double       current, commanded;
        if( !axis || !number( *axis, "current", current ) || !number( *axis, "target", commanded ) ||
            axis->getState() == pcf::IndiProperty::Busy || axis->getState() == pcf::IndiProperty::Alert ||
            std::abs( current - target ) > m_voltageTolerance || std::abs( commanded - target ) > m_voltageTolerance )
        {
            m_quietSince = 0;
            return false;
        }
    }
    if( m_quietSince == 0 )
        m_quietSince = now;
    return now - m_quietSince >= m_controlSettle;
}

inline void autoJay::diagnostics( const std::array<autoJayDetail::frameStack, 3> &stacks, const std::string &reason )
{
    try
    {
        std::filesystem::create_directories( m_diagnosticsDir );
        const std::string prefix =
            m_diagnosticsDir + "/" + m_configName + "_" + std::to_string( static_cast<uint64_t>( realtime() * 1e6 ) );
        mx::fits::fitsHeader<> head;
        head.append( "REASON", reason, "acquisition diagnostic reason" );
        head.append( "CALID", m_cycle.id, "active calibration" );
        head.append( "COUNTER", m_counter, "accepted measurements" );
        head.append( "SESSION", m_session, "process session" );
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            head.append( "PIXTIME", m_pixelTime, "last camera acquisition epoch seconds" );
        }
        head.append( "MAXSHIFT", m_quality.maxShift, "accepted detector-pixel limit" );
        mx::fits::fitsFile<float> file;
        for( size_t i = 0; i < 3; ++i )
        {
            if( !stacks[i].mean.size() )
                continue;
            auto h = head;
            h.append( "FIRST", stacks[i].firstCounter, "first camera frame" );
            h.append( "LAST", stacks[i].lastCounter, "last camera frame" );
            h.append( "NFRAMES", stacks[i].count, "distinct averaged frames" );
            if( !!file.write( prefix + "_" + std::to_string( i ) + ".fits", stacks[i].mean, h ) )
                throw std::runtime_error( "diagnostic FITS write failed" );
        }
        for( size_t i = 0; i < 2; ++i )
            if( m_cycle.dm[i].reference.size() &&
                !!file.write( prefix + "_reference" + std::to_string( i ) + ".fits", m_cycle.dm[i].reference, head ) )
                throw std::runtime_error( "diagnostic reference write failed" );
    }
    catch( const std::exception &e )
    {
        log<text_log>( e.what(), logPrio::LOG_ERROR );
    }
}

inline void autoJay::finish()
{
    std::array<autoJayDetail::frameStack, 3> stacks;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        stacks   = m_acquisition.stacks();
        m_active = false;
    }
    std::string error;
    if( !clearChannels( error ) )
    {
        stop( error );
        return;
    }
    auto result = m_capture ? autoJayDetail::validateCapture( stacks, m_geometry, m_quality )
                            : autoJayDetail::registerPair( m_cycle, stacks, m_geometry, m_quality );
    bool save;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        save              = m_saveDiagnostics;
        m_saveDiagnostics = false;
    }
    if( !result.valid || save )
        diagnostics( stacks, result.valid ? "requested" : result.reason );
    if( !result.valid )
    {
        stop( "measurement rejected: " + result.reason );
        return;
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        geometry                    current;
        if( m_cycleGeneration != m_generation || m_stopRequested || !checkSetup( m_cycle, current, m_capture, error ) )
            error = "setup changed during acquisition/processing: " + error;
    }
    if( !error.empty() )
    {
        stop( error );
        return;
    }
    if( m_capture )
    {
        m_cycle.baseline        = stacks[0].mean;
        m_cycle.dm[0].reference = stacks[1].mean - stacks[0].mean;
        m_cycle.dm[1].reference = stacks[2].mean - stacks[0].mean;
        if( !autoJayDetail::saveCalibration( m_cycle, m_calibDir + "/" + m_configName, error ) )
        {
            stop( "calibration save failed: " + error );
            return;
        }
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            geometry                    current;
            if( m_cycleGeneration != m_generation || m_stopRequested || !checkSetup( m_cycle, current, true, error ) )
                error = "reference bundle saved but not installed: setup changed during save";
            else
            {
                m_calibration = std::make_shared<const calibration>( m_cycle );
                ++m_generation;
            }
        }
        stop( error.empty() ? "aligned reference calibration saved" : error );
        return;
    }
    // Serialize validation/publication against setup and stop callbacks.
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( m_cycleGeneration != m_generation || m_stopRequested )
            error = "setup changed before publication";
        else
        {
            m_deltaX = result.deltaX;
            m_deltaY = result.deltaY;
            ++m_counter;
            updateIfChanged( m_indiP_quality,
                             std::vector<std::string>{ "ncpc_x",
                                                       "ncpc_y",
                                                       "ncpc_corr",
                                                       "ncpc_snr",
                                                       "tweeter_x",
                                                       "tweeter_y",
                                                       "tweeter_corr",
                                                       "tweeter_snr" },
                             std::vector<double>{ result.dm[0].dx,
                                                  result.dm[0].dy,
                                                  result.dm[0].correlation,
                                                  result.dm[0].snr,
                                                  result.dm[1].dx,
                                                  result.dm[1].dy,
                                                  result.dm[1].correlation,
                                                  result.dm[1].snr } );
            updateIfChanged(
                m_indiP_measurement,
                std::vector<std::string>{ "delta_x", "delta_y", "counter", "valid", "session", "timestamp" },
                std::vector<double>{
                    m_deltaX, m_deltaY, double( m_counter ), m_controllerEnabled ? 1.0 : 0.0, m_session, realtime() },
                INDI_OK );
            m_waitingControl  = m_controllerEnabled;
            m_controlDeadline = monotonic() + m_controlTimeout;
            m_quietSince      = 0;
            m_status          = m_controllerEnabled ? "measurement accepted; waiting for correction acknowledgment"
                                                    : "measurement accepted; corrections disabled";
            if( !m_controllerEnabled && m_continuous )
                m_startRequested = true;
        }
    }
    if( !error.empty() )
        stop( error );
    else
        recordTelem( static_cast<const telem_pokeloop *>( nullptr ) );
}

inline int autoJay::appStartup()
{
    m_session = static_cast<double>( static_cast<uint64_t>( realtime() * 1e3 ) );
    for( const auto &rule : m_captureConfig.setup )
        if( subscribe( rule.device, rule.property ) < 0 || subscribe( rule.device, "fsm" ) < 0 )
            return -1;
    for( const auto &name : { "roi_region_state", "fsm" } )
        if( subscribe( m_captureConfig.camera, name ) < 0 )
            return -1;
    if( m_controllerEnabled )
        for( const auto &name : { "loop_state", "processed" } )
            if( subscribe( m_controller, name ) < 0 )
                return -1;
    for( const auto &name : { "axis1_voltage", "axis2_voltage", "fsm" } )
        if( subscribe( m_actuator, name ) < 0 )
            return -1;
    CREATE_REG_INDI_RO_NUMBER( m_indiP_measurement, "measurement", "Registration residual", "Alignment" );
    for( const auto &name : { "delta_x", "delta_y", "counter", "valid", "session", "timestamp" } )
        m_indiP_measurement.add( pcf::IndiElement( name, 0.0 ) );
    CREATE_REG_INDI_NEW_TOGGLESWITCH_NOCB( m_indiP_ready, "ready" );
    m_indiP_ready.setPerm( pcf::IndiProperty::ReadOnly );
    createROIndiText( m_indiP_status, "status", "current" );
    m_indiP_status.add( pcf::IndiElement( "calibration" ) );
    if( registerIndiPropertyReadOnly( m_indiP_status ) < 0 )
        return -1;
    CREATE_REG_INDI_RO_NUMBER( m_indiP_quality, "quality", "Registration quality", "Alignment" );
    for( const auto &name :
         { "ncpc_x", "ncpc_y", "ncpc_corr", "ncpc_snr", "tweeter_x", "tweeter_y", "tweeter_corr", "tweeter_snr" } )
        m_indiP_quality.add( pcf::IndiElement( name, 0.0 ) );
    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_continuous, "continuous" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_measure, "measure" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_capture, "capture" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_reload, "reload" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_stop, "stop" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_diagnostics, "save_diagnostics" );
    invalidate();
    recall();
    SHMIMMONITOR_APP_STARTUP;
    TELEMETER_APP_STARTUP;
    state( stateCodes::READY );
    return 0;
}

inline int autoJay::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    try
    {
        bool cancel, capture, start, reload;
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            cancel             = m_stopRequested;
            m_stopRequested    = false;
            capture            = m_captureRequested;
            m_captureRequested = false;
            start              = m_startRequested;
            m_startRequested   = false;
            reload             = m_reloadRequested;
            m_reloadRequested  = false;
        }
        if( cancel )
            stop( "stopped by operator" );
        else if( reload )
        {
            stop( "recalling calibration" );
            recall();
        }
        else if( capture || start )
        {
            if( m_active || m_waitingControl )
                m_status = "request rejected: acquisition is busy";
            else if( !begin( capture ) )
                stop( m_status );
        }
        if( m_waitingControl )
        {
            std::string error;
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mutex );
                geometry                    current;
                checkSetup( m_cycle, current, false, error, true );
            }
            if( error.empty() && controlSettled( error ) )
            {
                m_waitingControl = false;
                if( m_continuous )
                {
                    if( !begin( false ) )
                        stop( m_status );
                }
                else
                    stop( "single measurement complete" );
            }
            else if( !error.empty() )
                stop( error );
        }
        bool        done = false;
        std::string error;
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            if( m_active )
            {
                geometry current;
                if( m_generation != m_cycleGeneration || !checkSetup( m_cycle, current, m_capture, error ) )
                    m_acquisition.fail( "setup changed or became unavailable: " + error );
                else if( !channelsValid( error ) )
                    m_acquisition.fail( error );
                else
                {
                    auto counters = triggerCounters();
                    m_acquisition.poll( counters, monotonic(), false );
                    if( m_acquisition.commandPending() )
                    {
                        if( writeChannels( m_acquisition.command(), error ) )
                            m_acquisition.commanded( counters, monotonic() );
                        else
                            m_acquisition.fail( error );
                    }
                    auto accepted = triggerCounters();
                    bool applied  = contributionApplied( accepted );
                    m_acquisition.poll( accepted, monotonic(), applied );
                }
                if( m_acquisition.state() == autoJayDetail::acquisition::step::failed )
                    error = m_acquisition.error();
                done = m_acquisition.state() == autoJayDetail::acquisition::step::complete;
            }
        }
        if( !error.empty() )
            stop( error );
        else if( done )
            finish();
        report();
    }
    catch( const std::exception &e )
    {
        stop( std::string( "acquisition failed: " ) + e.what() );
        report();
    }
    catch( ... )
    {
        stop( "unknown acquisition failure" );
        report();
    }
    return 0;
}

inline void autoJay::report()
{
    updateIfChanged( m_indiP_status, "current", m_status );
    updateIfChanged( m_indiP_status, "calibration", m_calibration ? m_calibration->id : std::string( "none" ) );
    updateSwitchIfChanged( m_indiP_continuous, "toggle", m_continuous ? pcf::IndiElement::On : pcf::IndiElement::Off );
    for( auto *p : { &m_indiP_measure, &m_indiP_capture, &m_indiP_reload, &m_indiP_stop, &m_indiP_diagnostics } )
        updateSwitchIfChanged( *p, "request", pcf::IndiElement::Off );
    state( m_active || m_waitingControl ? stateCodes::OPERATING : stateCodes::READY );
}

inline int autoJay::appShutdown()
{
    stop( "shutting down" );
    SHMIMMONITOR_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline int autoJay::allocate( const dev::shmimT &tag )
{
    static_cast<void>( tag );
    std::lock_guard<std::mutex> lock( m_mutex );
    ++m_generation;
    m_lastFrameTime = 0;
    m_streamWidth   = shmimMonitorT::width();
    m_streamHeight  = shmimMonitorT::height();
    m_copyFrame     = getPixFramePointer<float, false>( shmimMonitorT::dataType() );
    if( m_streamWidth == 0 || m_streamHeight == 0 ||
        uint64_t( m_streamWidth ) * m_streamHeight > autoJayDetail::calibrationIO::maxPixels )
    {
        m_copyFrame = nullptr;
    }

    if( m_active )
    {
        m_acquisition.fail( "camera stream allocated/recreated" );
    }

    return m_copyFrame ? 0 : -1;
}

inline int autoJay::processImage( void *source, const dev::shmimT &tag )
{
    const double received = monotonic();
    static_cast<void>( tag );
    auto *metadata = shmimMonitorT::m_imageStream.md;
    if( !metadata || metadata->write || !m_copyFrame )
    {
        return 0;
    }

    uint64_t counter = metadata->cnt0, plane = metadata->cnt1;
    timespec acquired = metadata->atime;
    imageT   copy( shmimMonitorT::width(), shmimMonitorT::height() );
    m_copyFrame( copy.data(), source, copy.size() );

    if( metadata->write || metadata->cnt0 != counter || metadata->cnt1 != plane )
    {
        return 0;
    }

    std::lock_guard<std::mutex> lock( m_mutex );
    m_lastFrameTime = received;
    m_pixelTime     = acquired.tv_sec + acquired.tv_nsec * 1e-9;
    m_cameraCounter = counter;
    if( m_active )
    {
        m_acquisition.frame( copy, counter, m_lastFrameTime );
    }

    return 0;
}

inline int autoJay::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_pokeloop() );
}
inline int autoJay::recordTelem( const telem_pokeloop *type )
{
    static_cast<void>( type );
    telem<telem_pokeloop>( { static_cast<uint8_t>( m_active || m_waitingControl ), m_deltaX, m_deltaY, m_counter } );
    return 0;
}

INDI_NEWCALLBACK_DEFN( autoJay, m_indiP_continuous )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_continuous, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    std::lock_guard<std::mutex> lock( m_mutex );
    m_continuous = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
    if( m_continuous )
        m_startRequested = true;
    else
        m_stopRequested = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( autoJay, m_indiP_measure )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_measure, ipRecv );
    std::lock_guard<std::mutex> lock( m_mutex );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        m_startRequested = true;
        m_continuous     = false;
    }
    return 0;
}
INDI_NEWCALLBACK_DEFN( autoJay, m_indiP_capture )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_capture, ipRecv );
    std::lock_guard<std::mutex> lock( m_mutex );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        m_captureRequested = true;
    return 0;
}
INDI_NEWCALLBACK_DEFN( autoJay, m_indiP_reload )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reload, ipRecv );
    std::lock_guard<std::mutex> lock( m_mutex );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        m_reloadRequested = true;
    return 0;
}
INDI_NEWCALLBACK_DEFN( autoJay, m_indiP_stop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stop, ipRecv );
    std::lock_guard<std::mutex> lock( m_mutex );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        m_stopRequested = true;
    return 0;
}
INDI_NEWCALLBACK_DEFN( autoJay, m_indiP_diagnostics )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_diagnostics, ipRecv );
    std::lock_guard<std::mutex> lock( m_mutex );
    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        m_saveDiagnostics = true;
    return 0;
}

} // namespace app
} // namespace MagAOX
#endif // autoJay_hpp
