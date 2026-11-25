/** \file modalFilter.hpp
 * \brief CACAO-compatible modal integration and per-mode predictive feedback.
 * \ingroup modalFilter_files
 */

#ifndef modalFilter_hpp
#define modalFilter_hpp

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <vector>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../libMagAOX/app/dev/modalPredictorFrame.hpp"
#include "../../magaox_git_version.h"

/** \defgroup modalFilter Modal filter
 * \ingroup apps
 */

/** \defgroup modalFilter_files
 * \ingroup modalFilter
 */

namespace MagAOX::app
{

/// Modal parameter and input stream roles.
enum class ModalFilterStream
{
    gain,
    mult,
    limit,
    zero,
    feedback,
    wfs,
    predictor
};

/// Name the stream monitor's config section and INDI group.
template <ModalFilterStream stream>
struct ModalFilterTag
{
    /// Return the application's configuration section for this stream.
    static std::string configSection()
    {
        return indiPrefix() + "Shmim";
    }

    /// Return the monitor's INDI property prefix.
    static std::string indiPrefix()
    {
        constexpr std::array names{
            "gainFact", "multFact", "limitFact", "zeroPoint", "dmFeedback", "modeval", "predictor" };
        return names[static_cast<size_t>( stream )];
    }
};

/// Apply closed-loop modal correction and publish commands on the CACAO output stream.
class modalFilter : public MagAOXApp<true>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::gain>>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::mult>>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::limit>>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::zero>>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::feedback>>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::wfs>>,
                    public dev::shmimMonitor<modalFilter, ModalFilterTag<ModalFilterStream::predictor>>,
                    public dev::frameGrabber<modalFilter>,
                    public dev::telemeter<modalFilter>
{
    friend class modalFilter_test;
    template <class, class>
    friend class dev::shmimMonitor;
    friend class dev::frameGrabber<modalFilter>;
    friend class dev::telemeter<modalFilter>;

    using Stream = ModalFilterStream;
    template <Stream stream>
    using Tag = ModalFilterTag<stream>;
    template <Stream stream>
    using Monitor       = dev::shmimMonitor<modalFilter, Tag<stream>>;
    using Gain          = Monitor<Stream::gain>;
    using Mult          = Monitor<Stream::mult>;
    using Limit         = Monitor<Stream::limit>;
    using Zero          = Monitor<Stream::zero>;
    using Feedback      = Monitor<Stream::feedback>;
    using Wfs           = Monitor<Stream::wfs>;
    using Predictor     = Monitor<Stream::predictor>;
    using frameGrabberT = dev::frameGrabber<modalFilter>;
    using telemeterT    = dev::telemeter<modalFilter>;

    static constexpr bool   c_frameGrabber_flippable = false; ///< Modal order must be preserved.
    static constexpr size_t c_pendingFrames          = 32;    ///< Bound the backlog of unprocessed WFS frames.

  protected:
    /** \name Configuration - Data
     * @{
     */
    std::string m_fpsDevice;               ///< INDI device reporting the WFS sample rate.
    std::string m_fpsProperty{ "fps" };    ///< Sample-rate property.
    std::string m_fpsElement{ "current" }; ///< Sample-rate element.
    float       m_fpsTol{ 0 };             ///< Minimum change in Hz that triggers reconfiguration.
    int         m_loopNum{ 1 };            ///< CACAO loop number used to name the streams.
    ///@}

    std::atomic<float>      m_fps{ 0 };  ///< Latest configured WFS sampling rate in Hz.
    std::mutex              m_filtMutex; ///< Guards commands, control snapshots, and queued frames.
    std::condition_variable m_ready;     ///< Signals new WFS input, a shape change, or a zero request.

    bool  m_loop{ false };            ///< Whether closed-loop commands may be published.
    bool  m_zeroRequested{ false };   ///< Pending explicit zero request, including while open.
    bool  m_feedbackPending{ false }; ///< Whether an external filtered command must seed the integrator.
    bool  m_pcRequested{ false };     ///< Operator's predictive-control request.
    bool  m_pcActive{ false };        ///< Whether the last processed command used a ready predictor.
    float m_gain{ 0 };                ///< SI global gain.
    float m_mult{ 1 };                ///< SI global command multiplier.
    float m_limit{ 1 };               ///< Global modal command limit.
    float m_pcGain{ 1 };              ///< PC global gain.
    float m_pcMult{ 1 };              ///< PC global command multiplier.

    /// CACAO factors, zero point, and external feedback in stream-role order.
    std::array<std::vector<float>, static_cast<size_t>( Stream::wfs )> m_parameters;
    std::optional<uint64_t>           m_feedbackCount;  ///< Last received filtered DM vector counter.
    std::vector<float>                m_predictorData;  ///< Owned coherent gain and tap frame.
    std::vector<modalPredictor::Mode> m_predictorModes; ///< Validated views into the owned frame.
    size_t m_historyNeeded{ 0 }; ///< Number of past samples required by the longest enabled predictor.

    /// Copy of one pending WFS sample and its original stream metadata.
    struct Sample
    {
        std::vector<float> modes;      ///< Modal values in native stream order.
        timespec           atime{};    ///< Acquisition timestamp to propagate to the command.
        uint64_t           count{ 0 }; ///< Source frame counter, used to detect a discontinuity.
    };

    Sample                              m_input;     ///< Preallocated scratch row, owned by the WFS monitor thread.
    std::array<Sample, c_pendingFrames> m_pending;   ///< Preallocated WFS queue in acquisition order.
    size_t                              m_head{ 0 }; ///< Oldest pending sample.
    size_t                              m_pendingCount{ 0 }; ///< Number of samples waiting for the command thread.
    uint64_t                            m_lastCount{ 0 }; ///< Counter of the last sample processed by the controller.
    bool                                m_haveCount{ false }; ///< Whether m_lastCount is valid.
    bool                                m_queueGap{ false };  ///< Set when queue overflow discarded a sample.

    size_t             m_modevalSz{ 0 };    ///< Active WFS modal dimension.
    std::vector<float> m_command;           ///< Last command, retained while the loop is open.
    std::vector<float> m_output;            ///< Framegrabber-owned publication snapshot.
    eigenImage<float>  m_wfsHistory;        ///< Past WFS errors, one frame per column.
    eigenImage<float>  m_dmHistory;         ///< Past clipped DM commands, one frame per column.
    size_t             m_historyNext{ 0 };  ///< Next history column to write.
    size_t             m_historyCount{ 0 }; ///< Number of consecutive past samples available.

  public:
    /// Configure monitors to read the current parameter and predictor frames on attach.
    modalFilter();

    /// Release filter storage after the framework has stopped the workers.
    ~modalFilter() noexcept override = default;

    /// Register the loop and stream configuration.
    void setupConfig() override;

    /// Load configuration and assign CACAO stream names.
    void loadConfig() override;

    /// Load stream configuration with the framework's error-returning helpers.
    int loadConfigImpl( mx::app::appConfigurator &settings /**< [in] App configuration. */ );

    /// Start stream monitors, the framegrabber and INDI controls.
    int appStartup() override;

    /// Service the monitors and publish loop status.
    int appLogic() override;

    /// Stop the worker threads.
    int appShutdown() override;

    /// Validate an input stream and initialize WFS storage on attachment.
    template <Stream stream>
    int allocate( Tag<stream> /**< [in] Monitor role. */ );

    /// Copy a stable input frame into the queue or install a parameter snapshot.
    template <Stream stream>
    int processImage( void *source /**< [in] Selected shared-memory frame. */, Tag<stream> /**< [in] Monitor role. */ );

    /// Size the output to the active WFS mode vector.
    int configureAcquisition();

    /// Return the configured WFS frame rate.
    float fps();

    /// Confirm the framegrabber is ready to receive samples.
    int startAcquisition();

    /// Consume one queued sample or zero request and prepare its output.
    int acquireAndCheckValid();

    /// Copy the immutable command prepared by acquireAndCheckValid.
    int loadImageIntoStream( void *dest /**< [out] Framegrabber output buffer. */ );

    /// Complete a framegrabber reconfiguration.
    int reconfig();

    /// Check whether framegrabber timing telemetry is due.
    int checkRecordTimes();

    /// Record framegrabber timing telemetry.
    int recordTelem( const telem_fgtimings * /**< [in] Timing record tag. */ );

  protected:
    /// Allocate all sample buffers and clear history on a WFS stream attachment.
    void configureModes( size_t modes /**< [in] Number of WFS modes. */ );

    /// Transfer the already copied WFS sample into the bounded queue.
    void enqueue( Sample &sample /**< [in,out] Sample to queue and scratch storage to recycle. */ );

    /// Retire past inputs and commands while holding m_filtMutex.
    void resetHistory();

    /// Perform one controller update while holding m_filtMutex.
    void filter( const Sample &sample /**< [in] Next WFS sample in order. */ );

    /// Validate and install a complete predictor snapshot.
    bool installPredictor( std::vector<float> &data /**< [in,out] Complete mode-major frame. */,
                           size_t              width /**< [in] Floats per mode. */,
                           size_t              modes /**< [in] Number of mode columns. */ );

    /// Fetch a parameter when its stream has the active modal dimension.
    float parameter( Stream stream /**< [in] Parameter role. */,
                     size_t mode /**< [in] Mode index. */,
                     float  initial /**< [in] CACAO default before attachment. */ ) const;

    pcf::IndiProperty m_indiP_fpsSource; ///< Subscription to the frame-rate source.
    pcf::IndiProperty m_indiP_loop;      ///< Loop state control.
    pcf::IndiProperty m_indiP_loopZero;  ///< Explicit zero request.
    pcf::IndiProperty m_indiP_gain;      ///< Global SI gain control.
    pcf::IndiProperty m_indiP_mult;      ///< Global SI multiplier control.
    pcf::IndiProperty m_indiP_limit;     ///< Global modal limit control.
    pcf::IndiProperty m_indiP_pcGain;    ///< Global PC gain control.
    pcf::IndiProperty m_indiP_pcMult;    ///< Global PC multiplier control.
    pcf::IndiProperty m_indiP_pcOn;      ///< Active predictive state and enable request.

    /// Receive the WFS sample rate.
    INDI_SETCALLBACK_DECL( modalFilter, m_indiP_fpsSource );

    /// Open or close the modal loop.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_loop );

    /// Queue a zero command.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_loopZero );

    /// Update the global SI gain.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_gain );

    /// Update the global SI multiplier.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_mult );

    /// Update the global limit.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_limit );

    /// Update the global PC gain.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_pcGain );

    /// Update the global PC multiplier.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_pcMult );

    /// Enable or disable the predictor.
    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_pcOn );
};

modalFilter::modalFilter() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    Gain::m_getExistingFirst      = true;
    Mult::m_getExistingFirst      = true;
    Limit::m_getExistingFirst     = true;
    Zero::m_getExistingFirst      = true;
    Feedback::m_getExistingFirst  = true;
    Predictor::m_getExistingFirst = true;
    frameGrabberT::m_staged       = true;
}

void modalFilter::setupConfig()
{
    config.add( "circBuff.fpsDevice",
                "",
                "circBuff.fpsDevice",
                argType::Required,
                "circBuff",
                "fpsDevice",
                false,
                "string",
                "INDI device reporting the WFS frame rate." );
    config.add( "circBuff.fpsProperty",
                "",
                "circBuff.fpsProperty",
                argType::Required,
                "circBuff",
                "fpsProperty",
                false,
                "string",
                "Frame-rate property." );
    config.add( "circBuff.fpsElement",
                "",
                "circBuff.fpsElement",
                argType::Required,
                "circBuff",
                "fpsElement",
                false,
                "string",
                "Frame-rate element." );
    config.add( "circBuff.fpsTol",
                "",
                "circBuff.fpsTol",
                argType::Required,
                "circBuff",
                "fpsTol",
                false,
                "float",
                "Tolerance for frame-rate changes in Hz." );
    config.add( "loop.number",
                "",
                "loop.number",
                argType::Required,
                "loop",
                "number",
                false,
                "int",
                "CACAO loop number used to name the streams." );

    SHMIMMONITORT_SETUP_CONFIG( Gain, config );
    SHMIMMONITORT_SETUP_CONFIG( Mult, config );
    SHMIMMONITORT_SETUP_CONFIG( Limit, config );
    SHMIMMONITORT_SETUP_CONFIG( Zero, config );
    SHMIMMONITORT_SETUP_CONFIG( Feedback, config );
    SHMIMMONITORT_SETUP_CONFIG( Wfs, config );
    SHMIMMONITORT_SETUP_CONFIG( Predictor, config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );
}

int modalFilter::loadConfigImpl( mx::app::appConfigurator &settings )
{
    settings( m_fpsDevice, "circBuff.fpsDevice" );
    settings( m_fpsProperty, "circBuff.fpsProperty" );
    settings( m_fpsElement, "circBuff.fpsElement" );
    settings( m_fpsTol, "circBuff.fpsTol" );
    settings( m_loopNum, "loop.number" );

    const std::string prefix   = "aol" + std::to_string( m_loopNum ) + "_";
    Gain::m_shmimName          = prefix + "mgainfact";
    Mult::m_shmimName          = prefix + "mmultfact";
    Limit::m_shmimName         = prefix + "mlimitfact";
    Zero::m_shmimName          = prefix + "mzeropoint";
    Feedback::m_shmimName      = prefix + "modevalDMf";
    Wfs::m_shmimName           = prefix + "modevalWFS";
    Predictor::m_shmimName     = prefix + "modalPredictor";
    frameGrabberT::m_shmimName = prefix + "modevalDM";

    SHMIMMONITORT_LOAD_CONFIG( Gain, settings );
    SHMIMMONITORT_LOAD_CONFIG( Mult, settings );
    SHMIMMONITORT_LOAD_CONFIG( Limit, settings );
    SHMIMMONITORT_LOAD_CONFIG( Zero, settings );
    SHMIMMONITORT_LOAD_CONFIG( Feedback, settings );
    SHMIMMONITORT_LOAD_CONFIG( Wfs, settings );
    SHMIMMONITORT_LOAD_CONFIG( Predictor, settings );
    FRAMEGRABBER_LOAD_CONFIG( settings );
    TELEMETER_LOAD_CONFIG( settings );
    return 0;
}

void modalFilter::loadConfig()
{
    loadConfigImpl( config );
}

int modalFilter::appStartup()
{
    if( m_fpsDevice.empty() )
    {
        return log<software_critical, -1>( { "Configure circBuff.fpsDevice for the WFS sample rate." } );
    }

    SHMIMMONITORT_APP_STARTUP( Gain );
    SHMIMMONITORT_APP_STARTUP( Mult );
    SHMIMMONITORT_APP_STARTUP( Limit );
    SHMIMMONITORT_APP_STARTUP( Zero );
    SHMIMMONITORT_APP_STARTUP( Feedback );
    SHMIMMONITORT_APP_STARTUP( Wfs );
    SHMIMMONITORT_APP_STARTUP( Predictor );
    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_loop, "loop_state" );
    createStandardIndiRequestSw( m_indiP_loopZero, "loop_zero", "Loop Zero", "Loop" );
    if( registerIndiPropertyNew( m_indiP_loopZero, INDI_NEWCALLBACK( m_indiP_loopZero ) ) < 0 )
    {
        return -1;
    }

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_gain, "loop_gain", 0, 10, 0.01, "%0.03f", "Gain", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_mult, "loop_multcoeff", 0, 1, 0.01, "%0.03f", "Mult. Coeff.", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_limit, "loop_max_limit", 0, 10, 0.01, "%0.03f", "Limit", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_pcGain, "loop_pcgain", 0, 1, 0.01, "%0.03f", "PC Gain", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_pcMult, "loop_pcmultcoeff", 0, 1, 0.01, "%0.03f", "PC Mult. Coeff.", "Loop" );
    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_pcOn, "loop_pcOn" );

    state( stateCodes::OPERATING );
    return 0;
}

int modalFilter::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( Gain );
    SHMIMMONITORT_APP_LOGIC( Mult );
    SHMIMMONITORT_APP_LOGIC( Limit );
    SHMIMMONITORT_APP_LOGIC( Zero );
    SHMIMMONITORT_APP_LOGIC( Feedback );
    SHMIMMONITORT_APP_LOGIC( Wfs );
    SHMIMMONITORT_APP_LOGIC( Predictor );
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    SHMIMMONITORT_UPDATE_INDI( Gain );
    SHMIMMONITORT_UPDATE_INDI( Mult );
    SHMIMMONITORT_UPDATE_INDI( Limit );
    SHMIMMONITORT_UPDATE_INDI( Zero );
    SHMIMMONITORT_UPDATE_INDI( Feedback );
    SHMIMMONITORT_UPDATE_INDI( Wfs );
    SHMIMMONITORT_UPDATE_INDI( Predictor );
    FRAMEGRABBER_UPDATE_INDI;

    bool  loop, pcActive, pcRequested, zeroPending;
    float gain, mult, limit, pcGain, pcMult;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_filtMutex );
        loop        = m_loop;
        pcActive    = m_pcActive;
        pcRequested = m_pcRequested;
        zeroPending = m_zeroRequested;
        gain        = m_gain;
        mult        = m_mult;
        limit       = m_limit;
        pcGain      = m_pcGain;
        pcMult      = m_pcMult;
    }

    updateSwitchIfChanged(
        m_indiP_loop, "toggle", loop ? pcf::IndiElement::On : pcf::IndiElement::Off, loop ? INDI_OK : INDI_IDLE );
    updateSwitchIfChanged( m_indiP_pcOn,
                           "toggle",
                           pcActive ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           pcActive ? INDI_OK : ( pcRequested ? INDI_BUSY : INDI_IDLE ) );
    updateSwitchIfChanged( m_indiP_loopZero, "request", pcf::IndiElement::Off, zeroPending ? INDI_BUSY : INDI_IDLE );
    updatesIfChanged<float>( m_indiP_gain, { "current", "target" }, { gain, gain } );
    updatesIfChanged<float>( m_indiP_mult, { "current", "target" }, { mult, mult } );
    updatesIfChanged<float>( m_indiP_limit, { "current", "target" }, { limit, limit } );
    updatesIfChanged<float>( m_indiP_pcGain, { "current", "target" }, { pcGain, pcGain } );
    updatesIfChanged<float>( m_indiP_pcMult, { "current", "target" }, { pcMult, pcMult } );
    return 0;
}

int modalFilter::appShutdown()
{
    m_ready.notify_all();
    SHMIMMONITORT_APP_SHUTDOWN( Gain );
    SHMIMMONITORT_APP_SHUTDOWN( Mult );
    SHMIMMONITORT_APP_SHUTDOWN( Limit );
    SHMIMMONITORT_APP_SHUTDOWN( Zero );
    SHMIMMONITORT_APP_SHUTDOWN( Feedback );
    SHMIMMONITORT_APP_SHUTDOWN( Wfs );
    SHMIMMONITORT_APP_SHUTDOWN( Predictor );
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

template <modalFilter::Stream stream>
int modalFilter::allocate( Tag<stream> )
{

    if( Monitor<stream>::m_dataType != _DATATYPE_FLOAT ||
        ( stream != Stream::predictor && ( Monitor<stream>::m_height != 1 || Monitor<stream>::m_width == 0 ) ) ||
        ( stream == Stream::predictor && !modalPredictor::validWidth( Monitor<stream>::m_width ) ) )
    {
        return log<software_error, -1>( { "Invalid shape or datatype for " + Monitor<stream>::m_shmimName } );
    }

    if constexpr( stream == Stream::wfs )
    {
        configureModes( Monitor<stream>::m_width );
    }
    else if constexpr( stream == Stream::feedback )
    {
        std::lock_guard<std::mutex> lock( m_filtMutex );
        m_feedbackCount.reset();
        m_feedbackPending = false;
    }

    return 0;
}

template <modalFilter::Stream stream>
int modalFilter::processImage( void *source, Tag<stream> )
{
    const auto *md = Monitor<stream>::m_imageStream.md;

    if( __atomic_load_n( &md->write, __ATOMIC_ACQUIRE ) != 0 )
    {
        return 0;
    }

    const uint64_t     count = Monitor<stream>::currCnt0();
    const timespec     atime = Monitor<stream>::currAtime();
    const size_t       size  = static_cast<size_t>( Monitor<stream>::m_width ) * Monitor<stream>::m_height;
    const float       *data  = static_cast<const float *>( source );
    std::vector<float> snapshot;

    if constexpr( stream == Stream::wfs )
    {
        std::copy_n( data, size, m_input.modes.begin() );
        m_input.count = count;
        m_input.atime = atime;
    }
    else
    {
        snapshot.assign( data, data + size );
    }

    std::atomic_thread_fence( std::memory_order_acquire );
    if( __atomic_load_n( &md->write, __ATOMIC_ACQUIRE ) != 0 || Monitor<stream>::currCnt0() != count )
    {
        return 0;
    }

    if constexpr( stream == Stream::wfs )
    {
        enqueue( m_input );
    }
    else if constexpr( stream == Stream::predictor )
    {
        if( count > 0 )
        {
            installPredictor( snapshot, Monitor<stream>::m_width, Monitor<stream>::m_height );
        }
    }
    else
    {
        for( float value : snapshot )
        {
            if( !mx::math::isFinite( value ) || ( stream == Stream::limit && value < 0 ) )
            {
                return 0;
            }
        }

        std::lock_guard<std::mutex> lock( m_filtMutex );
        if constexpr( stream == Stream::feedback )
        {
            if( count == 0 || m_feedbackCount == count )
            {
                return 0;
            }

            m_feedbackCount   = count;
            m_feedbackPending = true;
        }

        m_parameters[static_cast<size_t>( stream )].swap( snapshot );

        if constexpr( stream == Stream::zero )
        {
            if( m_parameters[static_cast<size_t>( stream )] != snapshot )
            {
                resetHistory();
            }
        }
    }

    return 0;
}

void modalFilter::configureModes( size_t modes )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_filtMutex );
        if( modes != m_modevalSz )
        {
            m_modevalSz = modes;
            m_command.assign( modes, 0 );
            m_input.modes.resize( modes );
            for( auto &sample : m_pending )
            {
                sample.modes.resize( modes );
            }

            m_wfsHistory.resize( modes, modalPredictor::maxTaps );
            m_dmHistory.resize( modes, modalPredictor::maxTaps );
            frameGrabberT::m_reconfig = true;
        }

        m_head         = 0;
        m_pendingCount = 0;
        m_queueGap     = false;
        resetHistory();
    }
    m_ready.notify_all();
}

void modalFilter::enqueue( Sample &sample )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_filtMutex );
        if( !m_loop || sample.modes.size() != m_modevalSz )
        {
            return;
        }

        if( m_pendingCount == m_pending.size() )
        {
            m_head = ( m_head + 1 ) % m_pending.size();
            --m_pendingCount;
            m_queueGap = true;
        }

        auto &pending = m_pending[( m_head + m_pendingCount ) % m_pending.size()];
        std::swap( pending, sample );
        ++m_pendingCount;
    }
    m_ready.notify_one();
}

void modalFilter::resetHistory()
{
    m_historyNext  = 0;
    m_historyCount = 0;
    m_haveCount    = false;
    m_pcActive     = false;
}

bool modalFilter::installPredictor( std::vector<float> &data, size_t width, size_t modes )
{
    if( !modalPredictor::validWidth( width ) || data.size() / width != modes || data.size() % width != 0 )
    {
        return false;
    }

    std::vector<modalPredictor::Mode> decoded( modes );
    size_t                            needed = 0;

    for( size_t n = 0; n < modes; ++n )
    {
        auto &mode = decoded[n];
        if( !modalPredictor::decode( mode, { data.data() + n * width, width } ) )
        {
            return false;
        }

        if( mode.enabled() )
        {
            needed = std::max( { needed, mode.a.size(), mode.b.size() - 1 } );
        }
    }

    std::lock_guard<std::mutex> lock( m_filtMutex );
    m_predictorData.swap( data );
    m_predictorModes.swap( decoded );
    m_historyNeeded = needed;
    return true;
}

float modalFilter::parameter( Stream stream, size_t mode, float initial ) const
{
    const auto &values = m_parameters[static_cast<size_t>( stream )];
    return values.size() == m_modevalSz ? values[mode] : initial;
}

void modalFilter::filter( const Sample &sample )
{
    if( m_queueGap || ( m_haveCount && sample.count != m_lastCount + 1 ) )
    {
        resetHistory();
        m_queueGap = false;
    }

    m_lastCount = sample.count;
    m_haveCount = true;

    if( m_feedbackPending )
    {
        const auto &feedback = m_parameters[static_cast<size_t>( Stream::feedback )];
        if( feedback.size() == m_modevalSz )
        {
            std::copy( feedback.begin(), feedback.end(), m_command.begin() );
            m_historyCount = 0;
        }

        m_feedbackPending = false;
    }

    constexpr size_t length = modalPredictor::maxTaps;
    m_pcActive              = m_pcRequested && m_historyNeeded != 0 && m_historyCount >= m_historyNeeded &&
                 m_predictorModes.size() == m_modevalSz;
    const size_t current = m_historyNext;

    for( size_t n = 0; n < m_modevalSz; ++n )
    {
        const double error         = static_cast<double>( parameter( Stream::zero, n, 0 ) ) - sample.modes[n];
        const float  gain          = m_gain * parameter( Stream::gain, n, 1 );
        const float  mult          = m_mult * parameter( Stream::mult, n, 1 );
        const float  bound         = m_limit * parameter( Stream::limit, n, 1 );
        const float  command       = error * gain + m_command[n] * mult;
        m_command[n]               = std::clamp( command, -bound, bound );
        m_wfsHistory( n, current ) = error;
    }

    if( m_pcActive )
    {
        for( size_t n = 0; n < m_modevalSz; ++n )
        {
            const auto &mode = m_predictorModes[n];
            if( !mode.enabled() )
            {
                continue;
            }

            float wfs = mode.b[0] * m_wfsHistory( n, current );
            float dm  = 0;

            for( size_t tap = 1; tap < mode.b.size(); ++tap )
            {
                wfs += mode.b[tap] * m_wfsHistory( n, ( current + length - tap ) % length );
            }

            for( size_t tap = 0; tap < mode.a.size(); ++tap )
            {
                dm += mode.a[tap] * m_dmHistory( n, ( current + length - 1 - tap ) % length );
            }

            const float command = m_pcGain * mode.gain * wfs + m_pcMult * mode.mult * dm;
            const float bound   = m_limit * parameter( Stream::limit, n, 1 );
            m_command[n]        = std::clamp( command, -bound, bound );
        }
    }

    std::copy( m_command.begin(), m_command.end(), m_dmHistory.col( current ).data() );
    m_historyNext  = ( current + 1 ) % length;
    m_historyCount = std::min( m_historyCount + 1, length );
}

int modalFilter::configureAcquisition()
{
    std::unique_lock<std::mutex> lock( m_filtMutex );
    while( m_modevalSz == 0 && !shutdown() )
    {
        m_ready.wait_for( lock, std::chrono::seconds( 1 ) );
    }

    if( shutdown() )
    {
        return -1;
    }

    frameGrabberT::m_width    = m_modevalSz;
    frameGrabberT::m_height   = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;
    m_output.resize( m_modevalSz );
    return 0;
}

float modalFilter::fps()
{
    return m_fps.load();
}

int modalFilter::startAcquisition()
{
    return 0;
}

int modalFilter::acquireAndCheckValid()
{
    std::unique_lock<std::mutex> lock( m_filtMutex );
    m_ready.wait_for( lock,
                      std::chrono::seconds( 1 ),
                      [&]
                      {
                          return m_pendingCount > 0 || m_zeroRequested || shutdown() || frameGrabberT::m_reconfig;
                      } );

    if( shutdown() || frameGrabberT::m_reconfig || m_modevalSz == 0 || m_output.size() != m_modevalSz )
    {
        return 1;
    }

    if( m_zeroRequested )
    {
        m_zeroRequested   = false;
        m_feedbackPending = false;
        resetHistory();
        std::fill( m_command.begin(), m_command.end(), 0 );
        clock_gettime( CLOCK_ISIO, &this->m_currImageTimestamp );
    }
    else
    {
        if( !m_loop || m_pendingCount == 0 )
        {
            return 1;
        }

        const auto &sample = m_pending[m_head];
        m_head             = ( m_head + 1 ) % m_pending.size();
        --m_pendingCount;

        if( m_haveCount && sample.count == m_lastCount )
        {
            return 1;
        }

        filter( sample );
        frameGrabberT::m_currImageTimestamp = sample.atime;
    }

    std::copy( m_command.begin(), m_command.end(), m_output.begin() );
    return 0;
}

int modalFilter::loadImageIntoStream( void *dest )
{
    std::memcpy( dest, m_output.data(), m_output.size() * sizeof( float ) );
    return 0;
}

int modalFilter::reconfig()
{
    return 0;
}

int modalFilter::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int modalFilter::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

INDI_SETCALLBACK_DEFN( modalFilter, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) )
    {
        const float received = ipRecv[m_fpsElement].get<float>();
        if( mx::math::isFinite( received ) && received > 0 && std::abs( received - m_fps.load() ) > m_fpsTol )
        {
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_filtMutex );
                m_fps = received;
                resetHistory();
                frameGrabberT::m_reconfig = true;
            }
            m_ready.notify_all();
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_loop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loop, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        const bool                  close = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
        std::lock_guard<std::mutex> lock( m_filtMutex );
        if( close != m_loop )
        {
            m_loop         = close;
            m_pendingCount = 0;
            m_queueGap     = false;
            resetHistory();
            if( close )
            {
                log<loop_closed>();
            }
            else
            {
                log<loop_open>();
            }
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_loopZero )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loopZero, ipRecv );

    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_filtMutex );
            m_zeroRequested = true;
        }
        m_ready.notify_one();
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_gain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gain, ipRecv );
    float target = 0;
    if( indiTargetUpdate( m_indiP_gain, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) || target < 0 )
    {
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_filtMutex );
    m_gain = target;
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_mult )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_mult, ipRecv );
    float target = 0;
    if( indiTargetUpdate( m_indiP_mult, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) || target < 0 )
    {
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_filtMutex );
    m_mult = target;
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_limit )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_limit, ipRecv );
    float target = 0;
    if( indiTargetUpdate( m_indiP_limit, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) || target < 0 )
    {
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_filtMutex );
    m_limit = target;
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_pcGain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcGain, ipRecv );
    float target = 0;
    if( indiTargetUpdate( m_indiP_pcGain, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) || target < 0 ||
        target > 1 )
    {
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_filtMutex );
    m_pcGain = target;
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_pcMult )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcMult, ipRecv );
    float target = 0;
    if( indiTargetUpdate( m_indiP_pcMult, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) || target < 0 ||
        target > 1 )
    {
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_filtMutex );
    m_pcMult = target;
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_pcOn )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcOn, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        std::lock_guard<std::mutex> lock( m_filtMutex );
        m_pcRequested = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
        if( !m_pcRequested )
        {
            m_pcActive = false;
        }
    }

    return 0;
}

} // namespace MagAOX::app

#endif // modalFilter_hpp
