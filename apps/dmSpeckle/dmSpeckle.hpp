/** \file dmSpeckle.hpp
 * \brief Generate DM speckles and tip/tilt waveforms, or play arbitrary FITS cubes.
 *
 * \ingroup dmSpeckle_files
 */

#ifndef dmSpeckle_hpp
#define dmSpeckle_hpp

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>

#include <mx/math/floatUtils.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/stringUtils.hpp>
#include <mx/sys/timeUtils.hpp>
#include <mx/sigproc/fourierModes.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "../../libMagAOX/app/dev/dmSyncopationProducer.hpp"

/** \defgroup dmSpeckle
 * \brief The DM speckle maker app
 *
 * Applies generated Fourier modes, a loaded FITS cube, or a tip/tilt waveform to a DM channel.
 * Triggered or loop-locked sparkle follows the AO loop cadence.
 *
 *
 * <a href="../handbook/operating/software/apps/dmSpeckle.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup dmSpeckle_files
 * \ingroup dmSpeckle
 */

namespace MagAOX
{
namespace app
{

#define SPARKLE 0
#define ARBCUBE 1
#define TIPTILT 2

/// The MagAO-X DM mode commander
/**
 * \ingroup dmSpeckle
 */
class dmSpeckle : public MagAOXApp<true>, public dev::telemeter<dmSpeckle>, public dev::dmSyncopationProducer<dmSpeckle>
{

    typedef float                                 realT;
    typedef dev::telemeter<dmSpeckle>             telemeterT;
    typedef dev::dmSyncopationProducer<dmSpeckle> dmSyncopationProducerT;

    friend class dev::telemeter<dmSpeckle>;
    friend class dev::dmSyncopationProducer<dmSpeckle>;
    friend class dmSpeckle_test;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_dmName; ///< The descriptive name of this dm. Default is the channel name.

    std::string m_dmChannelName; ///< The name of the DM channel to write to.

    std::string m_dmTriggerChannel; ///< The DM channel to monitor as a trigger

    float m_triggerDelay{ 0 }; ///< Delay in microseconds before a legacy triggered write.

    int m_triggerSemaphore{ 9 }; ///< The semaphore to use (default 9)

    bool m_trigger{ true }; ///< Run in trigger mode if true (default)

    realT m_separation{ 15.0 }; ///< The radial separation of the speckles (default 15.0)

    realT m_angle{ 0.0 }; ///< The angle of the speckle pattern c.c.w. from up on camsci1/2 (default 0.0)

    realT m_angleOffset{ 28.0 }; ///< The calibration offset of angle so that up on camsci1/2 is 0

    realT m_amp{ 0.01 }; ///< Speckle or tip/tilt amplitude, or a gain applied to arbitrary cube pixels.

    bool m_cross{ true }; ///< If true, also apply the cross speckles rotated by 90 degrees

    realT m_frequency{ 2000 }; ///< Timed plane rate, or physical tip/tilt frequency, in Hz.

    unsigned m_dwell{ 1 }; ///< Frame hold count for legacy operation and timed cube playback.

    int m_single{ -1 }; ///< if >= 0 a single frame is non-zero.

    std::atomic<int> m_opMode{ SPARKLE }; ///< INDI-selected waveform, sampled by the producer before polling.

    std::atomic<bool> m_loopLock{ true }; ///< Restrict managed sparkle to the main AO slot.

    std::string m_fileName; ///< Full filesystem path to the arbcube FITS file, settable at runtime via INDI.

    ///@}

    mx::improc::eigenCube<realT> m_shapes; ///< Generated speckle patterns owned by the modulation worker.

    mx::improc::eigenCube<realT> m_arbcube; ///< Unscaled FITS frames owned by the modulation worker.

    std::string m_loadedArbcubeFile; ///< Path that supplied the cached arbitrary cube.

    std::atomic<bool> m_reloadFile{
        false }; ///< Set by the load_file.request callback; consumed by the modulator thread.

    IMAGE    m_imageStream{}; ///< The only writable channel; an auxiliary source in managed mode.
    uint32_t m_width{ 0 };    ///< The width of the image
    uint32_t m_height{ 0 };   ///< The height of the image.

    IMAGE m_triggerStream{}; ///< Observation-only primary trigger for legacy operation.

    uint8_t m_dataType{ 0 }; ///< The ImageStreamIO type code.
    size_t  m_typeSize{ 0 }; ///< The size of the type, in bytes.

    bool m_opened{ false }; ///< Whether the writable stream is attached.

    bool m_triggerOpened{ false }; ///< Whether the legacy trigger stream is attached.

    bool m_restart{ false }; ///< Requests writable-stream reconnection.

    std::atomic<bool> m_modulating{ false }; ///< Enables modulation across INDI and the worker thread.

    std::atomic<bool> m_restartSp{ false }; ///< Regenerates shapes after an INDI parameter change.

    std::atomic<bool> m_syncopationZeroRequested{
        false }; ///< Queue an explicit tagged zero without mutating owned pixels.

  public:
    /// Construct the application.
    dmSpeckle();

    /// Destroy the application.
    ~dmSpeckle() noexcept;

    /// Register legacy modulation and optional provider configuration.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load configuration and request shutdown for invalid provider settings.
    virtual void loadConfig();

    /// Register INDI controls and start the modulation worker and telemeter.
    virtual int appStartup();

    /// Implementation of the FSM for dmSpeckle.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Join the worker and release streams and telemetry resources.
    virtual int appShutdown();

  protected:
    /// Read and validate an unscaled FITS cube against the attached DM channel.
    /**
     *  \returns 0 on success
     *  \returns -1 on error
     */
    int loadArbcubeFile( mx::improc::eigenCube<realT> &cube, /**< [out] Destination for FITS data. */
                         const std::string            &fileName /**< [in] Snapshot of the configured FITS path. */ );

    /// Check that the worker has loaded the configured FITS path. Caller holds m_indiMutex.
    bool arbcubeReady() const;

    /// Publish the state of the FITS load. Caller holds m_indiMutex.
    void setFileState( pcf::IndiProperty::PropertyStateType fileState /**< [in] Load status to publish. */ );

    /// Load a requested or missing cube on the worker, releasing m_indiMutex during FITS I/O.
    void handleFileReload();

    /// Stop modulation and publish a failed start or restart. Caller holds m_indiMutex.
    void modulationFailed();

    /// Name of an operating mode as used in configuration and the mode INDI property.
    static std::string opModeName( int opMode /**< [in] SPARKLE, ARBCUBE, or TIPTILT. */ );

    /// Prepare generated speckles when selected and publish the active settings. Caller holds m_indiMutex.
    void generateSpeckles();

    /// Validate the optional provider mode without changing legacy channel semantics.
    int validateSyncopationConfig();

    /// Fill a circular tip/tilt vibration with each axis at the requested physical frequency.
    static void fillTipTilt( realT   *pixels,    /**< [out] Column-major destination pixels. */
                             uint32_t width,     /**< [in] Number of rows. */
                             uint32_t height,    /**< [in] Number of columns. */
                             realT    amplitude, /**< [in] Edge amplitude of each axis. */
                             realT    frequency, /**< [in] Actual vibration frequency in Hz. */
                             int64_t  targetNs /**< [in] CLOCK_MONOTONIC command target time. */ );

    /// Return whether the managed producer should submit only for the main AO slot.
    bool syncopationMainOnly() const;

    /// Build a sparkle plane from the main cycle, a timed waveform sample, or an explicit zero.
    int prepareSyncopationCommand(
        mx::improc::eigenImage<realT> &command /**< [out] Preallocated full source command. */,
        const synco::Request          &slot /**< [in] Advertised main cycle and monotonic target. */,
        bool zero /**< [in] Build an explicitly requested zero rather than a modulation sample. */ );

    /// Follow scheduler cutoffs without independently triggering a main DM channel.
    void syncopationThreadExec();

    /** \name Modulator Thread
     * This thread sends the signal to the dm at the prescribed frequency
     *
     * @{
     */
    int m_modThreadPrio{ 60 }; ///< Priority of the modulator thread, should normally be > 00.

    std::string m_modThreadCpuset; ///< The cpuset for the modulator thread.

    std::thread m_modThread; ///< A separate thread for the modulation

    bool m_modThreadInit{ true }; ///< Holds the worker until the thread starter completes initialization.

    pid_t m_modThreadID{ 0 }; ///< Modulate thread PID.

    pcf::IndiProperty m_modThreadProp; ///< The property to hold the modulator thread details.

    /// Thread starter, called by modThreadStart on thread construction.  Calls modThreadExec.
    static void modThreadStart( dmSpeckle *d /**< [in] a pointer to a dmSpeckle instance (normally this) */ );

    /// Run the legacy modulation loop or dispatch to the managed provider.
    void modThreadExec();

    /// Finish a legacy run and clear its DM contribution.
    void finishLegacyModulation( bool trigger /**< [in] Whether the finished run used a legacy trigger. */ );

    ///@}

    // INDI:
  protected:
    // declare our properties
    pcf::IndiProperty m_indiP_dm; ///< INDI dm control/status.

    pcf::IndiProperty m_indiP_trigger; ///< INDI trigger control/status.

    pcf::IndiProperty m_indiP_loopLock; ///< INDI cadence selection for managed sparkle.

    pcf::IndiProperty m_indiP_delay; ///< INDI delay control/status.

    pcf::IndiProperty m_indiP_separation; ///< INDI separation control/status.

    pcf::IndiProperty m_indiP_angle; ///< INDI angle control/status.

    pcf::IndiProperty m_indiP_amp; ///< INDI amp control/status.

    pcf::IndiProperty m_indiP_cross; ///< INDI cross control/status.

    pcf::IndiProperty m_indiP_frequency; ///< INDI frequency control/status.

    pcf::IndiProperty m_indiP_dwell; ///< INDI dwell control/status.

    pcf::IndiProperty m_indiP_single; ///< INDI single control/status.

    pcf::IndiProperty m_indiP_modulating; ///< INDI modulating control/status.

    pcf::IndiProperty m_indiP_zero; ///< INDI zero control/status.

    pcf::IndiProperty m_indiP_mode; ///< Selection switch for SPARKLE / ARBCUBE, plus TIPTILT when syncopated.

    pcf::IndiProperty m_indiP_file; ///< Configured FITS path and load status.

    pcf::IndiProperty m_indiP_loadFile; ///< Request switch that triggers an arbcube reload.

    pcf::IndiProperty m_indiP_nFrames; ///< Read-only count of available playback frames.

  public:
    /// Handle changes to the trigger property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_trigger );

    /// Change managed sparkle between loop-locked and frequency-driven playback.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_loopLock );

    /// Handle changes to the delay property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_delay );

    /// Handle changes to the separation property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_separation );

    /// Handle changes to the angle property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_angle );

    /// Handle changes to the cross property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_cross );

    /// Handle changes to the amp property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_amp );

    /// Handle changes to the frequency property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_frequency );

    /// Handle changes to the dwell property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_dwell );

    /// Handle changes to the single property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_single );

    /// Handle changes to the modulating property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_modulating );

    /// Handle changes to the zero property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_zero );

    /// Select a modulation mode.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_mode );

    /// Stage a FITS path for the next load.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_file );

    /// Queue a FITS load on the modulation worker.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_loadFile );

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether modulation telemetry is due.
    int checkRecordTimes();

    /// Record the modulation telemetry requested by the telemeter.
    int recordTelem( const telem_dmspeck *unused /**< [in] Telemetry type tag. */ );

    /// Record modulation state when it changes or recording is forced.
    int recordDmSpeck( bool force = false /**< [in] Record even when the state is unchanged. */ );

    ///@}
};

dmSpeckle::dmSpeckle() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline dmSpeckle::~dmSpeckle() noexcept = default;

void dmSpeckle::setupConfig()
{
    TELEMETER_SETUP_CONFIG( config );
    DMSYNCOPATIONPRODUCER_SETUP_CONFIG( config );
    config.add( "dm.channelName",
                "",
                "dm.channelName",
                argType::Required,
                "dm",
                "channelName",
                false,
                "string",
                "The name of the DM channel to write to." );
    config.add( "dm.triggerChannel",
                "",
                "dm.triggerChannel",
                argType::Required,
                "dm",
                "triggerChannel",
                false,
                "string",
                "The name of the DM channel to trigger on." );
    config.add( "dm.triggerSemaphore",
                "",
                "dm.triggerSemaphore",
                argType::Required,
                "dm",
                "triggerSemaphore",
                false,
                "int",
                "The semaphore to use (default 9)." );
    config.add( "dm.trigger",
                "",
                "dm.trigger",
                argType::True,
                "dm",
                "trigger",
                false,
                "bool",
                "Run in trigger mode if true (default)." );
    config.add( "dm.triggerDelay",
                "",
                "dm.triggerDelay",
                argType::Required,
                "dm",
                "triggerDelay",
                false,
                "float",
                "Delay to apply to the trigger." );

    config.add( "dm.separation",
                "",
                "dm.separation",
                argType::Required,
                "dm",
                "separation",
                false,
                "float",
                "The radial separation of the speckles (default 15.0)." );
    config.add( "dm.angle",
                "",
                "dm.angle",
                argType::Required,
                "dm",
                "angle",
                false,
                "float",
                "The angle of the speckle pattern c.c.w. from up on camsci1/2 (default 0.0)." );
    config.add( "dm.angleOffset",
                "",
                "dm.angleOffset",
                argType::Required,
                "dm",
                "angleOffset",
                false,
                "float",
                "The calibration offset of angle so that up on camsci1/2 is 0." );
    config.add( "dm.amp",
                "",
                "dm.amp",
                argType::Required,
                "dm",
                "amp",
                false,
                "float",
                "The speckle amplitude on the DM (default 0.01)." );
    config.add( "dm.cross",
                "",
                "dm.cross",
                argType::True,
                "dm",
                "cross",
                false,
                "bool",
                "If true, also apply the cross speckles rotated by 90 degrees." );

    config.add( "dm.frequency",
                "",
                "dm.frequency",
                argType::Required,
                "dm",
                "frequency",
                false,
                "float",
                "Timed playback rate or managed tiptilt frequency in Hz (default 2000)." );

    config.add( "dm.dwell",
                "",
                "dm.dwell",
                argType::True,
                "dm",
                "dwell",
                false,
                "int",
                "The dwell time for each speckle, or for how many frames it is held. Default=1." );

    config.add( "syncopation.loopLock",
                "",
                "syncopation.loopLock",
                argType::Required,
                "syncopation",
                "loopLock",
                false,
                "bool",
                "Lock managed sparkle to the main AO slot (default true)." );

    config.add( "modulator.threadPrio",
                "",
                "modulator.threadPrio",
                argType::Required,
                "modulator",
                "threadPrio",
                false,
                "int",
                "The real-time priority of the modulator thread." );

    config.add( "modulator.cpuset",
                "",
                "modulator.cpuset",
                argType::Required,
                "modulator",
                "cpuset",
                false,
                "string",
                "The cpuset to assign the modulator thread to." );
    config.add( "modulator.opMode",
                "",
                "modulator.opMode",
                argType::Required,
                "modulator",
                "opMode",
                false,
                "string",
                "Mode: sparkle (default), arbcube, or managed tiptilt; arbcube requires fileName." );
    config.add( "modulator.fileName",
                "",
                "modulator.fileName",
                argType::Required,
                "modulator",
                "fileName",
                false,
                "string",
                "Full filesystem path to the FITS cube to modulate the DM with.  "
                "Only used if opMode==arbcube; can be changed at runtime via the file INDI property." );
}

int dmSpeckle::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_dmChannelName, "dm.channelName" );
    DMSYNCOPATIONPRODUCER_LOAD_CONFIG( _config );

    bool loopLock = true;
    _config( loopLock, "syncopation.loopLock" );
    m_loopLock = loopLock;

    m_dmName = m_dmChannelName;
    _config( m_dmName, "dm.name" );

    _config( m_dmTriggerChannel, "dm.triggerChannel" );
    _config( m_triggerSemaphore, "dm.triggerSemaphore" );

    if( _config.isSet( "dm.trigger" ) )
    {
        _config( m_trigger, "dm.trigger" );
    }

    _config( m_triggerDelay, "dm.triggerDelay" );

    std::string opMode = "sparkle";
    _config( opMode, "modulator.opMode" );

    if( opMode == "arbcube" )
    {
        m_opMode = ARBCUBE;
    }
    else if( opMode == "tiptilt" )
    {
        m_opMode = TIPTILT;
    }
    else
    {
        m_opMode = SPARKLE;
    }
    _config( m_fileName, "modulator.fileName" );
    _config( m_separation, "dm.separation" );
    _config( m_angle, "dm.angle" );
    _config( m_angleOffset, "dm.angleOffset" );

    _config( m_amp, "dm.amp" );

    if( _config.isSet( "dm.cross" ) )
    {
        _config( m_cross, "dm.cross" );
    }

    _config( m_frequency, "dm.frequency" );
    _config( m_dwell, "dm.dwell" );

    _config( m_modThreadPrio, "modulator.threadPrio" );
    _config( m_modThreadCpuset, "modulator.cpuset" );

    TELEMETER_LOAD_CONFIG( _config );
    return validateSyncopationConfig();
}

void dmSpeckle::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, "Invalid modulation/provider configuration" } );
        m_shutdown = true;
    }
}

int dmSpeckle::validateSyncopationConfig()
{
    if( m_syncopationTiming.empty() )
    {
        return m_opMode == TIPTILT ? -1 : 0;
    }
    constexpr size_t suffixLength = 4;
    if( m_dmChannelName.size() <= suffixLength ||
        m_dmChannelName.compare( m_dmChannelName.size() - suffixLength, suffixLength, "_aux" ) != 0 ||
        m_syncopationTiming == m_dmChannelName || !mx::math::isFinite( m_writeLeadUs ) || m_writeLeadUs <= 0 ||
        m_writeLeadUs > 1e6 || !mx::math::isFinite( m_frequency ) || m_frequency < 0 || !mx::math::isFinite( m_amp ) )
    {
        return -1;
    }
    m_trigger = false;
    return 0;
}

void dmSpeckle::fillTipTilt(
    realT *pixels, uint32_t width, uint32_t height, realT amplitude, realT frequency, int64_t targetNs )
{
    long double cycles = std::fmod( static_cast<long double>( targetNs ) * frequency / 1e9L, 1.0L );
    double      phase  = 2 * mx::math::pi<double>() * static_cast<double>( cycles );
    realT       tip    = amplitude * std::cos( phase );
    realT       tilt   = amplitude * std::sin( phase );
    for( uint32_t col = 0; col < height; ++col )
    {
        realT y = height > 1 ? 2.0 * col / ( height - 1 ) - 1 : 0;
        for( uint32_t row = 0; row < width; ++row )
        {
            realT x                   = width > 1 ? 2.0 * row / ( width - 1 ) - 1 : 0;
            pixels[col * width + row] = tip * x + tilt * y;
        }
    }
}

bool dmSpeckle::syncopationMainOnly() const
{
    return m_opMode.load( std::memory_order_relaxed ) == SPARKLE && m_loopLock.load( std::memory_order_relaxed );
}

int dmSpeckle::prepareSyncopationCommand( mx::improc::eigenImage<realT> &command,
                                          const synco::Request          &slot,
                                          bool                           zero )
{
    if( zero )
    {
        command.setZero();
        return 0;
    }

    const int mode = m_opMode;

    if( mode == TIPTILT )
    {
        if( m_restartSp )
        {
            m_restartSp = false;
        }

        fillTipTilt( command.data(), m_width, m_height, m_amp, m_frequency, slot.targetNs );
        return 0;
    }

    const bool mainOnly = syncopationMainOnly();

    if( mainOnly && ( slot.step != 0 || slot.cycle < 0 ) )
    {
        return -1;
    }

    const auto &shapes = mode == ARBCUBE ? m_arbcube : m_shapes;

    if( m_restartSp || ( mode == SPARKLE && shapes.planes() == 0 ) )
    {
        m_restartSp = false;
        generateSpeckles();
    }

    if( shapes.planes() == 0 )
    {
        return -1;
    }

    int plane;

    if( mainOnly )
    {
        plane = static_cast<int>( slot.cycle % shapes.planes() );
    }
    else
    {
        const long double index =
            std::floor( static_cast<long double>( slot.targetNs ) * m_frequency / ( 1e9L * std::max( 1U, m_dwell ) ) );
        plane = static_cast<int>( std::fmod( index, shapes.planes() ) );
    }

    if( mode == ARBCUBE )
    {
        command = shapes.image( plane ) * m_amp;
    }
    else
    {
        command = shapes.image( plane );
    }

    return 0;
}

void dmSpeckle::syncopationThreadExec()
{
    mx::improc::eigenImage<realT> command( m_width, m_height );

    while( !m_shutdown )
    {
        // Finish any FITS I/O before selecting a deadline.
        if( m_reloadFile || ( m_modulating && m_restartSp ) )
        {
            handleFileReload();
        }

        if( !m_modulating && !m_syncopationZeroRequested )
        {
            mx::sys::milliSleep( 1 );
            continue;
        }

        const bool     mainOnly = syncopationMainOnly();
        synco::Request request{};
        const int      pollResult = dmSyncopationProducerT::syncopationPoll( request, m_imageStream, mainOnly );

        if( pollResult < 0 )
        {
            mx::sys::milliSleep( 100 );
            continue;
        }

        if( pollResult == 0 )
        {
            mx::sys::nanoSleep( 10000 );
            continue;
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_indiMutex );

            if( m_shutdown || ( !m_modulating && !m_syncopationZeroRequested ) || mainOnly != syncopationMainOnly() )
            {
                continue;
            }

            const bool zero = !m_modulating && m_syncopationZeroRequested;

            if( !zero && ( m_reloadFile || ( m_opMode == ARBCUBE && m_restartSp && !arbcubeReady() ) ) )
            {
                continue;
            }

            if( prepareSyncopationCommand( command, request, zero ) < 0 )
            {
                modulationFailed();
                continue;
            }

            if( dmSyncopationProducerT::syncopationPublish( command.data(), request, m_imageStream ) && zero )
            {
                m_syncopationZeroRequested = false;
                log<text_log>( "zero submitted for syncopation" );
            }
        }
    }

    dmSyncopationProducerT::syncopationClose();
}

int dmSpeckle::appStartup()
{

    REG_INDI_NEWPROP_NOCB( m_indiP_dm, "dm", pcf::IndiProperty::Text );
    m_indiP_dm.add( pcf::IndiElement( "name" ) );
    m_indiP_dm["name"] = m_dmName;
    m_indiP_dm.add( pcf::IndiElement( "channel" ) );
    m_indiP_dm["channel"] = m_dmChannelName;
    if( !m_syncopationTiming.empty() )
    {
        m_indiP_dm.add( pcf::IndiElement( "timing" ) );
        m_indiP_dm["timing"] = m_syncopationTiming;
    }

    createStandardIndiNumber<float>( m_indiP_delay, "delay", 0, 0, 1, "%f" );
    m_indiP_delay["current"] = m_triggerDelay;
    m_indiP_delay["target"]  = m_triggerDelay;
    registerIndiPropertyNew( m_indiP_delay, INDI_NEWCALLBACK( m_indiP_delay ) );

    // The sparkle-specific properties are always registered so users can set them while in
    // arbcube mode; they take effect when mode switches back to sparkle.
    createStandardIndiNumber<float>( m_indiP_separation, "separation", 0, 0, 100, "%f" );
    m_indiP_separation["current"] = m_separation;
    m_indiP_separation["target"]  = m_separation;
    registerIndiPropertyNew( m_indiP_separation, INDI_NEWCALLBACK( m_indiP_separation ) );

    createStandardIndiNumber<float>( m_indiP_angle, "angle", 0, 0, 100, "%f" );
    m_indiP_angle["current"] = m_angle;
    m_indiP_angle["target"]  = m_angle;
    registerIndiPropertyNew( m_indiP_angle, INDI_NEWCALLBACK( m_indiP_angle ) );

    createStandardIndiToggleSw( m_indiP_cross, "cross" );
    if( registerIndiPropertyNew( m_indiP_cross, INDI_NEWCALLBACK( m_indiP_cross ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
    if( m_cross )
    {
        m_indiP_cross["toggle"] = pcf::IndiElement::On;
    }
    else
    {
        m_indiP_cross["toggle"] = pcf::IndiElement::Off;
    }

    createStandardIndiNumber<float>( m_indiP_amp, "amp", -1, 1, 0, "%f" );
    m_indiP_amp["current"] = m_amp;
    m_indiP_amp["target"]  = m_amp;
    registerIndiPropertyNew( m_indiP_amp, INDI_NEWCALLBACK( m_indiP_amp ) );

    createStandardIndiNumber<float>( m_indiP_frequency, "frequency", 0, 10000, 0, "%f" );
    m_indiP_frequency["current"] = m_frequency;
    m_indiP_frequency["target"]  = m_frequency;
    registerIndiPropertyNew( m_indiP_frequency, INDI_NEWCALLBACK( m_indiP_frequency ) );

    createStandardIndiToggleSw( m_indiP_trigger, "trigger" );
    if( registerIndiPropertyNew( m_indiP_trigger, INDI_NEWCALLBACK( m_indiP_trigger ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
    if( m_trigger )
    {
        m_indiP_trigger["toggle"] = pcf::IndiElement::On;
    }
    else
    {
        m_indiP_trigger["toggle"] = pcf::IndiElement::Off;
    }

    if( !m_syncopationTiming.empty() )
    {
        CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_loopLock, "loop_lock" );
        m_indiP_loopLock["toggle"] = m_loopLock ? pcf::IndiElement::On : pcf::IndiElement::Off;
    }

    createStandardIndiNumber<int>( m_indiP_dwell, "dwell", 1, 100, 1, "%d" );
    m_indiP_dwell["current"] = m_dwell;
    m_indiP_dwell["target"]  = m_dwell;
    registerIndiPropertyNew( m_indiP_dwell, INDI_NEWCALLBACK( m_indiP_dwell ) );

    createStandardIndiNumber<int>( m_indiP_single, "single", -1, 3, 1, "%d" );
    m_indiP_single["current"] = m_single;
    m_indiP_single["target"]  = m_single;
    registerIndiPropertyNew( m_indiP_single, INDI_NEWCALLBACK( m_indiP_single ) );

    createStandardIndiToggleSw( m_indiP_modulating, "modulating" );
    if( registerIndiPropertyNew( m_indiP_modulating, INDI_NEWCALLBACK( m_indiP_modulating ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiRequestSw( m_indiP_zero, "zero" );
    if( registerIndiPropertyNew( m_indiP_zero, INDI_NEWCALLBACK( m_indiP_zero ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    // tiptilt is only valid with a syncopation timing source, see validateSyncopationConfig().
    std::vector<std::string> modeNames = { opModeName( SPARKLE ), opModeName( ARBCUBE ) };
    if( !m_syncopationTiming.empty() )
    {
        modeNames.push_back( opModeName( TIPTILT ) );
    }
    if( createStandardIndiSelectionSw( m_indiP_mode, "mode", modeNames ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
    for( const auto &modeName : modeNames )
    {
        m_indiP_mode[modeName] = ( modeName == opModeName( m_opMode ) ) ? pcf::IndiElement::On : pcf::IndiElement::Off;
    }
    if( registerIndiPropertyNew( m_indiP_mode, INDI_NEWCALLBACK( m_indiP_mode ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( createStandardIndiText( m_indiP_file, "file" ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
    m_indiP_file["current"] = m_fileName;
    m_indiP_file["target"]  = m_fileName;
    if( registerIndiPropertyNew( m_indiP_file, INDI_NEWCALLBACK( m_indiP_file ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiRequestSw( m_indiP_loadFile, "load_file" );
    if( registerIndiPropertyNew( m_indiP_loadFile, INDI_NEWCALLBACK( m_indiP_loadFile ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    REG_INDI_NEWPROP_NOCB( m_indiP_nFrames, "n_frames", pcf::IndiProperty::Number );
    m_indiP_nFrames.add( pcf::IndiElement( "current" ) );
    m_indiP_nFrames["current"] = 0;

    if( threadStart( m_modThread,
                     m_modThreadInit,
                     m_modThreadID,
                     m_modThreadProp,
                     m_modThreadPrio,
                     m_modThreadCpuset,
                     "modulator",
                     this,
                     modThreadStart ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        return -1;
    }

    TELEMETER_APP_STARTUP;

    state( stateCodes::NOTCONNECTED );

    return 0;
}

int dmSpeckle::appLogic()
{
    if( state() == stateCodes::NOTCONNECTED )
    {
        m_opened  = false;
        m_restart = false; // Set this up front, since we're about to restart.

        if( ImageStreamIO_openIm( &m_imageStream, m_dmChannelName.c_str() ) == 0 )
        {
            if( m_imageStream.md[0].sem < 10 ) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
            {
                ImageStreamIO_closeIm( &m_imageStream );
            }
            else
            {
                m_opened = true;
            }
        }

        // Only bother to try if previous worked and we have a spec
        if( m_opened == true && m_syncopationTiming.empty() && m_dmTriggerChannel != "" )
        {
            if( ImageStreamIO_openIm( &m_triggerStream, m_dmTriggerChannel.c_str() ) == 0 )
            {
                if( m_triggerStream.md[0].sem <
                    10 ) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
                {
                    ImageStreamIO_closeIm( &m_triggerStream );
                    m_opened = false;
                }
                else
                {
                    m_triggerOpened = true;
                }
            }
        }

        if( m_opened )
        {
            state( stateCodes::CONNECTED );
        }
    }

    if( state() == stateCodes::CONNECTED )
    {
        m_dataType = m_imageStream.md[0].datatype;
        m_typeSize = ImageStreamIO_typesize( m_dataType );
        m_width    = m_imageStream.md[0].size[0];
        m_height   = m_imageStream.md[0].size[1];

        if( !m_syncopationTiming.empty() && ( m_imageStream.md->naxis != 2 || m_width == 0 || m_height == 0 ) )
        {
            return log<text_log, -1>( "Auxiliary source must be a nonempty 2-D image.", logPrio::LOG_CRITICAL );
        }

        if( m_dataType != _DATATYPE_FLOAT )
        {
            return log<text_log, -1>( "Data type of DM channel is not float.", logPrio::LOG_CRITICAL );
        }

        if( m_typeSize != sizeof( realT ) )
        {
            return log<text_log, -1>( "Type-size mismatch, realT is not float.", logPrio::LOG_CRITICAL );
        }

        state( stateCodes::READY );
    }

    TELEMETER_APP_LOGIC;

    return 0;
}

int dmSpeckle::appShutdown()
{
    if( m_modThread.joinable() )
    {
        try
        {
            m_modThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }

    if( m_triggerOpened )
    {
        ImageStreamIO_closeIm( &m_triggerStream );
        m_triggerOpened = false;
    }
    if( m_opened )
    {
        ImageStreamIO_closeIm( &m_imageStream );
        m_opened = false;
    }
    DMSYNCOPATIONPRODUCER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int dmSpeckle::loadArbcubeFile( mx::improc::eigenCube<realT> &cube, const std::string &fileName )
{
    if( fileName.empty() )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "arbcube file name is empty; nothing to load" } );
    }

    mx::fits::fitsFile<realT> ff;
    mx::error_t               errc = ff.read( cube, fileName );

    if( errc != mx::error_t::noerror )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "could not read arbcube FITS file: " + fileName } );
    }

    if( cube.rows() != m_width || cube.cols() != m_height || cube.planes() == 0 )
    {
        return log<software_error, -1>(
            { __FILE__,
              __LINE__,
              "arbcube must contain nonempty DM frames: expected " + std::to_string( m_width ) + "x" +
                  std::to_string( m_height ) + ", loaded " + std::to_string( cube.rows() ) + "x" +
                  std::to_string( cube.cols() ) + "x" + std::to_string( cube.planes() ) + " from " + fileName } );
    }

    return 0;
}

inline bool dmSpeckle::arbcubeReady() const
{
    return m_arbcube.planes() > 0 && m_loadedArbcubeFile == m_fileName;
}

inline void dmSpeckle::setFileState( pcf::IndiProperty::PropertyStateType fileState )
{
    if( m_indiP_file.getState() == fileState )
    {
        return;
    }

    m_indiP_file.setState( fileState );
    m_indiP_file.setTimeStamp( pcf::TimeStamp() );
    if( m_indiDriver )
    {
        m_indiDriver->sendSetProperty( m_indiP_file );
    }
}

void dmSpeckle::handleFileReload()
{
    std::string fileName;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_indiMutex );

        if( !m_reloadFile.exchange( false ) && ( !m_modulating || m_opMode != ARBCUBE || arbcubeReady() ) )
        {
            return;
        }

        fileName = m_fileName;
        setFileState( INDI_BUSY );
    }

    mx::improc::eigenCube<realT> cube;
    const int                    rv = loadArbcubeFile( cube, fileName );

    std::lock_guard<std::mutex> lock( m_indiMutex );

    if( m_fileName != fileName || m_reloadFile )
    {
        return;
    }

    if( rv < 0 )
    {
        m_arbcube.clear();
        m_loadedArbcubeFile.clear();
        setFileState( INDI_ALERT );

        if( m_modulating && m_opMode == ARBCUBE )
        {
            modulationFailed();
        }
    }
    else
    {
        m_arbcube           = std::move( cube );
        m_loadedArbcubeFile = fileName;
        log<text_log>( "loaded arbcube FITS file: " + fileName + " (" + std::to_string( m_arbcube.planes() ) +
                           " planes)",
                       logPrio::LOG_NOTICE );
        setFileState( INDI_OK );
    }

    updateIfChanged( m_indiP_nFrames, "current", static_cast<int>( m_arbcube.planes() ) );
}

inline void dmSpeckle::modulationFailed()
{
    m_modulating = false;
    updateSwitchIfChanged( m_indiP_modulating, "toggle", pcf::IndiElement::Off, INDI_ALERT );
}

std::string dmSpeckle::opModeName( int opMode )
{
    if( opMode == ARBCUBE )
    {
        return "arbcube";
    }
    if( opMode == TIPTILT )
    {
        return "tiptilt";
    }
    return "sparkle";
}

void dmSpeckle::generateSpeckles()
{
    if( m_opMode == SPARKLE )
    {
        mx::improc::eigenImage<realT> onesp, onespC;
        onesp.resize( m_width, m_height );
        onespC.resize( m_width, m_height );

        m_shapes.resize( m_width, m_height, 4 );

        realT m = m_separation * cos( mx::math::dtor<realT>( -1 * m_angle + m_angleOffset ) );
        realT n = m_separation * sin( mx::math::dtor<realT>( -1 * m_angle + m_angleOffset ) );

        mx::sigproc::makeFourierMode( m_shapes.image( 0 ), m, n, 1 );

        if( m_cross )
        {
            onesp = m_shapes.image( 0 );
            mx::sigproc::makeFourierMode( m_shapes.image( 0 ), -n, m, 1 );
            m_shapes.image( 0 ) += onesp;
        }

        m_shapes.image( 0 ) *= m_amp;
        m_shapes.image( 1 ) = -1 * m_shapes.image( 0 );

        mx::sigproc::makeFourierMode( m_shapes.image( 2 ), m, n, -1 );

        if( m_cross )
        {
            onesp = m_shapes.image( 2 );
            mx::sigproc::makeFourierMode( m_shapes.image( 2 ), -n, m, -1 );
            m_shapes.image( 2 ) += onesp;
        }

        m_shapes.image( 2 ) *= m_amp;
        m_shapes.image( 3 ) = -m_shapes.image( 2 );

        mx::fits::fitsFile<realT> ff;
        ff.write( "/tmp/specks.fits", m_shapes );

        if( m_single >= 0 )
        {
            for( int pp = 0; pp < m_shapes.planes(); ++pp )
            {
                if( pp != m_single )
                {
                    m_shapes.image( pp ) *= 0;
                }
            }
        }
        updateIfChanged( m_indiP_separation, "current", m_separation );
        updateIfChanged( m_indiP_angle, "current", m_angle );
    }

    const int nFrames = m_opMode == ARBCUBE ? m_arbcube.planes() : m_shapes.planes();
    updateIfChanged( m_indiP_nFrames, "current", nFrames );
    updateIfChanged( m_indiP_delay, "current", m_triggerDelay );
    updateIfChanged( m_indiP_dwell, "current", m_dwell );
    updateIfChanged( m_indiP_single, "current", m_single );
}

inline void dmSpeckle::modThreadStart( dmSpeckle *d )
{
    d->modThreadExec();
}

inline void dmSpeckle::modThreadExec()
{
    m_modThreadID = syscall( SYS_gettid );

    // Wait for the thread starter and the DM channel.
    while( ( m_modThreadInit == true || state() != stateCodes::READY ) && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    if( !m_syncopationTiming.empty() )
    {
        syncopationThreadExec();
        return;
    }

    mx::improc::eigenImage<realT> stagedFrame( m_width, m_height );
    bool                          legacyActive = false;
    bool                          runTriggered = false;

    while( !m_shutdown )
    {
        handleFileReload();

        if( !m_modulating )
        {
            if( legacyActive )
            {
                finishLegacyModulation( runTriggered );
                legacyActive = false;
            }

            mx::sys::milliSleep( 500 );
            continue;
        }

        if( m_modulating && !m_shutdown )
        {
            bool     arbcube;
            bool     trigger;
            realT    amp;
            double   triggerDelay;
            unsigned dwell;
            int64_t  freqNsec;

            { //mutex scope
                std::lock_guard<std::mutex> lock( m_indiMutex );

                arbcube = m_opMode == ARBCUBE;

                if( !m_modulating || m_reloadFile || ( arbcube && !arbcubeReady() ) )
                {
                    continue;
                }

                if( m_dmTriggerChannel.empty() )
                {
                    m_trigger = false;
                    updateSwitchIfChanged( m_indiP_trigger, "toggle", pcf::IndiElement::Off, INDI_IDLE );
                }

                trigger             = m_trigger;
                amp                 = m_amp;
                triggerDelay        = m_triggerDelay / 1e6;
                dwell               = std::max( 1U, m_dwell );
                const double period = trigger || m_frequency <= 0 ? 0 : 1e9 / m_frequency;

                if( !trigger && ( !mx::math::isFinite( period ) || period < 1 ||
                                  period >= static_cast<double>( std::numeric_limits<int64_t>::max() ) ) )
                {
                    log<software_error>( { __FILE__, __LINE__, "Invalid frequency for untriggered modulation" } );
                    modulationFailed();
                    continue;
                }

                freqNsec    = static_cast<int64_t>( period );
                m_restartSp = false;
                generateSpeckles();

                log<text_log>( "started modulating", logPrio::LOG_NOTICE );
                log<telem_dmspeck>( { m_modulating,
                                      m_trigger,
                                      m_frequency,
                                      std::vector<float>( { m_separation } ),
                                      std::vector<float>( { m_angle } ),
                                      std::vector<float>( { m_amp } ),
                                      std::vector<bool>( { m_cross } ) },
                                    logPrio::LOG_INFO );
                recordDmSpeck( true );
            }

            legacyActive = true;
            runTriggered = trigger;

            const auto &shapes = arbcube ? m_arbcube : m_shapes;
            int64_t     dnsec  = 0;
            int         idx    = 0;

            timespec modstart;
            timespec currtime;

            bool   triggered = false;
            sem_t *sem       = nullptr;

            if( trigger )
            {
                ImageStreamIO_semflush( &m_triggerStream, m_triggerSemaphore );

                sem = m_triggerStream.semptr[m_triggerSemaphore]; ///< The semaphore to monitor for new image data
            }

            clock_gettime( CLOCK_REALTIME, &modstart );

            unsigned dwelled = 0;
            double   t0, t1;

            while( m_modulating && !m_restartSp && !m_shutdown )
            {
                if( trigger )
                {
                    timespec ts;

                    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
                    {
                        log<software_critical>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
                        return;
                    }

                    ts.tv_sec += 1;

                    if( sem_timedwait( sem, &ts ) == 0 )
                    {
                        t0 = mx::sys::get_curr_time();
                        t1 = t0;

                        while( t1 - t0 < triggerDelay )
                        {
                            double dt = ( 1e8 ) * ( triggerDelay -
                                                    ( t1 - t0 ) ); // This is 0.1 times remaining time, but in nanosecs
                            if( dt <= 0 )
                            {
                                break;
                            }
                            mx::sys::nanoSleep( dt );
                            t1 = mx::sys::get_curr_time();
                        }

                        triggered = true;
                        clock_gettime( CLOCK_REALTIME, &currtime );
                    }
                    else
                    {
                        triggered = false;

                        // Check for why we timed out
                        if( errno == EINTR )
                        {
                            break; // This indicates signal interrupted us, time to restart or shutdown, loop will exit
                                   // normally if flags set.
                        }

                        // ETIMEDOUT just means we should wait more.
                        // Otherwise, report an error.
                        if( errno != ETIMEDOUT )
                        {
                            log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
                            break;
                        }
                    }
                }
                else
                {
                    mx::sys::nanoSleep( std::clamp<int64_t>( dnsec / 2, 0, 100000000 ) );
                    clock_gettime( CLOCK_REALTIME, &currtime );

                    dnsec =
                        ( currtime.tv_sec - modstart.tv_sec ) * 1000000000 + ( currtime.tv_nsec - modstart.tv_nsec );
                    triggered = false;
                }

                if( !m_modulating || m_restartSp || m_shutdown )
                {
                    break;
                }

                if( dwelled < dwell - 1 )
                {
                    ++dwelled;
                }
                else if( trigger ? triggered : dnsec >= freqNsec )
                {
                    dwelled                 = 0;
                    m_imageStream.md->write = 1;

                    auto         frame  = shapes.image( idx );
                    const realT *pixels = frame.data();
                    if( arbcube )
                    {
                        stagedFrame = frame * amp;
                        pixels      = stagedFrame.data();
                    }

                    memcpy( m_imageStream.array.raw, pixels, m_width * m_height * m_typeSize );

                    m_imageStream.md->atime     = currtime;
                    m_imageStream.md->writetime = currtime;

                    if( !trigger || triggerDelay > 0 )
                    {
                        m_imageStream.md->cnt0++;
                    }

                    m_imageStream.md->write = 0;
                    ImageStreamIO_sempost( &m_imageStream, -1 );

                    ++idx;
                    if( idx >= shapes.planes() )
                    {
                        idx = 0;
                    }

                    if( !trigger )
                    {
                        modstart.tv_sec += freqNsec / 1000000000;
                        modstart.tv_nsec += freqNsec % 1000000000;
                        if( modstart.tv_nsec >= 1000000000 )
                        {
                            modstart.tv_nsec -= 1000000000;
                            modstart.tv_sec += 1;
                        }
                        dnsec = freqNsec;
                    }
                }
            }
            if( m_restartSp && m_modulating )
            {
                continue;
            }

            finishLegacyModulation( runTriggered );
            legacyActive = false;
        }
    }

    if( legacyActive )
    {
        finishLegacyModulation( runTriggered );
    }
}

inline void dmSpeckle::finishLegacyModulation( bool trigger )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    recordDmSpeck( true );
    log<text_log>( "stopped modulating", logPrio::LOG_NOTICE );

    timespec currtime;
    clock_gettime( CLOCK_REALTIME, &currtime );
    m_imageStream.md->write = 1;
    memset( m_imageStream.array.raw, 0, m_width * m_height * m_typeSize );
    m_imageStream.md->atime     = currtime;
    m_imageStream.md->writetime = currtime;

    if( !trigger )
    {
        m_imageStream.md->cnt0++;
    }

    m_imageStream.md->write = 0;
    ImageStreamIO_sempost( &m_imageStream, -1 );
    log<text_log>( "zeroed" );
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_trigger )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_trigger.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( !m_syncopationTiming.empty() && ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        return -1;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_trigger = false;
        indi::updateSwitchIfChanged( m_indiP_trigger, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_trigger = true;
        indi::updateSwitchIfChanged( m_indiP_trigger, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
    }

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_loopLock )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loopLock, ipRecv );

    if( m_syncopationTiming.empty() || !ipRecv.find( "toggle" ) )
    {
        return -1;
    }

    const auto setting = ipRecv["toggle"].getSwitchState();

    if( setting != pcf::IndiElement::On && setting != pcf::IndiElement::Off )
    {
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_loopLock = setting == pcf::IndiElement::On;
    indi::updateSwitchIfChanged( m_indiP_loopLock, "toggle", setting, m_indiDriver, INDI_OK );

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_delay )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_delay.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float del = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        del = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        del = ipRecv["target"].get<float>();
    }

    if( del == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No requested delay" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_triggerDelay = del;
    updateIfChanged( m_indiP_delay, "target", m_triggerDelay );

    m_restartSp = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_separation )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_separation.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float sep = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        sep = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        sep = ipRecv["target"].get<float>();
    }

    if( sep == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No requested separation" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_separation = sep;
    updateIfChanged( m_indiP_separation, "target", m_separation );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_angle )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_angle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float ang = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        ang = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        ang = ipRecv["target"].get<float>();
    }

    if( ang == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No angle received" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_angle = ang;
    updateIfChanged( m_indiP_angle, "target", m_angle );

    m_restartSp = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_amp )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_amp.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float amp = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        amp = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        amp = ipRecv["target"].get<float>();
    }

    if( amp == -1000000000 || !mx::math::isFinite( amp ) )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested amp: " + std::to_string( amp ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_amp = amp;
    updatesIfChanged<float>( m_indiP_amp, { "target", "current" }, { amp, amp } );

    m_restartSp = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_cross )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_cross.createUniqueKey() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_cross = false;
        indi::updateSwitchIfChanged( m_indiP_cross, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_cross = true;
        indi::updateSwitchIfChanged( m_indiP_cross, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
    }

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_frequency )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_frequency.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float freq = -1;

    if( ipRecv.find( "current" ) )
    {
        freq = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        freq = ipRecv["target"].get<float>();
    }

    if( freq < 0 || !mx::math::isFinite( freq ) )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested frequency: " + std::to_string( freq ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_frequency = freq;
    updatesIfChanged<float>( m_indiP_frequency, { "target", "current" }, { freq, freq } );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_dwell )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_dwell.createUniqueKey() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    unsigned dwell = 0;

    if( ipRecv.find( "current" ) )
    {
        dwell = ipRecv["current"].get<unsigned>();
    }

    if( ipRecv.find( "target" ) )
    {
        dwell = ipRecv["target"].get<unsigned>();
    }

    if( dwell == 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested dwell: " + std::to_string( dwell ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_dwell = dwell;
    updateIfChanged( m_indiP_dwell, "target", m_dwell );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_single )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_single, ipRecv );

    int single = 0;

    if( ipRecv.find( "current" ) )
    {
        single = ipRecv["current"].get<int>();
    }

    if( ipRecv.find( "target" ) )
    {
        single = ipRecv["target"].get<int>();
    }

    if( single < -1 || single > 3 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested dwell: " + std::to_string( single ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_single = single;
    updateIfChanged( m_indiP_single, "target", m_single );
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_modulating )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_modulating.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_modulating = false;
        indi::updateSwitchIfChanged( m_indiP_modulating, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_modulating )
        {
            m_restartSp = true;
        }

        m_syncopationZeroRequested = false;
        m_modulating               = true;
        indi::updateSwitchIfChanged( m_indiP_modulating, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_zero )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_zero.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( m_modulating == true )
    {
        log<text_log>( "zero requested but currently modulating", logPrio::LOG_NOTICE );
        return 0;
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_opened || m_imageStream.md == nullptr )
        {
            return -1;
        }
        if( !m_syncopationTiming.empty() )
        {
            m_syncopationZeroRequested = true;
            log<text_log>( "zero queued for syncopation" );
            return 0;
        }

        m_imageStream.md->write = 1;
        memset( m_imageStream.array.raw, 0, m_width * m_height * m_typeSize );
        timespec currtime;
        clock_gettime( CLOCK_REALTIME, &currtime );
        m_imageStream.md->atime     = currtime;
        m_imageStream.md->writetime = currtime;
        m_imageStream.md->cnt0++;
        m_imageStream.md->write = 0;
        ImageStreamIO_sempost( &m_imageStream, -1 );
        log<text_log>( "zeroed" );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_mode )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_mode, ipRecv );

    // Iterate received elements to find the one newly switched On.
    // We skip elements already On in the device property because the pcf library merges
    // current device state into ipRecv when the client sends only the changed element,
    // which would otherwise cause the already-selected element to fire first.
    int newMode = -1;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }
        if( m_indiP_mode.find( elit->first ) && m_indiP_mode[elit->first].getSwitchState() == pcf::IndiElement::On )
        {
            continue;
        }
        if( elit->first == opModeName( SPARKLE ) )
        {
            newMode = SPARKLE;
        }
        else if( elit->first == opModeName( ARBCUBE ) )
        {
            newMode = ARBCUBE;
        }
        else if( elit->first == opModeName( TIPTILT ) && !m_syncopationTiming.empty() )
        {
            newMode = TIPTILT;
        }
    }
    if( newMode == -1 )
    {
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( newMode != m_opMode )
    {
        m_opMode = newMode;
        log<text_log>( "mode set to " + opModeName( newMode ), logPrio::LOG_NOTICE );
        m_restartSp = true;
    }

    indi::updateSelectionSwitchIfChanged( m_indiP_mode, opModeName( newMode ), m_indiDriver, pcf::IndiProperty::Ok );

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_file )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_file, ipRecv );

    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    std::string target = ipRecv["target"].get<std::string>();

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( m_fileName != target )
    {
        m_fileName = target;
        updatesIfChanged<std::string>( m_indiP_file, { "target", "current" }, { target, target }, INDI_IDLE );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSpeckle, m_indiP_loadFile )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loadFile, ipRecv );

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    m_reloadFile = true;
    m_restartSp  = true;
    setFileState( INDI_BUSY );
    return 0;
}

inline int dmSpeckle::checkRecordTimes()
{
    return telemeter<dmSpeckle>::checkRecordTimes( telem_dmspeck() );
}

inline int dmSpeckle::recordTelem( const telem_dmspeck *unused )
{
    static_cast<void>( unused );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    return recordDmSpeck( true );
}

inline int dmSpeckle::recordDmSpeck( bool force )
{
    static bool  lastModulating = m_modulating;
    static bool  lastTrigger    = m_trigger;
    static float lastFrequency  = m_frequency;
    static float lastSeparation = m_separation;
    static float lastAngle      = m_angle;
    static float lastAmp        = m_amp;
    static bool  lastCross      = m_cross;

    if( !( lastModulating == m_modulating ) || !( lastTrigger == m_trigger ) || !( lastFrequency == m_frequency ) ||
        !( lastSeparation == m_separation ) || !( lastAngle == m_angle ) || !( lastAmp == m_amp ) ||
        !( lastCross == m_cross ) || force )
    {
        telem<telem_dmspeck>( { m_modulating,
                                m_trigger,
                                m_frequency,
                                std::vector<float>( { m_separation } ),
                                std::vector<float>( { m_angle } ),
                                std::vector<float>( { m_amp } ),
                                std::vector<bool>( { m_cross } ) } );

        lastModulating = m_modulating;
        lastTrigger    = m_trigger;
        lastFrequency  = m_frequency;
        lastSeparation = m_separation;
        lastAngle      = m_angle;
        lastAmp        = m_amp;
        lastCross      = m_cross;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // dmSpeckle_hpp
