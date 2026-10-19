/** \file dmSpeckle.hpp
 * \brief The MagAO-X DM speckle maker header file
 *
 * \ingroup dmSpeckle_files
 */

#ifndef dmSpeckle_hpp
#define dmSpeckle_hpp

#include <atomic>
#include <cmath>

#include <mx/math/floatUtils.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/stringUtils.hpp>
#include <mx/sys/timeUtils.hpp>
#include <mx/sigproc/fourierModes.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "../../libMagAOX/app/dmSyncopationTiming.hpp"

/** \defgroup dmSpeckle
 * \brief The DM speckle maker app
 *
 * Creates a set of fourier modes to generate speckles, then applies them to a DM channel
 * at a specified rate.
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
class dmSpeckle : public MagAOXApp<true>, public dev::telemeter<dmSpeckle>
{

    typedef float                     realT;
    typedef dev::telemeter<dmSpeckle> telemeterT;

    friend class dev::telemeter<dmSpeckle>;
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

    realT m_amp{ 0.01 }; ///< The speckle amplitude on the DM

    bool m_cross{ true }; ///< If true, also apply the cross speckles rotated by 90 degrees

    realT m_frequency{ 2000 }; ///< Legacy frame rate, or physical tip/tilt vibration frequency in Hz.

    unsigned m_dwell{ 1 }; ///< The dwell time for each speckle, or for how many frames it is held.

    int m_single{ -1 }; ///< if >= 0 a single frame is non-zero.

    int m_opMode{ SPARKLE }; ///< Generated sparkle, file cube, or time-based tip/tilt waveform.

    std::string m_fileName; ///< Input cube for arbitrary-cube modulation.

    std::string m_syncopationTiming; ///< Nonempty timing stream selects auxiliary-provider operation.

    double m_writeLeadUs{ 50 }; ///< Advance before a provider snapshot cutoff, in microseconds.

    ///@}

    mx::improc::eigenCube<realT> m_shapes; ///< Prepared legacy shapes or arbitrary provider cube.

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

  public:
    /// Default c'tor.
    dmSpeckle();

    /// D'tor, declared and defined for noexcept.
    ~dmSpeckle() noexcept
    {
    }

    /// Register legacy modulation and optional provider configuration.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load configuration and request shutdown for invalid provider settings.
    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for dmSpeckle.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

  protected:
    /// Prepare shape storage for the configured legacy or managed modulation.
    int generateSpeckles();

    /// Validate the optional provider mode without changing legacy channel semantics.
    int validateSyncopationConfig();

    /// Fill a circular tip/tilt vibration with each axis at the requested physical frequency.
    static void fillTipTilt( realT   *pixels,    /**< [out] Column-major destination pixels. */
                             uint32_t width,     /**< [in] Number of rows. */
                             uint32_t height,    /**< [in] Number of columns. */
                             realT    amplitude, /**< [in] Edge amplitude of each axis. */
                             realT    frequency, /**< [in] Actual vibration frequency in Hz. */
                             int64_t  targetNs /**< [in] CLOCK_MONOTONIC command target time. */ );

    /// Read one coherent same-host provider schedule using normal stream publication metadata.
    static bool readSyncopationRecord( const IMAGE   &stream, /**< [in] Attached timing image. */
                                       synco::Record &record /**< [out] Stable timing record. */ );

    /// Publish a prepared auxiliary source if its cutoff has not passed.
    bool publishSyncopation( const realT *pixels, /**< [in] Prepared full source command. */
                             int64_t      cutoffNs /**< [in] Absolute monotonic snapshot cutoff. */ );

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

    bool m_modThreadInit{ true }; ///< Synchronizer to ensure f.g. thread initializes before doing dangerous things.

    pid_t m_modThreadID{ 0 }; ///< Modulate thread PID.

    pcf::IndiProperty m_modThreadProp; ///< The property to hold the modulator thread details.

    /// Thread starter, called by modThreadStart on thread construction.  Calls modThreadExec.
    static void modThreadStart( dmSpeckle *d /**< [in] a pointer to a dmSpeckle instance (normally this) */ );

    /// Execute the frame grabber main loop.
    void modThreadExec();

    ///@}

    // INDI:
  protected:
    // declare our properties
    pcf::IndiProperty m_indiP_dm; ///< INDI dm control/status.

    pcf::IndiProperty m_indiP_trigger; ///< INDI trigger control/status.

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

  public:
    /// Handle changes to the trigger property.
    INDI_NEWCALLBACK_DECL( dmSpeckle, m_indiP_trigger );

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

void dmSpeckle::setupConfig()
{
    TELEMETER_SETUP_CONFIG( config );
    config.add( "syncopation.timingStream",
                "",
                "syncopation.timingStream",
                argType::Required,
                "syncopation",
                "timingStream",
                false,
                "string",
                "Optional same-host dmSyncopation schedule; requires an _aux output channel." );
    config.add( "syncopation.writeLeadUs",
                "",
                "syncopation.writeLeadUs",
                argType::Required,
                "syncopation",
                "writeLeadUs",
                false,
                "double",
                "Provider write advance before the snapshot cutoff, in microseconds (default 50)." );
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
                "The frequency to modulate at if not triggering (default 2000 Hz)." );

    config.add( "dm.dwell",
                "",
                "dm.dwell",
                argType::True,
                "dm",
                "dwell",
                false,
                "int",
                "The dwell time for each speckle, or for how many frames it is held. Default=1." );

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
                "File name containing the FITS cube to modulate the DM with.  Only valid if opMode==arbcube." );
}

int dmSpeckle::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_dmChannelName, "dm.channelName" );
    _config( m_syncopationTiming, "syncopation.timingStream" );
    _config( m_writeLeadUs, "syncopation.writeLeadUs" );

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

bool dmSpeckle::readSyncopationRecord( const IMAGE &stream, synco::Record &record )
{
    if( stream.md->datatype != _DATATYPE_INT64 || stream.md->nelement != synco::words || stream.md->write )
    {
        return false;
    }
    uint64_t before = stream.md->cnt0;
    std::atomic_thread_fence( std::memory_order_acquire );
    memcpy( record.data(), stream.array.raw, sizeof( record ) );
    std::atomic_thread_fence( std::memory_order_acquire );
    return !stream.md->write && before == stream.md->cnt0;
}

bool dmSpeckle::publishSyncopation( const realT *pixels, int64_t cutoffNs )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( !m_modulating || synco::monotonicNs() >= cutoffNs )
    {
        return false;
    }
    m_imageStream.md->write = 1;
    memcpy( m_imageStream.array.raw, pixels, m_width * m_height * m_typeSize );
    ImageStreamIO_UpdateIm( &m_imageStream );
    return true;
}

void dmSpeckle::syncopationThreadExec()
{
    IMAGE                         timing{};
    bool                          timingOpened = false;
    int                           semaphore    = -1;
    int64_t                       lastEpoch = 0, lastGeneration = 0, lastTarget = -1, lastValid = 0;
    mx::improc::eigenImage<realT> command( m_width, m_height );

    while( !m_shutdown )
    {
        if( !timingOpened )
        {
            if( ImageStreamIO_openIm( &timing, m_syncopationTiming.c_str() ) != 0 )
            {
                mx::sys::milliSleep( 10 );
                continue;
            }
            semaphore = ImageStreamIO_getsemwaitindex( &timing, 6 );
            if( timing.md->datatype != _DATATYPE_INT64 || timing.md->nelement != synco::words || semaphore < 0 )
            {
                if( semaphore >= 0 )
                    timing.semReadPID[semaphore] = 0;
                ImageStreamIO_closeIm( &timing );
                mx::sys::milliSleep( 10 );
                continue;
            }
            timingOpened = true;
            lastTarget   = -1;
            lastValid    = synco::monotonicNs();
        }

        synco::Record record{};
        int64_t       now   = synco::monotonicNs();
        bool          valid = readSyncopationRecord( timing, record ) && synco::validRecord( record, now );
        if( valid )
            lastValid = now;
        if( valid && m_modulating )
        {
            int step = synco::nextStep( record, now );
            if( step >= 0 )
            {
                int64_t target    = record[synco::targets + step];
                int64_t cutoff    = record[synco::cutoffs + step];
                bool    newTarget = record[synco::epoch] != lastEpoch || record[synco::generation] != lastGeneration ||
                                 target != lastTarget;
                if( newTarget )
                {
                    realT  amplitude, frequency;
                    double advance;
                    { // mutex scope
                        std::lock_guard<std::mutex> lock( m_indiMutex );
                        amplitude = m_amp;
                        frequency = m_frequency;
                        advance   = m_writeLeadUs;
                    }
                    if( m_opMode == TIPTILT )
                    {
                        fillTipTilt( command.data(), m_width, m_height, amplitude, frequency, target );
                    }
                    else
                    {
                        if( m_shapes.planes() == 0 || m_restartSp )
                        {
                            m_restartSp = false;
                            if( generateSpeckles() < 0 )
                            {
                                m_modulating = false;
                                continue;
                            }
                        }
                        long double index = std::floor( static_cast<long double>( target ) * frequency /
                                                        ( 1e9L * std::max( 1U, m_dwell ) ) );
                        int         plane = static_cast<int>( std::fmod( index, m_shapes.planes() ) );
                        command           = m_shapes.image( plane );
                    }

                    int64_t writeTime = cutoff - static_cast<int64_t>( advance * 1e3 );
                    if( writeTime > now )
                    {
                        timespec ts{ writeTime / 1000000000, writeTime % 1000000000 };
                        int      rv;
                        do
                        {
                            rv = clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr );
                        } while( rv == EINTR && !m_shutdown && m_modulating );
                    }
                    synco::Record current{};
                    now = synco::monotonicNs();
                    if( !m_shutdown && m_modulating && readSyncopationRecord( timing, current ) &&
                        synco::validRecord( current, now ) && current[synco::epoch] == record[synco::epoch] &&
                        current[synco::generation] == record[synco::generation] &&
                        current[synco::cutoffs + step] == cutoff && current[synco::targets + step] == target )
                    {
                        publishSyncopation( command.data(), cutoff );
                    }
                    lastEpoch      = record[synco::epoch];
                    lastGeneration = record[synco::generation];
                    lastTarget     = target;
                    continue;
                }
            }
        }
        else if( ( record[synco::expires] > 0 && now >= record[synco::expires] ) || now - lastValid > 1000000000 ||
                 timing.md->sem == 0 )
        {
            timing.semReadPID[semaphore] = 0;
            ImageStreamIO_closeIm( &timing );
            timingOpened = false;
            continue;
        }

        timespec timeout;
        clock_gettime( CLOCK_REALTIME, &timeout );
        timeout.tv_nsec += 1000000;
        if( timeout.tv_nsec >= 1000000000 )
        {
            ++timeout.tv_sec;
            timeout.tv_nsec -= 1000000000;
        }
        sem_timedwait( timing.semptr[semaphore], &timeout );
    }
    if( timingOpened )
    {
        timing.semReadPID[semaphore] = 0;
        ImageStreamIO_closeIm( &timing );
    }
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

    if( m_opMode == SPARKLE )
    {
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
    }

    createStandardIndiNumber<float>( m_indiP_amp, "amp", -1, 0, 1, "%f" );
    m_indiP_amp["current"] = m_amp;
    m_indiP_amp["target"]  = m_amp;
    registerIndiPropertyNew( m_indiP_amp, INDI_NEWCALLBACK( m_indiP_amp ) );

    createStandardIndiNumber<float>( m_indiP_frequency, "frequency", 0, 0, 10000, "%f" );
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
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int dmSpeckle::generateSpeckles()
{
    if( m_opMode == TIPTILT )
    {
        m_shapes.resize( m_width, m_height, 1 );
    }
    else if( m_opMode == ARBCUBE )
    {
        mx::fits::fitsFile<float> ff;
        mx::error_t               errc = ff.read( m_shapes, m_fileName );
        if( errc != mx::error_t::noerror )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, "no file with that name" } );
        }

        if( m_shapes.rows() != m_width || m_shapes.cols() != m_height )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, "shape cube is not the right size" } );
        }

        for( int p = 0; p < m_shapes.planes(); ++p )
        {
            m_shapes.image( p ) *= (float)m_amp;
        }
    }
    else
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
    updateIfChanged( m_indiP_delay, "current", m_triggerDelay );
    updateIfChanged( m_indiP_amp, "current", m_amp );
    updateIfChanged( m_indiP_frequency, "current", m_frequency );
    updateIfChanged( m_indiP_dwell, "current", m_dwell );
    updateIfChanged( m_indiP_single, "current", m_single );

    return 0;
}

inline void dmSpeckle::modThreadStart( dmSpeckle *d )
{
    d->modThreadExec();
}

inline void dmSpeckle::modThreadExec()
{
    m_modThreadID = syscall( SYS_gettid );

    // Wait fpr the thread starter to finish initializing this thread.
    while( ( m_modThreadInit == true || state() != stateCodes::READY ) && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    if( !m_syncopationTiming.empty() )
    {
        syncopationThreadExec();
        return;
    }

    while( m_shutdown == 0 )
    {
        if( !m_modulating && !m_shutdown ) // If we aren't modulating we sleep for 1/2 a second
        {
            mx::sys::milliSleep( 500 );
        }

        if( m_modulating && !m_shutdown )
        {
            m_restartSp = false;
            if( generateSpeckles() < 0 )
            {
                m_modulating = false;
                continue;
            }

            int64_t freqNsec = ( 1.0 / m_frequency ) * 1e9;
            int64_t dnsec;

            int idx = 0;

            timespec modstart;
            timespec currtime;

            bool   triggered = false;
            sem_t *sem       = nullptr;
            if( m_dmTriggerChannel == "" )
            {
                m_trigger = false;
                indi::updateSwitchIfChanged(
                    m_indiP_trigger, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
            }
            else if( m_trigger == true )
            {
                ImageStreamIO_semflush( &m_triggerStream, m_triggerSemaphore );

                sem = m_triggerStream.semptr[m_triggerSemaphore]; ///< The semaphore to monitor for new image data
            }

            log<text_log>( "started modulating", logPrio::LOG_NOTICE );
            // To send a message
            log<telem_dmspeck>( { m_modulating,
                                  m_trigger,
                                  m_frequency,
                                  std::vector<float>( { m_separation } ),
                                  std::vector<float>( { m_angle } ),
                                  std::vector<float>( { m_amp } ),
                                  std::vector<bool>( { m_cross } ) },
                                logPrio::LOG_INFO );
            // The official record:
            recordDmSpeck( true );

            dnsec = 0;
            clock_gettime( CLOCK_REALTIME, &modstart );

            unsigned dwelled = 0;
            if( m_dwell == 0 )
                m_dwell = 1;

            float triggerDelay = m_triggerDelay / 1e6;

            double t0, t1;

            while( m_modulating && !m_restartSp && !m_shutdown )
            {
                if( m_trigger )
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
                                break;
                            mx::sys::nanoSleep( dt );
                            t1 = mx::sys::get_curr_time();
                        }

                        triggered = true;
                    }
                    else
                    {
                        triggered = false;

                        // Check for why we timed out
                        if( errno == EINTR )
                            break; // This indicates signal interrupted us, time to restart or shutdown, loop will exit
                                   // normally if flags set.

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
                    mx::sys::nanoSleep( 0.5 * dnsec );
                    clock_gettime( CLOCK_REALTIME, &currtime );

                    dnsec =
                        ( currtime.tv_sec - modstart.tv_sec ) * 1000000000 + ( currtime.tv_nsec - modstart.tv_nsec );
                    triggered = false;
                }

                if( dwelled < m_dwell - 1 )
                {
                    ++dwelled;
                }
                else if( dnsec >= freqNsec || triggered )
                {
                    // Do the write
                    dwelled = 0;

                    m_imageStream.md->write = 1;

                    memcpy( m_imageStream.array.raw, m_shapes.image( idx ).data(), m_width * m_height * m_typeSize );

                    m_imageStream.md->atime     = currtime;
                    m_imageStream.md->writetime = currtime;

                    if( !m_trigger || triggerDelay > 0 )
                    {
                        m_imageStream.md->cnt0++;
                    }

                    m_imageStream.md->write = 0;
                    ImageStreamIO_sempost( &m_imageStream, -1 );

                    ++idx;
                    if( idx >= m_shapes.planes() )
                        idx = 0;

                    if( !m_trigger )
                    {
                        modstart.tv_nsec += freqNsec;
                        if( modstart.tv_nsec >= 1000000000 )
                        {
                            modstart.tv_nsec -= 1000000000;
                            modstart.tv_sec += 1;
                        }
                        dnsec = freqNsec;
                    }
                }
            }
            if( m_restartSp )
                continue;

            recordDmSpeck( true );
            log<text_log>( "stopped modulating", logPrio::LOG_NOTICE );
            // Always zero when done
            clock_gettime( CLOCK_REALTIME, &currtime );
            m_imageStream.md->write = 1;

            memset( m_imageStream.array.raw, 0.0, m_width * m_height * m_typeSize );

            m_imageStream.md->atime     = currtime;
            m_imageStream.md->writetime = currtime;

            if( !m_trigger )
                m_imageStream.md->cnt0++;

            m_imageStream.md->write = 0;
            ImageStreamIO_sempost( &m_imageStream, -1 );
            log<text_log>( "zeroed" );
        }
    }
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
        return 0;

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
    updateIfChanged( m_indiP_amp, "target", m_amp );

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
        return 0;

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
    updateIfChanged( m_indiP_frequency, "target", m_frequency );

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
        return 0;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_modulating = false;
        indi::updateSwitchIfChanged( m_indiP_modulating, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_modulating = true;
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
        return 0;

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_opened || m_imageStream.md == nullptr )
        {
            return -1;
        }
        m_imageStream.md->write = 1;

        memset( m_imageStream.array.raw, 0, m_width * m_height * m_typeSize );
        timespec currtime;
        clock_gettime( CLOCK_REALTIME, &currtime );
        m_imageStream.md->atime     = currtime;
        m_imageStream.md->writetime = currtime;

        if( !m_syncopationTiming.empty() )
        {
            ImageStreamIO_UpdateIm( &m_imageStream );
        }
        else
        {
            m_imageStream.md->cnt0++;
            m_imageStream.md->write = 0;
            ImageStreamIO_sempost( &m_imageStream, -1 );
        }
        log<text_log>( "zeroed" );
    }

    return 0;
}

inline int dmSpeckle::checkRecordTimes()
{
    return telemeter<dmSpeckle>::checkRecordTimes( telem_dmspeck() );
}

inline int dmSpeckle::recordTelem( const telem_dmspeck *unused )
{
    static_cast<void>( unused );
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
