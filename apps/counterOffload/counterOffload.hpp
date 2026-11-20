/** \file counterOffload.hpp
 * \brief The MagAO-X tweeter counter-offload producer header file
 *
 * \ingroup counterOffload_files
 */

#ifndef counterOffload_hpp
#define counterOffload_hpp

#include <atomic>
#include <cmath>
#include <vector>

#include <mx/improc/eigenImage.hpp>
#include <mx/math/floatUtils.hpp>
#include <mx/sys/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "../../libMagAOX/app/dev/dmSyncopationProducer.hpp"

/** \defgroup counterOffload Tweeter Counter-Offload
 * \brief Counters woofer offloads on the tweeter through dmSyncopation.
 *
 * Watches the offload change published by t2wOffloader as it appears on the tweeter, and publishes its negative
 * to a dmSyncopation auxiliary source, ramped to follow the woofer's rise and optionally leaked back to zero.
 *
 * \ingroup apps
 *
 */

/** \defgroup counterOffload_files Tweeter Counter-Offload Files
 * \ingroup counterOffload
 */

namespace MagAOX
{
namespace app
{

/// MagAO-X application publishing tweeter counter-offload commands as a dmSyncopation producer.
/** Each update of the input stream (normally `aol0_tweeterOffload`) starts one ramp of that offload change.
 * For each scheduler slot the published command is `-gain * (A + sum_k delta_k * ramp(target - t0_k))`, where
 * completed ramps are folded into the accumulator `A`, which is multiplied by `(1 - leak)` after every
 * successful publication.  Disabling stops accepting new offloads and requests a zero, so the contribution is
 * cleared at the next slot.
 *
 * \ingroup counterOffload
 */
class counterOffload : public MagAOXApp<true>,
                       public dev::shmimMonitor<counterOffload>,
                       public dev::dmSyncopationProducer<counterOffload>
{
    typedef float realT; ///< Floating point type of the tweeter commands.

    typedef dev::shmimMonitor<counterOffload> shmimMonitorT; ///< The input stream monitor base.

    typedef dev::dmSyncopationProducer<counterOffload> dmSyncopationProducerT; ///< The syncopation producer base.

    friend class dev::shmimMonitor<counterOffload>;

    friend class dev::dmSyncopationProducer<counterOffload>;

    friend class counterOffload_test;

  public:
    /// One offload change and the time its ramp is referenced to.
    struct rampT
    {
        mx::improc::eigenImage<realT> delta; ///< The offload change as it appears on the tweeter.

        int64_t t0Ns{ 0 }; ///< CLOCK_MONOTONIC reference time of the woofer command, in ns.

        bool used{ false }; ///< Whether this pool entry holds an unfinished ramp.
    };

    static constexpr size_t m_maxRamps{ 8 }; ///< Size of the preallocated ramp pool.

    static constexpr realT m_zeroTol{ 1e-7 }; ///< A leaking contribution below this magnitude is set to zero.

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_channel{ "dm01disp08_aux" }; ///< The dmSyncopation auxiliary source this app writes.

    double m_rampStartUs{ 120 }; ///< Ramp start after the woofer command, in us.

    double m_rampDurationUs{ 330 }; ///< Ramp duration, in us.

    double m_leak{ 0 }; ///< Fraction of the completed contribution removed after each published slot, 0 to 1.

    double m_timeOffsetUs{ 0 }; ///< Offset added to the input receipt time to give the ramp reference, in us.

    double m_gain{ 1 }; ///< Scale applied to the published command, 0 to 1.

    double m_computeLeadUs{ 20 }; ///< Time reserved before the publish lead to compute the command, in us.

    int m_threadPrio{ 20 }; ///< Real-time priority of the producer thread.

    std::string m_threadCpuset; ///< The cpuset for the producer thread.

    ///@}

    /** \name Counter-Offload State - Data
     *@{
     */

    IMAGE m_auxStream{}; ///< The writable auxiliary source, owned by this app and managed by the producer helper.

    bool m_auxOpened{ false }; ///< Whether m_auxStream is open.

    uint32_t m_auxWidth{ 0 }; ///< Width of the auxiliary source.

    uint32_t m_auxHeight{ 0 }; ///< Height of the auxiliary source.

    std::vector<rampT> m_ramps; ///< Preallocated pool of unfinished ramps.

    mx::improc::eigenImage<realT> m_accum; ///< Sum of completed offload changes, leaked after each publication.

    mx::improc::eigenImage<realT> m_command; ///< The command for the current slot, owned by the producer thread.

    bool m_accumNonzero{ false }; ///< Whether m_accum holds a nonzero contribution.

    bool m_finalZero{ false }; ///< A leaked contribution reached zero and that zero must still be published.

    std::atomic<bool> m_enabled{ false }; ///< Whether new offloads are accepted.

    std::atomic<bool> m_zeroRequested{ false }; ///< Operator request to clear and publish a zero contribution.

    std::mutex m_stateMutex; ///< Protects the ramps, accumulator and ramp parameters.

    uint64_t m_events{ 0 }; ///< Offload updates accepted.

    uint64_t m_published{ 0 }; ///< Commands successfully published.

    uint64_t m_failed{ 0 }; ///< Commands not published by syncopationPublish.

    uint64_t m_overflows{ 0 }; ///< Offloads folded immediately because the ramp pool was full.

    ///@}

  public:
    /// Default c'tor.
    counterOffload();

    /// D'tor, declared and defined for noexcept.
    ~counterOffload() noexcept
    {
    }

    /// Set up the configuration system.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /**
     * \returns 0 on success
     * \returns -1 on an invalid configuration
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load the configuration.
    virtual void loadConfig();

    /// Startup function
    /** Registers INDI properties, starts the input monitor and the producer thread.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int appStartup();

    /// Implementation of the FSM for counterOffload.
    /** Opens the auxiliary source, then goes to OPERATING so the input monitor runs.
     *
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     * \returns 0 on success
     */
    virtual int appShutdown();

    /// Validate the loaded configuration.
    /**
     * \returns 0 if valid
     * \returns -1 if invalid
     */
    int validateConfig();

    /// Size the ramp pool, accumulator and command for the input stream, checking against the auxiliary source.
    /**
     * \returns 0 on success
     * \returns -1 if the size does not match an open auxiliary source
     */
    int setupBuffers( uint32_t width /**< [in] width of the offload images */,
                      uint32_t height /**< [in] height of the offload images */ );

    /// Allocate buffers for a newly connected input stream.
    /**
     * \returns 0 on success
     * \returns -1 on a size or type mismatch
     */
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Record a new offload change while enabled.
    /**
     * \returns 0 always
     */
    int processImage( void              *curr_src, /**< [in] pointer to start of current frame.*/
                      const dev::shmimT &dummy     /**< [in] tag to differentiate shmimMonitor parents.*/
    );

    /// Fraction of the ramp completed at a time after the woofer command.
    /** A raised cosine: 0 before the start, 1 at and after the end.
     *
     * \returns the ramp fraction, from 0 to 1
     */
    static realT rampFraction( double tauUs /**< [in] time after the woofer command, in us */,
                               double startUs /**< [in] ramp start, in us */,
                               double durationUs /**< [in] ramp duration, in us */ );

    /// Start a ramp of an offload change.  The caller must hold m_stateMutex.
    void addRamp( const realT *delta /**< [in] offload change, sized like m_accum */,
                  int64_t      t0Ns /**< [in] CLOCK_MONOTONIC ramp reference time, in ns */ );

    /// Whether the next slot needs a publication.  The caller must hold m_stateMutex.
    bool needPublish();

    /// Build m_command for a slot target, folding completed ramps into the accumulator.
    /** The caller must hold m_stateMutex.
     */
    void buildCommand( int64_t targetNs /**< [in] CLOCK_MONOTONIC slot target, in ns */ );

    /// Clear all ramps and the accumulator, and set m_command to zero.  The caller must hold m_stateMutex.
    void clearState();

    /// Apply the leak after a successful publication.  The caller must hold m_stateMutex.
    void afterPublish();

    /// Enable or disable accepting offloads; disabling also requests a zero of the contribution.
    /** \returns true if the enabled state changed
     */
    bool setEnabled( bool on /**< [in] true to accept new offloads */ );

  protected:
    /** \name Producer Thread
     * Publishes counter-offload commands for each dmSyncopation slot.
     *
     * @{
     */

    std::thread m_coThread; ///< The producer thread.

    bool m_coThreadInit{ true }; ///< Synchronizer to ensure the thread initializes before doing dangerous things.

    pid_t m_coThreadID{ 0 }; ///< Producer thread PID.

    pcf::IndiProperty m_coThreadProp; ///< The property to hold the producer thread details.

    /// Producer thread starter function.
    static void coThreadStart( counterOffload *c /**< [in] pointer to this */ );

    /// Producer thread function.  Runs until m_shutdown is true.
    void coThreadExec();

    ///@}

    /** \name INDI Interface
     *
     * @{
     */

    pcf::IndiProperty m_indiP_enable; ///< Toggle switch to accept offloads.

    pcf::IndiProperty m_indiP_zero; ///< Request switch to clear and publish a zero contribution.

    pcf::IndiProperty m_indiP_gain; ///< Property to set the command gain.

    pcf::IndiProperty m_indiP_leak; ///< Property to set the leak.

    pcf::IndiProperty m_indiP_rampStart; ///< Property to set the ramp start.

    pcf::IndiProperty m_indiP_rampDuration; ///< Property to set the ramp duration.

    pcf::IndiProperty m_indiP_timeOffset; ///< Property to set the ramp reference time offset.

    pcf::IndiProperty m_indiP_stats; ///< Read-only counters.

    /// Update a numeric parameter from an INDI new-property request, within limits.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int setNumber( pcf::IndiProperty       &prop /**< [in,out] the property being changed */,
                   const pcf::IndiProperty &ipRecv /**< [in] the received request */,
                   double                  &value /**< [out] the parameter to set */,
                   double                   minVal /**< [in] smallest allowed value */,
                   double                   maxVal /**< [in] largest allowed value */ );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_enable );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_zero );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_gain );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_leak );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_rampStart );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_rampDuration );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_timeOffset );

    ///@}
};

inline counterOffload::counterOffload() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_ramps.resize( m_maxRamps );
}

inline void counterOffload::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    DMSYNCOPATIONPRODUCER_SETUP_CONFIG( config );

    config.add( "counterOffload.channel",
                "",
                "counterOffload.channel",
                argType::Required,
                "counterOffload",
                "channel",
                false,
                "string",
                "The dmSyncopation auxiliary source to write.  Must end in _aux.  Default is dm01disp08_aux." );

    config.add( "counterOffload.rampStartUs",
                "",
                "counterOffload.rampStartUs",
                argType::Required,
                "counterOffload",
                "rampStartUs",
                false,
                "real",
                "Ramp start after the woofer command, in us.  Default is 120." );

    config.add( "counterOffload.rampDurationUs",
                "",
                "counterOffload.rampDurationUs",
                argType::Required,
                "counterOffload",
                "rampDurationUs",
                false,
                "real",
                "Ramp duration, in us.  Default is 330." );

    config.add( "counterOffload.leak",
                "",
                "counterOffload.leak",
                argType::Required,
                "counterOffload",
                "leak",
                false,
                "real",
                "Fraction of the completed contribution removed after each published slot, 0 to 1.  Default is 0." );

    config.add( "counterOffload.timeOffsetUs",
                "",
                "counterOffload.timeOffsetUs",
                argType::Required,
                "counterOffload",
                "timeOffsetUs",
                false,
                "real",
                "Offset added to the offload receipt time to give the ramp reference, in us.  Default is 0." );

    config.add( "counterOffload.gain",
                "",
                "counterOffload.gain",
                argType::Required,
                "counterOffload",
                "gain",
                false,
                "real",
                "Scale applied to the published command, 0 to 1.  Default is 1." );

    config.add( "counterOffload.computeLeadUs",
                "",
                "counterOffload.computeLeadUs",
                argType::Required,
                "counterOffload",
                "computeLeadUs",
                false,
                "real",
                "Time reserved before the publish lead to compute the command, in us.  Default is 20." );

    config.add( "counterOffload.threadPrio",
                "",
                "counterOffload.threadPrio",
                argType::Required,
                "counterOffload",
                "threadPrio",
                false,
                "int",
                "Real-time priority of the producer thread.  Default is 20." );

    config.add( "counterOffload.cpuset",
                "",
                "counterOffload.cpuset",
                argType::Required,
                "counterOffload",
                "cpuset",
                false,
                "string",
                "The cpuset for the producer thread." );
}

inline int counterOffload::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::m_shmimName = "aol0_tweeterOffload";

    SHMIMMONITOR_LOAD_CONFIG( _config );

    DMSYNCOPATIONPRODUCER_LOAD_CONFIG( _config );

    _config( m_channel, "counterOffload.channel" );
    _config( m_rampStartUs, "counterOffload.rampStartUs" );
    _config( m_rampDurationUs, "counterOffload.rampDurationUs" );
    _config( m_leak, "counterOffload.leak" );
    _config( m_timeOffsetUs, "counterOffload.timeOffsetUs" );
    _config( m_gain, "counterOffload.gain" );
    _config( m_computeLeadUs, "counterOffload.computeLeadUs" );
    _config( m_threadPrio, "counterOffload.threadPrio" );
    _config( m_threadCpuset, "counterOffload.cpuset" );

    return validateConfig();
}

inline void counterOffload::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<text_log>( "invalid configuration", logPrio::LOG_CRITICAL );
        m_shutdown = true;
    }
}

inline int counterOffload::validateConfig()
{
    constexpr size_t suffixLength = 4;

    auto inRange = []( double v, double lo, double hi )
    {
        return mx::math::isFinite( v ) && v >= lo && v <= hi;
    };

    if( m_syncopationTiming.empty() || m_channel.size() <= suffixLength ||
        m_channel.compare( m_channel.size() - suffixLength, suffixLength, "_aux" ) != 0 ||
        m_syncopationTiming == m_channel || !inRange( m_rampStartUs, 0, 1e6 ) || !inRange( m_rampDurationUs, 1, 1e6 ) ||
        !inRange( m_leak, 0, 1 ) || !inRange( m_timeOffsetUs, -1e6, 1e6 ) || !inRange( m_gain, 0, 1 ) ||
        !inRange( m_computeLeadUs, 0, 1e6 ) )
    {
        return -1;
    }

    return 0;
}

inline int counterOffload::appStartup()
{
    createStandardIndiToggleSw( m_indiP_enable, "enable" );
    if( registerIndiPropertyNew( m_indiP_enable, INDI_NEWCALLBACK( m_indiP_enable ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    createStandardIndiRequestSw( m_indiP_zero, "zero" );
    if( registerIndiPropertyNew( m_indiP_zero, INDI_NEWCALLBACK( m_indiP_zero ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    // Register the numeric parameters, each with current and target set to its configured value.
    struct numberT
    {
        pcf::IndiProperty &prop;
        const char        *name;
        double             minVal;
        double             maxVal;
        double             value;
        int ( *callback )( void *, const pcf::IndiProperty & );
    };

    numberT numbers[] = {
        { m_indiP_gain, "gain", 0, 1, m_gain, INDI_NEWCALLBACK( m_indiP_gain ) },
        { m_indiP_leak, "leak", 0, 1, m_leak, INDI_NEWCALLBACK( m_indiP_leak ) },
        { m_indiP_rampStart, "rampStartUs", 0, 1e6, m_rampStartUs, INDI_NEWCALLBACK( m_indiP_rampStart ) },
        { m_indiP_rampDuration, "rampDurationUs", 1, 1e6, m_rampDurationUs, INDI_NEWCALLBACK( m_indiP_rampDuration ) },
        { m_indiP_timeOffset, "timeOffsetUs", -1e6, 1e6, m_timeOffsetUs, INDI_NEWCALLBACK( m_indiP_timeOffset ) } };

    for( auto &n : numbers )
    {
        createStandardIndiNumber<double>( n.prop, n.name, n.minVal, n.maxVal, 0, "%g" );
        n.prop["current"] = n.value;
        n.prop["target"]  = n.value;

        if( registerIndiPropertyNew( n.prop, n.callback ) < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
    }

    createROIndiNumber( m_indiP_stats, "stats", "Counters" );
    for( const char *el : { "events", "published", "failed", "overflows", "active" } )
    {
        m_indiP_stats.add( pcf::IndiElement( el ) );
        m_indiP_stats[el] = 0;
    }
    if( registerIndiPropertyReadOnly( m_indiP_stats ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    SHMIMMONITOR_APP_STARTUP;

    if( threadStart( m_coThread,
                     m_coThreadInit,
                     m_coThreadID,
                     m_coThreadProp,
                     m_threadPrio,
                     m_threadCpuset,
                     "producer",
                     this,
                     coThreadStart ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    state( stateCodes::NOTCONNECTED );

    return 0;
}

inline int counterOffload::appLogic()
{
    if( state() == stateCodes::NOTCONNECTED )
    {
        if( ImageStreamIO_openIm( &m_auxStream, m_channel.c_str() ) == 0 )
        {
            if( m_auxStream.md->datatype != _DATATYPE_FLOAT || m_auxStream.md->naxis != 2 ||
                m_auxStream.md->size[0] == 0 || m_auxStream.md->size[1] == 0 )
            {
                ImageStreamIO_closeIm( &m_auxStream );
                return log<text_log, -1>( m_channel + " must be a nonempty 2-D float image.", logPrio::LOG_CRITICAL );
            }

            m_auxOpened = true;
            m_auxWidth  = m_auxStream.md->size[0];
            m_auxHeight = m_auxStream.md->size[1];

            log<text_log>( "opened " + m_channel );
            state( stateCodes::OPERATING );
        }
    }

    SHMIMMONITOR_APP_LOGIC;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    std::vector<double> stats;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );

        double active = 0;
        for( const auto &r : m_ramps )
        {
            active += r.used;
        }

        stats = { double( m_events ), double( m_published ), double( m_failed ), double( m_overflows ), active };
    }

    updateIfChanged(
        m_indiP_stats, std::vector<std::string>{ "events", "published", "failed", "overflows", "active" }, stats );

    return 0;
}

inline int counterOffload::appShutdown()
{
    if( m_coThread.joinable() )
    {
        try
        {
            m_coThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }

    SHMIMMONITOR_APP_SHUTDOWN;

    DMSYNCOPATIONPRODUCER_APP_SHUTDOWN;

    if( m_auxOpened )
    {
        ImageStreamIO_closeIm( &m_auxStream );
        m_auxOpened = false;
    }

    return 0;
}

inline int counterOffload::setupBuffers( uint32_t width, uint32_t height )
{
    if( m_auxOpened && ( width != m_auxWidth || height != m_auxHeight ) )
    {
        return log<text_log, -1>( "offload stream size does not match " + m_channel, logPrio::LOG_ERROR );
    }

    std::lock_guard<std::mutex> guard( m_stateMutex );

    for( auto &r : m_ramps )
    {
        r.delta.resize( width, height );
        r.used = false;
    }

    m_accum.resize( width, height );
    m_command.resize( width, height );
    clearState();

    return 0;
}

inline int counterOffload::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    if( shmimMonitorT::m_dataType != _DATATYPE_FLOAT )
    {
        return log<text_log, -1>( "offload stream is not float", logPrio::LOG_ERROR );
    }

    return setupBuffers( shmimMonitorT::m_width, shmimMonitorT::m_height );
}

inline int counterOffload::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    const int64_t received = synco::monotonicNs();

    if( !m_enabled )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_stateMutex );

    addRamp( static_cast<const realT *>( curr_src ), received + static_cast<int64_t>( m_timeOffsetUs * 1000 ) );

    return 0;
}

inline counterOffload::realT counterOffload::rampFraction( double tauUs, double startUs, double durationUs )
{
    if( tauUs <= startUs )
    {
        return 0;
    }

    if( tauUs >= startUs + durationUs )
    {
        return 1;
    }

    return 0.5 - 0.5 * std::cos( M_PI * ( tauUs - startUs ) / durationUs );
}

inline void counterOffload::addRamp( const realT *delta, int64_t t0Ns )
{
    if( m_accum.size() == 0 )
    {
        return;
    }

    rampT *slot = nullptr;

    for( auto &r : m_ramps )
    {
        if( !r.used )
        {
            slot = &r;
            break;
        }
    }

    if( !slot ) // pool full: fold the oldest ramp into the accumulator to make room
    {
        slot = &m_ramps[0];
        for( auto &r : m_ramps )
        {
            if( r.t0Ns < slot->t0Ns )
            {
                slot = &r;
            }
        }

        m_accum += slot->delta;
        m_accumNonzero = true;
        ++m_overflows;
    }

    slot->delta = Eigen::Map<const mx::improc::eigenImage<realT>>( delta, m_accum.rows(), m_accum.cols() );
    slot->t0Ns  = t0Ns;
    slot->used  = true;

    ++m_events;
}

inline bool counterOffload::needPublish()
{
    if( m_zeroRequested || m_finalZero || ( m_accumNonzero && m_leak > 0 ) )
    {
        return true;
    }

    for( const auto &r : m_ramps )
    {
        if( r.used )
        {
            return true;
        }
    }

    return false;
}

inline void counterOffload::buildCommand( int64_t targetNs )
{
    m_command.setZero();

    for( auto &r : m_ramps )
    {
        if( !r.used )
        {
            continue;
        }

        const realT frac = rampFraction( ( targetNs - r.t0Ns ) / 1000.0, m_rampStartUs, m_rampDurationUs );

        if( frac >= 1 ) // ramp complete: hand it to the accumulator
        {
            m_accum += r.delta;
            m_accumNonzero = true;
            r.used         = false;
        }
        else if( frac > 0 )
        {
            m_command += frac * r.delta;
        }
    }

    m_command = static_cast<realT>( -m_gain ) * ( m_command + m_accum );
}

inline void counterOffload::clearState()
{
    for( auto &r : m_ramps )
    {
        r.used = false;
    }

    m_accum.setZero();
    m_command.setZero();
    m_accumNonzero = false;
    m_finalZero    = false;
}

inline void counterOffload::afterPublish()
{
    ++m_published;

    if( m_finalZero )
    {
        m_finalZero = false;
        return;
    }

    if( !m_accumNonzero || m_leak <= 0 )
    {
        return;
    }

    m_accum *= static_cast<realT>( 1 - m_leak );

    if( m_accum.abs().maxCoeff() < m_zeroTol )
    {
        m_accum.setZero();
        m_accumNonzero = false;
        m_finalZero    = true;
    }
}

inline bool counterOffload::setEnabled( bool on )
{
    if( on == m_enabled )
    {
        return false;
    }

    if( !on )
    {
        m_zeroRequested = true; // clear the contribution at the next slot
    }

    m_enabled = on;

    return true;
}

inline void counterOffload::coThreadStart( counterOffload *c )
{
    c->coThreadExec();
}

inline void counterOffload::coThreadExec()
{
    m_coThreadID = syscall( SYS_gettid );

    // Wait for the thread starter to finish initializing this thread.
    while( ( m_coThreadInit == true || state() != stateCodes::OPERATING ) && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    while( !m_shutdown )
    {
        bool idle;
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_stateMutex );
            idle = !m_enabled && !needPublish(); // nothing to finish and no offloads accepted
        }

        if( idle )
        {
            mx::sys::milliSleep( 1 );
            continue;
        }

        synco::Request request{};
        const int      pollResult = dmSyncopationProducerT::syncopationPoll( request, m_auxStream );

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

        // Compute as late as possible so the most recent offload is included.
        const int64_t computeAt = request.cutoffNs - static_cast<int64_t>( ( m_writeLeadUs + m_computeLeadUs ) * 1000 );
        const timespec wake     = synco::toTimespec( computeAt );

        while( synco::monotonicNs() < computeAt && !m_shutdown )
        {
            clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr );
        }

        bool zero = false;

        { //mutex scope
            std::lock_guard<std::mutex> guard( m_stateMutex );

            if( m_command.size() == 0 || !needPublish() )
            {
                continue;
            }

            zero = m_zeroRequested;

            if( zero )
            {
                clearState();
            }
            else
            {
                buildCommand( request.targetNs );
            }
        }

        const bool published = dmSyncopationProducerT::syncopationPublish( m_command.data(), request, m_auxStream );

        std::lock_guard<std::mutex> guard( m_stateMutex );

        if( !published )
        {
            ++m_failed;
            continue;
        }

        if( zero )
        {
            m_zeroRequested = false;
            log<text_log>( "zero submitted for syncopation" );
        }

        afterPublish();
    }

    dmSyncopationProducerT::syncopationClose();
}

inline int counterOffload::setNumber(
    pcf::IndiProperty &prop, const pcf::IndiProperty &ipRecv, double &value, double minVal, double maxVal )
{
    double target;

    if( indiTargetUpdate( prop, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) || target < minVal ||
        target > maxVal )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid " + prop.getName() } );
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        value = target;
    }

    updateIfChanged( prop, "current", value );
    updateIfChanged( prop, "target", value );

    log<text_log>( "set " + prop.getName() + " to " + std::to_string( value ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_enable )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_enable, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    const bool on = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;

    if( setEnabled( on ) )
    {
        log<text_log>( on ? "enabled" : "disabled, zero requested", logPrio::LOG_NOTICE );
    }

    updateSwitchIfChanged(
        m_indiP_enable, "toggle", on ? pcf::IndiElement::On : pcf::IndiElement::Off, on ? INDI_OK : INDI_IDLE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_zero )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_zero, ipRecv );

    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        m_zeroRequested = true;
        log<text_log>( "zero requested", logPrio::LOG_NOTICE );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_gain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gain, ipRecv );
    return setNumber( m_indiP_gain, ipRecv, m_gain, 0, 1 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_leak )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_leak, ipRecv );
    return setNumber( m_indiP_leak, ipRecv, m_leak, 0, 1 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_rampStart )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_rampStart, ipRecv );
    return setNumber( m_indiP_rampStart, ipRecv, m_rampStartUs, 0, 1e6 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_rampDuration )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_rampDuration, ipRecv );
    return setNumber( m_indiP_rampDuration, ipRecv, m_rampDurationUs, 1, 1e6 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_timeOffset )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_timeOffset, ipRecv );
    return setNumber( m_indiP_timeOffset, ipRecv, m_timeOffsetUs, -1e6, 1e6 );
}

} // namespace app
} // namespace MagAOX

#endif // counterOffload_hpp
