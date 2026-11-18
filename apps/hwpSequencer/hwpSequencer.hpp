/** \file hwpSequencer.hpp
 * \brief The MagAO-X HWP sequencer header file
 *
 * \ingroup hwpSequencer_files
 */

#ifndef hwpSequencer_hpp
#define hwpSequencer_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <atomic>
#include <cmath>
#include <sstream>

/** \defgroup hwpSequencer
 * \brief The MagAO-X application to step the HWP through a polarimetric sequence synchronized with camera frames.
 *
 * <a href="../handbook/operating/software/apps/hwpSequencer.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup hwpSequencer_files
 * \ingroup hwpSequencer
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X HWP sequencer
/** Steps the half-wave plate through a list of positions, pausing the camera trigger (a function generator channel)
 * around each move so that no frame is exposed while the HWP is moving.
 *
 * A single sequencer thread owns the whole sequence: trigger, HWP and observers commands, and all waits on camera
 * frames. INDI callbacks only validate input, set request flags, and wake the thread. Every wait is bounded and
 * checks for shutdown and abort, and every ending (completed, stopped, aborted, error, shutdown) goes through
 * finishSequence().
 *
 * See agents/plans/2026-11/hwpSequencer-sequencer-thread.md for the design record.
 *
 * \ingroup hwpSequencer
 */
class hwpSequencer : public MagAOXApp<true>, public dev::shmimMonitor<hwpSequencer>
{
    friend class dev::shmimMonitor<hwpSequencer>;

    typedef dev::shmimMonitor<hwpSequencer> shmimMonitorT;

  public:
    /// Outcome of a bounded wait.
    enum class waitResult
    {
        done,        ///< The awaited event happened (or the dwell deadline was reached).
        timeout,     ///< The wait timed out.
        stopped,     ///< A stop was requested (dwell only).
        interrupted, ///< An abort was requested or the app is shutting down.
        error        ///< A system call failed; the error has been logged.
    };

    /// Reason a sequence ended.
    enum class endReason
    {
        completed, ///< The requested number of cycles, or the last cycle, finished.
        stopped,   ///< A stop was requested, or the app is shutting down.
        aborted,   ///< An abort was requested.
        error      ///< A step failed.
    };

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_hwpTracker{ "hwptrack" }; ///< The hwpTracker device name, which owns `hwp_position`.

    std::string m_fxngenName{ "fxngensync" }; ///< The function generator device name that triggers the camera.

    std::string m_fxngenChannel{ "C2" }; ///< The function generator channel that triggers the camera.

    std::string m_obsAppName{ "observers" }; ///< The observerCtrl device name, which owns `obs_on`.

    std::vector<float> m_hwpPositions{ { 0.0, 45.0, 22.5, 67.5 } }; ///< The HWP positions of one cycle [deg].

    float m_hwpTolerance{ 0.5 }; ///< The tolerance for the HWP to be considered in position [deg].

    unsigned m_hwpTimeout{ 5000 }; ///< The time to wait for the HWP to reach a position [msec].

    unsigned m_hwpWait{ 100 }; ///< The polling interval while waiting for the HWP or the trigger state [msec].

    unsigned m_fxngenTimeout{ 5000 }; ///< The time to wait for the trigger output to confirm a change [msec].

    double m_frameMargin{ 0.5 }; ///< Margin added to the trigger period for a frame wait [sec].

    double m_frameMaxTimeout{ 610 }; ///< Maximum frame wait, and the fallback when the trigger rate is unknown [sec].

    ///@}

    /** \name Sequence Parameters - Data
     * Set over INDI. Both may change during a sequence (see the time_per_pos and num_cycles callbacks).
     *@{
     */

    std::atomic<double> m_timePerPos{ 0 }; ///< Dwell time at each position [sec], read at the start of each dwell.

    std::atomic<int> m_numCycles{ 0 }; ///< Number of cycles to run, 0 means run until stopped.

    ///@}

    /** \name Requests - Data
     * Set by INDI callbacks, consumed by the sequencer thread.
     *@{
     */

    std::atomic<bool> m_startRequested{ false }; ///< A start has been requested.

    std::atomic<bool> m_stopRequested{ false }; ///< A graceful stop has been requested.

    std::atomic<bool> m_abortRequested{ false }; ///< An abort has been requested.

    std::atomic<bool> m_lastCycle{ false }; ///< End the sequence at the end of the current cycle.

    ///@}

    /** \name Sequence State - Data
     *@{
     */

    std::atomic<bool> m_sequencing{ false }; ///< True from the start of a sequence until its cleanup finishes.

    std::atomic<size_t> m_posIndex{ 0 }; ///< Index of the current position in the cycle, written by the thread.

    std::atomic<int> m_cycle{ 0 }; ///< Current cycle number, starting at 1, written by the thread.

    std::atomic<bool> m_weStartedObs{ false }; ///< The sequencer turned `obs_on` On and still owns it.

    ///@}

    /** \name Remote Device State - Data
     * Last values received from other devices. Values are kept until updated; only "never received" blocks a start.
     *@{
     */

    std::atomic<float> m_reportedHwpPos{ 0 }; ///< Last `hwp_position.current` from the hwpTracker [deg].

    std::atomic<bool> m_haveHwpPos{ false }; ///< True once `hwp_position.current` has been received.

    std::atomic<bool> m_fxngenOutp{ false }; ///< Last trigger output state, true if On.

    std::atomic<bool> m_haveFxngenOutp{ false }; ///< True once the trigger output state has been received.

    std::atomic<double> m_fxngenFreq{ 0 }; ///< Last trigger frequency [Hz].

    std::atomic<bool> m_haveFxngenFreq{ false }; ///< True once the trigger frequency has been received.

    std::atomic<bool> m_obsOn{ false }; ///< Last `obs_on` state from observers.

    std::atomic<bool> m_haveObsOn{ false }; ///< True once `obs_on` has been received.

    ///@}

    /** \name Sequencer Thread - Data
     *@{
     */

    sem_t m_smSemaphore; ///< Posted by processImage() for each new frame; also posted by abort to wake frame waits.

    sem_t m_wakeSemaphore; ///< Posted by callbacks to wake the sequencer thread.

    bool m_sequencerThreadInit{ true }; ///< Initialization flag for the sequencer thread.

    pid_t m_sequencerThreadID{ 0 }; ///< Sequencer thread PID.

    pcf::IndiProperty m_sequencerThreadProp; ///< The property to hold the sequencer thread details.

    std::thread m_sequencerThread; ///< The sequencer thread.

    ///@}

  public:
    /// Default c'tor.
    hwpSequencer();

    /// D'tor, declared and defined for noexcept.
    ~hwpSequencer() noexcept;

    /// Setup the configuration system.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     *
     * \returns 0 on success
     * \returns -1 on an invalid configuration, which is logged
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration from which to load
                                                                    values*/ );

    /// Load the configuration.
    virtual void loadConfig();

    /// Startup function.
    /** Registers INDI properties and subscriptions, initializes the semaphores and starts the sequencer thread.
     *
     * \returns 0 on success
     * \returns -1 on an error requiring shutdown
     */
    virtual int appStartup();

    /// Implementation of the FSM for hwpSequencer.
    /** Only runs shmimMonitor housekeeping and checks that the sequencer thread is alive.
     *
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /** Wakes and joins the sequencer thread, letting it clean up, before stopping the shmimMonitor.
     *
     * \returns 0 always
     */
    virtual int appShutdown();

  protected:
    /** \name Sequencer
     *@{
     */

    /// Sequencer thread starter function.
    static void sequencerThreadStart( hwpSequencer *h /**< [in] pointer to this */ );

    /// Sequencer thread function.
    /** Waits for a start request, runs the sequence and cleans up, until shutdown.
     */
    void sequencerThreadExec();

    /// Check whether a sequence can start.
    /**
     * \returns true if all start conditions hold
     * \returns false otherwise, with the reason in \p why
     */
    bool canStart( std::string &why /**< [out] the reason a start is refused */ );

    /// Run the cycles of a sequence until it ends.
    /**
     * \returns the reason the sequence ended
     */
    endReason runSequence( const std::vector<float> &positions /**< [in] the positions snapshot for this sequence */ );

    /// Perform one step: pause the trigger, wait for the in-flight frame, move the HWP, resume the trigger.
    /** A step that has started always completes unless aborted, shut down, or failed.
     *
     * \returns 0 on success
     * \returns 1 if interrupted by abort or shutdown
     * \returns -1 on error, which is logged
     */
    int stepTo( float target, /**< [in] the HWP position for this step [deg] */
                bool  first   /**< [in] true for the first step of a sequence, which turns `obs_on` On */ );

    /// Clean up after a sequence ends, for any reason.
    /** Turns the trigger On; for a normal ending waits for the in-flight frame; turns `obs_on` Off if the sequencer
     * turned it On; returns the HWP to the first position unless the sequence ended in error; resets switches and
     * the FSM state.
     *
     * \returns 0 on success
     * \returns -1 if a step of the cleanup failed, which is logged
     */
    int finishSequence( endReason                 why,      /**< [in] the reason the sequence ended */
                        const std::vector<float> &positions /**< [in] the positions snapshot for this sequence */ );

    /// Publish the sequence status after a step.
    void publishStatus( size_t idx,   /**< [in] the position index */
                        int    cycle, /**< [in] the cycle number */
                        float  target /**< [in] the HWP position [deg] */ );

    /// Update the sequence, last_cycle and abort switches.
    void publishSwitches( bool sequencing /**< [in] the sequence switch state */ );

    /// Get the frame-wait timeout from the trigger frequency.
    /**
     * \returns the trigger period plus margin, capped at the maximum, or the maximum if the rate is unknown [sec]
     */
    double frameTimeout();

    /// Check whether waits should end early.
    /**
     * \returns true if an abort was requested or the app is shutting down
     */
    bool interrupted();

    /// Get the positions as a comma-separated string.
    /**
     * \returns the positions, e.g. "0,45,22.5,67.5"
     */
    std::string positionsString();

    ///@}

    /** \name Waits
     * Bounded waits used by the sequencer thread.
     *@{
     */

    /// Fill a timespec with an absolute CLOCK_REALTIME deadline.
    /**
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    int deadline( timespec &ts,     /**< [out] the deadline */
                  double    seconds /**< [in] seconds from now */ );

    /// Poll a condition until it holds, the wait times out, or it is stopped.
    /**
     * \returns waitResult::done when \p done is true
     * \returns waitResult::interrupted when \p stop is true
     * \returns waitResult::timeout otherwise after \p timeout
     */
    template <class doneT, class stopT>
    static waitResult waitFor( doneT    done,    /**< [in] returns true when the condition holds */
                               stopT    stop,    /**< [in] returns true when the wait should end early */
                               double   timeout, /**< [in] the timeout [sec] */
                               unsigned pollMs   /**< [in] the polling interval [msec] */ );

    ///@}

    /** \name Device Actions
     * The device-facing primitives of a step. They are virtual so unit tests can record and script them.
     *@{
     */

    /// Send a trigger output change to the function generator.
    /**
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    virtual int sendTrigger( bool on /**< [in] true for On, false for Off */ );

    /// Send a new HWP position to the hwpTracker.
    /**
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    virtual int sendHwpTarget( float target /**< [in] the HWP position [deg] */ );

    /// Send a new `obs_on` state to observers.
    /**
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    virtual int sendObsOn( bool on /**< [in] true for On, false for Off */ );

    /// Wait for the trigger output to confirm a state.
    /**
     * \returns the wait result
     */
    virtual waitResult waitTrigger( bool on /**< [in] the state to wait for */ );

    /// Wait for the HWP to report a position within tolerance.
    /**
     * \returns the wait result
     */
    virtual waitResult waitHwp( float target /**< [in] the HWP position [deg] */ );

    /// Wait for the next frame from the shmimMonitor.
    /**
     * \returns waitResult::done if a frame arrived, waitResult::timeout if none arrived, waitResult::interrupted
     *          on abort or shutdown, waitResult::error on a semaphore error
     */
    virtual waitResult waitFrame( double timeout /**< [in] the timeout [sec] */ );

    /// Dwell at a position.
    /**
     * \returns waitResult::done at the deadline, waitResult::stopped on a stop request, waitResult::interrupted on
     *          abort or shutdown, waitResult::error on a semaphore error
     */
    virtual waitResult dwell( double seconds /**< [in] the dwell time [sec] */ );

    ///@}

    /** \name shmimMonitor Interface
     *@{
     */

    /// Allocate for a new stream; nothing to allocate.
    /**
     * \returns 0 always
     */
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents */ );

    /// Signal a new frame to the sequencer thread.
    /**
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    int processImage( void              *curr_src, /**< [in] pointer to the frame, unused */
                      const dev::shmimT &dummy     /**< [in] tag to differentiate shmimMonitor parents */ );

    ///@}

    /** \name INDI - Data
     *@{
     */

    pcf::IndiProperty m_indiP_sequence; ///< Toggle to start (On) or gracefully stop (Off) a sequence.

    pcf::IndiProperty m_indiP_lastCycle; ///< Toggle to end the sequence at the end of the current cycle.

    pcf::IndiProperty m_indiP_abort; ///< Toggle to abort the sequence without waiting for the in-flight frame.

    pcf::IndiProperty m_indiP_timePerPos; ///< The dwell time at each position [sec].

    pcf::IndiProperty m_indiP_numCycles; ///< The number of cycles, 0 for until stopped.

    pcf::IndiProperty m_indiP_status; ///< Read-only sequence status: pos_index, cycle, target.

    pcf::IndiProperty m_indiP_positions; ///< Read-only list of configured positions.

    pcf::IndiProperty m_indiP_hwpTarget; ///< Template for sending `hwp_position.target` to the hwpTracker.

    pcf::IndiProperty m_indiP_hwpPosition; ///< Subscription to `hwp_position` from the hwpTracker.

    pcf::IndiProperty m_indiP_fxngenOutp; ///< Template for sending the trigger output state.

    pcf::IndiProperty m_indiP_fxngenOutpStatus; ///< Subscription to the trigger output state.

    pcf::IndiProperty m_indiP_fxngenFreq; ///< Subscription to the trigger frequency.

    pcf::IndiProperty m_indiP_obsOn; ///< Subscription to `obs_on` from observers.

    ///@}

  public:
    /** \name INDI
     *@{
     */

    INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_sequence );

    INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_lastCycle );

    INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_abort );

    INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_timePerPos );

    INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_numCycles );

    INDI_SETCALLBACK_DECL( hwpSequencer, m_indiP_hwpPosition );

    INDI_SETCALLBACK_DECL( hwpSequencer, m_indiP_fxngenOutpStatus );

    INDI_SETCALLBACK_DECL( hwpSequencer, m_indiP_fxngenFreq );

    INDI_SETCALLBACK_DECL( hwpSequencer, m_indiP_obsOn );

    ///@}
};

inline hwpSequencer::hwpSequencer() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline hwpSequencer::~hwpSequencer() noexcept
{
}

inline void hwpSequencer::setupConfig()
{
    config.add( "hwp.devName",
                "",
                "hwp.devName",
                argType::Required,
                "hwp",
                "devName",
                false,
                "string",
                "HWP tracker app name, default is 'hwptrack'" );

    config.add( "hwp.positions",
                "",
                "hwp.positions",
                argType::Required,
                "hwp",
                "positions",
                false,
                "vector<float>",
                "HWP positions of one cycle in order [deg], no repeats, default is 0,45,22.5,67.5" );

    config.add( "hwp.tolerance",
                "",
                "hwp.tolerance",
                argType::Required,
                "hwp",
                "tolerance",
                false,
                "float",
                "Tolerance for the HWP to be in position [deg], default is 0.5" );

    config.add( "hwp.timeout",
                "",
                "hwp.timeout",
                argType::Required,
                "hwp",
                "timeout",
                false,
                "unsigned",
                "Time to wait for the HWP to reach a position [msec], default is 5000" );

    config.add( "hwp.wait",
                "",
                "hwp.wait",
                argType::Required,
                "hwp",
                "wait",
                false,
                "unsigned",
                "Polling interval while waiting for the HWP or trigger state [msec], default is 100" );

    config.add( "fxngen.devName",
                "",
                "fxngen.devName",
                argType::Required,
                "fxngen",
                "devName",
                false,
                "string",
                "Function generator device name, default is 'fxngensync'." );

    config.add( "fxngen.channel",
                "",
                "fxngen.channel",
                argType::Required,
                "fxngen",
                "channel",
                false,
                "string",
                "Function generator output channel, default is 'C2'." );

    config.add( "fxngen.timeout",
                "",
                "fxngen.timeout",
                argType::Required,
                "fxngen",
                "timeout",
                false,
                "unsigned",
                "Time to wait for the trigger output to confirm a change [msec], default is 5000" );

    config.add( "frame.margin",
                "",
                "frame.margin",
                argType::Required,
                "frame",
                "margin",
                false,
                "double",
                "Margin added to the trigger period when waiting for a frame [sec], default is 0.5" );

    config.add( "frame.maxTimeout",
                "",
                "frame.maxTimeout",
                argType::Required,
                "frame",
                "maxTimeout",
                false,
                "double",
                "Maximum time to wait for a frame [sec], default is 610" );

    config.add( "observers.devName",
                "",
                "observers.devName",
                argType::Required,
                "observers",
                "devName",
                false,
                "string",
                "Observers application name, default is 'observers'" );

    SHMIMMONITOR_SETUP_CONFIG( config );
}

inline int hwpSequencer::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_hwpTracker, "hwp.devName" );
    _config( m_hwpPositions, "hwp.positions" );
    _config( m_hwpTolerance, "hwp.tolerance" );
    _config( m_hwpTimeout, "hwp.timeout" );
    _config( m_hwpWait, "hwp.wait" );
    _config( m_fxngenName, "fxngen.devName" );
    _config( m_fxngenChannel, "fxngen.channel" );
    _config( m_fxngenTimeout, "fxngen.timeout" );
    _config( m_frameMargin, "frame.margin" );
    _config( m_frameMaxTimeout, "frame.maxTimeout" );
    _config( m_obsAppName, "observers.devName" );

    SHMIMMONITOR_LOAD_CONFIG( _config );

    if( m_hwpPositions.empty() )
    {
        return log<text_log, -1>( "hwp.positions must not be empty", logPrio::LOG_CRITICAL );
    }

    for( size_t n = 0; n < m_hwpPositions.size(); ++n )
    {
        if( m_hwpPositions[n] < -360 || m_hwpPositions[n] > 360 )
        {
            return log<text_log, -1>( "hwp.positions values must be within -360 to 360 deg", logPrio::LOG_CRITICAL );
        }

        for( size_t m = 0; m < n; ++m )
        {
            if( m_hwpPositions[m] == m_hwpPositions[n] )
            {
                return log<text_log, -1>( "hwp.positions must not repeat a position", logPrio::LOG_CRITICAL );
            }
        }
    }

    if( m_hwpTolerance <= 0 || m_hwpTimeout == 0 || m_hwpWait == 0 || m_fxngenTimeout == 0 )
    {
        return log<text_log, -1>( "hwp.tolerance, hwp.timeout, hwp.wait and fxngen.timeout must be positive",
                                  logPrio::LOG_CRITICAL );
    }

    if( m_frameMargin < 0 || m_frameMaxTimeout <= 0 )
    {
        return log<text_log, -1>( "frame.margin must not be negative and frame.maxTimeout must be positive",
                                  logPrio::LOG_CRITICAL );
    }

    return 0;
}

inline void hwpSequencer::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = 1;
    }
}

inline int hwpSequencer::appStartup()
{
    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_sequence, "sequence" );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_lastCycle, "last_cycle" );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_abort, "abort" );

    CREATE_REG_INDI_NEW_NUMBERD(
        m_indiP_timePerPos, "time_per_pos", 1, 600, 0.1, "%0.1f", "Time per position [s]", "Sequencer" );
    m_indiP_timePerPos["current"].set( m_timePerPos.load() );
    m_indiP_timePerPos["target"].set( m_timePerPos.load() );

    CREATE_REG_INDI_NEW_NUMBERI(
        m_indiP_numCycles, "num_cycles", 0, 10800, 1, "%d", "Number of cycles (0 = until stopped)", "Sequencer" );
    m_indiP_numCycles["current"].set( m_numCycles.load() );
    m_indiP_numCycles["target"].set( m_numCycles.load() );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_status, "status", "Sequence status", "Sequencer" );
    m_indiP_status.add( pcf::IndiElement( "pos_index" ) );
    m_indiP_status["pos_index"].set( 0 );
    m_indiP_status.add( pcf::IndiElement( "cycle" ) );
    m_indiP_status["cycle"].set( 0 );
    m_indiP_status.add( pcf::IndiElement( "target" ) );
    m_indiP_status["target"].set( m_hwpPositions[0] );

    if( createROIndiText( m_indiP_positions, "positions", "value", "HWP positions [deg]", "Sequencer" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createROIndiText" } );
    }

    m_indiP_positions["value"].set( positionsString() );

    if( registerIndiPropertyReadOnly( m_indiP_positions ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyReadOnly" } );
    }

    m_indiP_hwpTarget = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_hwpTarget.setDevice( m_hwpTracker );
    m_indiP_hwpTarget.setName( "hwp_position" );
    m_indiP_hwpTarget.add( pcf::IndiElement( "target" ) );

    REG_INDI_SETPROP( m_indiP_hwpPosition, m_hwpTracker, "hwp_position" );

    m_indiP_fxngenOutp = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_fxngenOutp.setDevice( m_fxngenName );
    m_indiP_fxngenOutp.setName( m_fxngenChannel + "outp" );
    m_indiP_fxngenOutp.add( pcf::IndiElement( "value" ) );

    REG_INDI_SETPROP( m_indiP_fxngenOutpStatus, m_fxngenName, m_fxngenChannel + "outp" );

    REG_INDI_SETPROP( m_indiP_fxngenFreq, m_fxngenName, m_fxngenChannel + "freq" );

    REG_INDI_SETPROP( m_indiP_obsOn, m_obsAppName, "obs_on" );

    SHMIMMONITOR_APP_STARTUP;

    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Initializing frame semaphore" } );
    }

    if( sem_init( &m_wakeSemaphore, 0, 0 ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Initializing wake semaphore" } );
    }

    if( threadStart( m_sequencerThread,
                     m_sequencerThreadInit,
                     m_sequencerThreadID,
                     m_sequencerThreadProp,
                     0,
                     "",
                     "sequencer",
                     this,
                     sequencerThreadStart ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    state( stateCodes::READY );

    return 0;
}

inline int hwpSequencer::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;

    SHMIMMONITOR_UPDATE_INDI;

    // do a join check to see if the sequencer thread has exited.
    if( pthread_tryjoin_np( m_sequencerThread.native_handle(), 0 ) == 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "sequencer thread has exited" } );
    }

    return 0;
}

inline int hwpSequencer::appShutdown()
{
    // Join the sequencer thread first so its cleanup runs while the INDI client is still up.
    // Its waits check shutdown(), and SIGUSR1 interrupts sem_timedwait.
    if( m_sequencerThread.joinable() )
    {
        sem_post( &m_wakeSemaphore );

        pthread_kill( m_sequencerThread.native_handle(), SIGUSR1 );

        try
        {
            m_sequencerThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }

    SHMIMMONITOR_APP_SHUTDOWN;

    return 0;
}

inline void hwpSequencer::sequencerThreadStart( hwpSequencer *h )
{
    h->sequencerThreadExec();
}

inline void hwpSequencer::sequencerThreadExec()
{
    m_sequencerThreadID = syscall( SYS_gettid );

    // Wait for the thread starter to finish initializing this thread.
    while( m_sequencerThreadInit == true && shutdown() == 0 )
    {
        sleep( 1 );
    }

    while( shutdown() == 0 )
    {
        timespec ts;

        if( deadline( ts, 1.0 ) < 0 )
        {
            return;
        }

        // Timeouts and interruptions are expected; the request flags decide what to do.
        sem_timedwait( &m_wakeSemaphore, &ts );

        if( shutdown() != 0 )
        {
            break;
        }

        if( !m_startRequested.exchange( false ) )
        {
            continue;
        }

        std::string why;

        if( !canStart( why ) )
        {
            log<text_log>( "cannot start sequence: " + why, logPrio::LOG_WARNING );

            publishSwitches( false );

            continue;
        }

        std::vector<float> positions = m_hwpPositions;

        m_stopRequested  = false;
        m_abortRequested = false;
        m_lastCycle      = false;
        m_weStartedObs   = false;
        m_posIndex       = 0;
        m_cycle          = 0;
        m_sequencing     = true;

        // The shmimMonitor only reads frames while OPERATING, so stay OPERATING until cleanup is done.
        state( stateCodes::OPERATING );

        publishSwitches( true );

        int numCycles = m_numCycles;

        log<text_log>( "starting sequence: positions " + positionsString() + ", " +
                           std::to_string( m_timePerPos.load() ) + " s per position, " +
                           ( numCycles > 0 ? std::to_string( numCycles ) + " cycles" : std::string( "until stopped" ) ),
                       logPrio::LOG_NOTICE );

        endReason whyEnded = runSequence( positions );

        finishSequence( whyEnded, positions );
    }
}

inline bool hwpSequencer::canStart( std::string &why )
{
    if( m_hwpPositions.empty() )
    {
        why = "no positions configured";
        return false;
    }

    if( !m_haveHwpPos )
    {
        why = "no hwp_position received from " + m_hwpTracker;
        return false;
    }

    if( !m_haveFxngenOutp )
    {
        why = "no trigger output state received from " + m_fxngenName + "." + m_fxngenChannel + "outp";
        return false;
    }

    if( !m_haveFxngenFreq )
    {
        why = "no trigger frequency received from " + m_fxngenName + "." + m_fxngenChannel + "freq";
        return false;
    }

    double freq = m_fxngenFreq;

    if( freq <= 0 )
    {
        why = "trigger frequency is " + std::to_string( freq ) + " Hz";
        return false;
    }

    if( !m_haveObsOn )
    {
        why = "no obs_on received from " + m_obsAppName;
        return false;
    }

    if( m_obsOn )
    {
        why = m_obsAppName + " is already observing: stop observing first";
        return false;
    }

    double timePerPos = m_timePerPos;

    if( timePerPos <= 0 )
    {
        why = "time_per_pos is " + std::to_string( timePerPos ) + " s";
        return false;
    }

    if( timePerPos < 1.0 / freq )
    {
        why = "time_per_pos (" + std::to_string( timePerPos ) + " s) is shorter than one trigger period (" +
              std::to_string( 1.0 / freq ) + " s)";
        return false;
    }

    return true;
}

inline hwpSequencer::endReason hwpSequencer::runSequence( const std::vector<float> &positions )
{
    bool first = true;

    for( int cycle = 1;; ++cycle )
    {
        for( size_t idx = 0; idx < positions.size(); ++idx )
        {
            if( m_abortRequested )
            {
                return endReason::aborted;
            }

            if( m_stopRequested || shutdown() != 0 )
            {
                return endReason::stopped;
            }

            int rv = stepTo( positions[idx], first );

            if( rv < 0 )
            {
                return endReason::error;
            }

            if( rv > 0 )
            {
                return m_abortRequested ? endReason::aborted : endReason::stopped;
            }

            first = false;

            publishStatus( idx, cycle, positions[idx] );

            log<text_log>( "cycle " + std::to_string( cycle ) + " position " + std::to_string( idx ) + " (" +
                               std::to_string( positions[idx] ) + " deg): trigger on",
                           logPrio::LOG_INFO );

            waitResult wr = dwell( m_timePerPos );

            if( wr == waitResult::interrupted )
            {
                return m_abortRequested ? endReason::aborted : endReason::stopped;
            }

            if( wr == waitResult::stopped )
            {
                return endReason::stopped;
            }

            if( wr == waitResult::error )
            {
                return endReason::error;
            }
        }

        int numCycles = m_numCycles;

        log<text_log>( "cycle " + std::to_string( cycle ) +
                           ( numCycles > 0 ? " of " + std::to_string( numCycles ) : std::string( "" ) ) + " complete",
                       logPrio::LOG_INFO );

        if( m_lastCycle || ( numCycles > 0 && cycle >= numCycles ) )
        {
            return endReason::completed;
        }
    }
}

inline int hwpSequencer::stepTo( float target, bool first )
{
    // Pause the trigger and wait for the function generator to confirm it.
    if( sendTrigger( false ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to turn the trigger off" } );
    }

    waitResult wr = waitTrigger( false );

    if( wr == waitResult::interrupted )
    {
        return 1;
    }

    if( wr != waitResult::done )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "timed out waiting for the trigger to turn off" } );
    }

    // Frames that finished before the confirmed Off are discarded, so the next frame is the one in flight.
    XWC_SEM_FLUSH( m_smSemaphore );

    wr = waitFrame( frameTimeout() );

    if( wr == waitResult::interrupted )
    {
        return 1;
    }

    if( wr == waitResult::error )
    {
        return -1;
    }

    if( wr == waitResult::timeout )
    {
        log<text_log>( "no frame in flight after trigger off", logPrio::LOG_DEBUG );
    }

    // Move the HWP and wait for it to arrive.
    if( sendHwpTarget( target ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to send the HWP position" } );
    }

    wr = waitHwp( target );

    if( wr == waitResult::interrupted )
    {
        return 1;
    }

    if( wr != waitResult::done )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "timed out waiting for the HWP to reach " + std::to_string( target ) + " deg" } );
    }

    // Resume the trigger.
    if( sendTrigger( true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to turn the trigger on" } );
    }

    wr = waitTrigger( true );

    if( wr == waitResult::interrupted )
    {
        return 1;
    }

    if( wr != waitResult::done )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "timed out waiting for the trigger to turn on" } );
    }

    // Start saving once the HWP is at the first position.
    if( first )
    {
        if( sendObsOn( true ) < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "failed to turn obs_on on" } );
        }

        m_weStartedObs = true;
    }

    return 0;
}

inline int hwpSequencer::finishSequence( endReason why, const std::vector<float> &positions )
{
    int rv = 0;

    // Leave the camera triggering.
    if( sendTrigger( true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "failed to turn the trigger on during cleanup" } );
        rv = -1;
    }
    else if( waitTrigger( true ) == waitResult::timeout )
    {
        log<text_log>( "trigger did not confirm on during cleanup", logPrio::LOG_WARNING );
        rv = -1;
    }

    // Let the in-flight frame land before observers stops, unless aborted, failed or shutting down.
    if( ( why == endReason::completed || why == endReason::stopped ) && !interrupted() )
    {
        XWC_SEM_FLUSH( m_smSemaphore );

        if( waitFrame( frameTimeout() ) == waitResult::timeout )
        {
            log<text_log>( "no final frame arrived during cleanup", logPrio::LOG_WARNING );
        }
    }

    if( m_weStartedObs.exchange( false ) )
    {
        if( sendObsOn( false ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "failed to turn obs_on off during cleanup" } );
            rv = -1;
        }
    }

    // Prepare for the next sequence, but leave the HWP in place after an error for diagnosis.
    if( why != endReason::error && !positions.empty() )
    {
        if( sendHwpTarget( positions[0] ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "failed to return the HWP to the first position" } );
            rv = -1;
        }
    }

    m_lastCycle      = false;
    m_stopRequested  = false;
    m_abortRequested = false;
    m_sequencing     = false;

    publishSwitches( false );

    switch( why )
    {
    case endReason::completed:
        log<text_log>( "sequence complete", logPrio::LOG_NOTICE );
        break;
    case endReason::stopped:
        log<text_log>( "sequence stopped", logPrio::LOG_NOTICE );
        break;
    case endReason::aborted:
        log<text_log>( "sequence aborted", logPrio::LOG_NOTICE );
        break;
    case endReason::error:
        log<text_log>( "sequence ended with an error", logPrio::LOG_ERROR );
        break;
    }

    // An error leaves a sticky FSM alert until an operator clears it.
    state( stateCodes::READY, why == endReason::error );

    return rv;
}

inline void hwpSequencer::publishStatus( size_t idx, int cycle, float target )
{
    m_posIndex = idx;
    m_cycle    = cycle;

    std::lock_guard<std::mutex> guard( m_indiMutex );

    updatesIfChanged<double>(
        m_indiP_status,
        { "pos_index", "cycle", "target" },
        { static_cast<double>( idx ), static_cast<double>( cycle ), static_cast<double>( target ) } );
}

inline void hwpSequencer::publishSwitches( bool sequencing )
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    updateSwitchIfChanged( m_indiP_sequence,
                           "toggle",
                           sequencing ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           sequencing ? INDI_BUSY : INDI_IDLE );

    updateSwitchIfChanged( m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE );

    updateSwitchIfChanged( m_indiP_abort, "toggle", pcf::IndiElement::Off, INDI_IDLE );
}

inline double hwpSequencer::frameTimeout()
{
    double freq = m_fxngenFreq;

    if( freq <= 0 )
    {
        return m_frameMaxTimeout;
    }

    return std::min( 1.0 / freq + m_frameMargin, m_frameMaxTimeout );
}

inline bool hwpSequencer::interrupted()
{
    return m_abortRequested || shutdown() != 0;
}

inline std::string hwpSequencer::positionsString()
{
    std::ostringstream str;

    for( size_t n = 0; n < m_hwpPositions.size(); ++n )
    {
        if( n > 0 )
        {
            str << ",";
        }

        str << m_hwpPositions[n];
    }

    return str.str();
}

inline int hwpSequencer::deadline( timespec &ts, double seconds )
{
    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    if( seconds < 0 )
    {
        seconds = 0;
    }

    time_t sec = static_cast<time_t>( seconds );

    ts.tv_sec += sec;
    mx::sys::timespecAddNsec( ts, static_cast<unsigned>( ( seconds - sec ) * 1e9 ) );

    return 0;
}

template <class doneT, class stopT>
hwpSequencer::waitResult hwpSequencer::waitFor( doneT done, stopT stop, double timeout, unsigned pollMs )
{
    double t0 = mx::sys::get_curr_time();

    while( !done() )
    {
        if( stop() )
        {
            return waitResult::interrupted;
        }

        if( mx::sys::get_curr_time() - t0 > timeout )
        {
            return waitResult::timeout;
        }

        mx::sys::milliSleep( pollMs );
    }

    return waitResult::done;
}

inline int hwpSequencer::sendTrigger( bool on )
{
    return sendNewProperty( m_indiP_fxngenOutp, "value", std::string( on ? "On" : "Off" ) );
}

inline int hwpSequencer::sendHwpTarget( float target )
{
    return sendNewProperty( m_indiP_hwpTarget, "target", target );
}

inline int hwpSequencer::sendObsOn( bool on )
{
    return sendNewStandardIndiToggle( m_obsAppName, "obs_on", on );
}

inline hwpSequencer::waitResult hwpSequencer::waitTrigger( bool on )
{
    return waitFor( [this, on]() { return m_haveFxngenOutp && m_fxngenOutp == on; },
                    [this]() { return interrupted(); },
                    m_fxngenTimeout / 1000.0,
                    m_hwpWait );
}

inline hwpSequencer::waitResult hwpSequencer::waitHwp( float target )
{
    return waitFor( [this, target]() { return std::fabs( m_reportedHwpPos.load() - target ) <= m_hwpTolerance; },
                    [this]() { return interrupted(); },
                    m_hwpTimeout / 1000.0,
                    m_hwpWait );
}

inline hwpSequencer::waitResult hwpSequencer::waitFrame( double timeout )
{
    timespec ts;

    if( deadline( ts, timeout ) < 0 )
    {
        return waitResult::error;
    }

    while( true )
    {
        if( interrupted() )
        {
            return waitResult::interrupted;
        }

        if( sem_timedwait( &m_smSemaphore, &ts ) == 0 )
        {
            // abort also posts this semaphore, so check before calling it a frame
            if( interrupted() )
            {
                return waitResult::interrupted;
            }

            return waitResult::done;
        }

        if( errno == EINTR )
        {
            continue;
        }

        if( errno == ETIMEDOUT )
        {
            return waitResult::timeout;
        }

        log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );

        return waitResult::error;
    }
}

inline hwpSequencer::waitResult hwpSequencer::dwell( double seconds )
{
    timespec ts;

    if( deadline( ts, seconds ) < 0 )
    {
        return waitResult::error;
    }

    while( true )
    {
        if( interrupted() )
        {
            return waitResult::interrupted;
        }

        if( m_stopRequested )
        {
            return waitResult::stopped;
        }

        // Wake-ups from other requests just loop back to the checks above.
        if( sem_timedwait( &m_wakeSemaphore, &ts ) == 0 || errno == EINTR )
        {
            continue;
        }

        if( errno == ETIMEDOUT )
        {
            return waitResult::done;
        }

        log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );

        return waitResult::error;
    }
}

inline int hwpSequencer::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

inline int hwpSequencer::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( curr_src );
    static_cast<void>( dummy );

    if( sem_post( &m_smSemaphore ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_sequence )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sequence, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( m_sequencing )
        {
            return 0;
        }

        m_startRequested = true;

        { //mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );
            updateSwitchIfChanged( m_indiP_sequence, "toggle", pcf::IndiElement::On, INDI_BUSY );
        } //mutex scope
    }
    else
    {
        m_startRequested = false;

        if( !m_sequencing )
        {
            return 0;
        }

        m_stopRequested = true;

        log<text_log>( "stop requested", logPrio::LOG_NOTICE );
    }

    sem_post( &m_wakeSemaphore );

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_lastCycle )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_lastCycle, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_sequencing )
        {
            log<text_log>( "last_cycle ignored: not sequencing", logPrio::LOG_NOTICE );

            std::lock_guard<std::mutex> guard( m_indiMutex );
            updateSwitchIfChanged( m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE );

            return 0;
        }

        m_lastCycle = true;

        size_t nPos      = m_hwpPositions.size();
        size_t idx       = m_posIndex;
        size_t remaining = ( idx + 1 < nPos ) ? nPos - idx - 1 : 0;

        log<text_log>( "last cycle: will finish after " + std::to_string( remaining ) + " more positions (approx. " +
                           std::to_string( ( remaining + 1 ) * m_timePerPos.load() ) + " s)",
                       logPrio::LOG_INFO );

        std::lock_guard<std::mutex> guard( m_indiMutex );
        updateSwitchIfChanged( m_indiP_lastCycle, "toggle", pcf::IndiElement::On, INDI_BUSY );
    }
    else
    {
        m_lastCycle = false;

        log<text_log>( "last cycle cancelled", logPrio::LOG_INFO );

        std::lock_guard<std::mutex> guard( m_indiMutex );
        updateSwitchIfChanged( m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_abort )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_abort, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    if( !m_sequencing )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );
        updateSwitchIfChanged( m_indiP_abort, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        return 0;
    }

    m_abortRequested = true;

    log<text_log>( "abort requested", logPrio::LOG_NOTICE );

    // Wake whatever the sequencer thread is waiting on; every wait checks the abort flag after waking.
    sem_post( &m_wakeSemaphore );
    sem_post( &m_smSemaphore );

    std::lock_guard<std::mutex> guard( m_indiMutex );
    updateSwitchIfChanged( m_indiP_abort, "toggle", pcf::IndiElement::On, INDI_BUSY );

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_timePerPos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_timePerPos, ipRecv );

    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    double target = ipRecv["target"].get<double>();
    double freq   = m_fxngenFreq;

    std::string reject;

    if( target <= 0 )
    {
        reject = "time_per_pos must be positive";
    }
    else if( m_sequencing && freq > 0 && target < 1.0 / freq )
    {
        reject = "time_per_pos (" + std::to_string( target ) + " s) is shorter than one trigger period (" +
                 std::to_string( 1.0 / freq ) + " s)";
    }

    if( reject != "" )
    {
        log<text_log>( reject, logPrio::LOG_WARNING );
    }
    else
    {
        m_timePerPos = target;

        log<text_log>( "time_per_pos set to " + std::to_string( target ) + " s" +
                           ( m_sequencing ? std::string( ", applies from the next dwell" ) : std::string( "" ) ),
                       logPrio::LOG_NOTICE );
    }

    double current = m_timePerPos;

    std::lock_guard<std::mutex> guard( m_indiMutex );
    updatesIfChanged<double>( m_indiP_timePerPos, { "current", "target" }, { current, current } );

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_numCycles )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numCycles, ipRecv );

    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    int target = ipRecv["target"].get<int>();

    if( target < 0 )
    {
        log<text_log>( "num_cycles cannot be negative (0 runs until stopped)", logPrio::LOG_WARNING );
    }
    else
    {
        m_numCycles = target;

        log<text_log>(
            "num_cycles set to " + std::to_string( target ) +
                ( m_sequencing ? std::string( ", applies at the end of the current cycle" ) : std::string( "" ) ),
            logPrio::LOG_NOTICE );
    }

    int current = m_numCycles;

    std::lock_guard<std::mutex> guard( m_indiMutex );
    updatesIfChanged<int>( m_indiP_numCycles, { "current", "target" }, { current, current } );

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpSequencer, m_indiP_hwpPosition )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_hwpPosition, ipRecv );

    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    m_reportedHwpPos = ipRecv["current"].get<float>();
    m_haveHwpPos     = true;

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpSequencer, m_indiP_fxngenOutpStatus )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fxngenOutpStatus, ipRecv );

    if( !ipRecv.find( "value" ) )
    {
        return 0;
    }

    std::string outp = ipRecv["value"].get<std::string>();

    if( outp == "On" )
    {
        m_fxngenOutp     = true;
        m_haveFxngenOutp = true;
    }
    else if( outp == "Off" )
    {
        m_fxngenOutp     = false;
        m_haveFxngenOutp = true;
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpSequencer, m_indiP_fxngenFreq )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fxngenFreq, ipRecv );

    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    m_fxngenFreq     = ipRecv["current"].get<double>();
    m_haveFxngenFreq = true;

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpSequencer, m_indiP_obsOn )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_obsOn, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    bool on = ( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On );

    m_obsOn     = on;
    m_haveObsOn = true;

    // If observing stops mid-sequence (obs_duration or an operator), keep sequencing but give up ownership.
    if( !on && m_sequencing && m_weStartedObs.exchange( false ) )
    {
        log<text_log>( "obs_on turned off during the sequence; continuing without saving", logPrio::LOG_WARNING );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // hwpSequencer_hpp
