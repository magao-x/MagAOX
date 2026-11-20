/** \file counterOffload.hpp
 * \brief The MagAO-X tweeter counter-offload producer header file
 *
 * \ingroup counterOffload_files
 */

#ifndef counterOffload_hpp
#define counterOffload_hpp

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
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

/// Tag for the closed-loop PSD stream monitor used by calibrate mode.
/**
 * \ingroup counterOffload
 */
struct psdShmimT
{
    /// Config section of the PSD stream monitor.
    static std::string configSection()
    {
        return "psds";
    };

    /// INDI prefix of the PSD stream monitor.
    static std::string indiPrefix()
    {
        return "psds";
    };
};

/// MagAO-X application publishing tweeter counter-offload commands as a dmSyncopation producer.
/** Each update of the input stream (normally `aol0_tweeterOffload`) starts one ramp of that offload change.
 * For each scheduler slot the published command is `-(A + sum_k gain_k * delta_k * ramp(target - t0_k))`, where
 * gain_k is the gain when offload k arrived, completed ramps are folded into the accumulator `A`, which is multiplied
 * by `(1 - leak)` after every successful publication, and `A` and the command are clipped to +/- maxCommand.  A gain
 * change therefore affects only new offloads.  Disabling stops accepting new offloads and requests a zero, so the contribution is
 * cleared at the next slot.
 *
 * Calibrate mode scans rampStartUs, rampDurationUs and gain to minimize the closed-loop WFS PSD power in a band around
 * the offload rate, summed over the offloaded modes, relative to a gain-0 baseline, and applies the best values.
 *
 * \ingroup counterOffload
 */
class counterOffload : public MagAOXApp<true>,
                       public dev::shmimMonitor<counterOffload>,
                       public dev::shmimMonitor<counterOffload, psdShmimT>,
                       public dev::dmSyncopationProducer<counterOffload>
{
    typedef float realT; ///< Floating point type of the tweeter commands.

    typedef dev::shmimMonitor<counterOffload> shmimMonitorT; ///< The input stream monitor base.

    typedef dev::shmimMonitor<counterOffload, psdShmimT> psdShmimMonitorT; ///< The PSD stream monitor base.

    typedef dev::dmSyncopationProducer<counterOffload> dmSyncopationProducerT; ///< The syncopation producer base.

    friend class dev::shmimMonitor<counterOffload>;

    friend class dev::shmimMonitor<counterOffload, psdShmimT>;

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

    /// Calibrated parameters, in order: rampStartUs, rampDurationUs, gain.
    typedef std::array<double, 3> calibParamsT;

    /// Calibration evaluator: parameters, axis (0-2), pass (0 coarse, 1 refine), and the ratio to fill in.
    /** Returns false to abort the scan.
     */
    typedef std::function<bool( const calibParamsT &, int, int, double & )> calibEvalT;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_channel{ "dm01disp08_aux" }; ///< The dmSyncopation auxiliary source this app writes.

    double m_rampStartUs{ 120 }; ///< Ramp start after the woofer command, in us.

    double m_rampDurationUs{ 330 }; ///< Ramp duration, in us.

    double m_leak{ 0 }; ///< Fraction of the completed contribution removed after each published slot, 0 to 1.

    double m_timeOffsetUs{ 0 }; ///< Offset added to the input receipt time to give the ramp reference, in us.

    double m_gain{ 1 }; ///< Scale applied to each new offload when it arrives, 0 to 2.

    double m_maxCommand{ 0.5 }; ///< Largest magnitude of any pixel of the command and of the accumulator.

    double m_computeLeadUs{ 20 }; ///< Time reserved before the publish lead to compute the command, in us.

    int m_threadPrio{ 20 }; ///< Real-time priority of the producer thread.

    std::string m_threadCpuset; ///< The cpuset for the producer thread.

    ///@}

    /** \name Calibrate Parameters - Data
     *@{
     */

    std::string m_freqStream{ "aol1_freq" }; ///< Stream holding the frequency axis of the PSD stream, Hz.

    std::string m_offloadDevice{ "t2wOffloader" }; ///< INDI device providing the offload rate (fps) and numModes.

    double m_halfWidthHz{ 10 }; ///< Half width of the band integrated around the offload rate, Hz.

    int m_modeMax{ 0 }; ///< Number of PSD modes summed.  0 uses the offloader's numModes.

    int m_nAverage{ 3 }; ///< Settled averaged PSDs, published 0.5 s apart, averaged for each evaluation, at least 2.

    int m_baselineEvery{ 5 }; ///< Re-measure the gain-0 baseline before every this many evaluations.

    double m_settleS{ 11 }; ///< Wait after a change before PSDs are collected; covers the modalPSDs average, s.

    double m_timeoutS{ 5 }; ///< Longest wait for the PSDs of one evaluation, after settling, s.

    double m_abortRatio{ 3 }; ///< Abort when an evaluation exceeds this multiple of the baseline.

    double m_calibLeak{ 0.0025 }; ///< Leak used during a calibration run, restored afterwards; must be > 0.

    std::vector<double> m_startGridUs{ 0, 400, 25 }; ///< rampStartUs scan: min, max, step, in us.

    std::vector<double> m_durationGridUs{ 100, 600, 50 }; ///< rampDurationUs scan: min, max, step, in us.

    std::vector<double> m_gainGrid{ 0.1, 1.0, 0.1 }; ///< gain scan: min, max, step.

    std::string m_csvDir{ "/home/xsup/counterOffload" }; ///< Directory for the per-run calibration CSV files.

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

    std::atomic<uint64_t> m_clipped{ 0 }; ///< Slots in which the command or the accumulator was clipped.

    uint64_t m_clippedLogged{ 0 }; ///< m_clipped at the last clipping warning.  appLogic only.

    uint64_t m_calibClipStart{ 0 }; ///< m_clipped at the start of the current calibration run.

    ///@}

    /** \name Calibrate State - Data
     *@{
     */

    std::mutex m_calibMutex; ///< Protects the band, collected PSD powers, offloader values and calibration readbacks.

    std::condition_variable m_calibCond; ///< Signals that a PSD band power was collected.

    std::atomic<bool> m_calibStart{ false }; ///< Request to start a run, consumed by the calibration thread.

    std::atomic<bool> m_calibRunning{ false }; ///< Whether a calibration run is requested or in progress.

    std::atomic<bool> m_calibAbort{ false }; ///< Request to abort the run and restore the original parameters.

    float m_offloadFps{ 0 }; ///< Effective offload rate reported by the offloader, Hz.

    uint32_t m_offloadModes{ 0 }; ///< Number of offloaded modes reported by the offloader.

    uint32_t m_psdNFreq{ 0 }; ///< Number of frequencies in each PSD.

    uint32_t m_psdNModes{ 0 }; ///< Number of modes in each PSD.

    bool m_bandValid{ false }; ///< Whether a band is set and PSD band powers are collected.

    double m_df{ 0 }; ///< PSD frequency resolution for the current run, Hz.

    double m_bandCenterHz{ 0 }; ///< Band center for the current run: the offload rate, Hz.

    double m_bandHalfWidthHz{ 0 }; ///< Band half width for the current run, Hz.

    uint32_t m_bandModes{ 0 }; ///< Number of modes summed in the current run.

    int64_t m_psdAfterNs{ 0 }; ///< Only PSDs received after this CLOCK_MONOTONIC time, in ns, are collected.

    std::vector<double> m_psdPowers; ///< Band powers of the PSDs collected for the current evaluation.

    double m_fomOff{ 0 }; ///< Latest gain-0 baseline band power.

    int m_calibStage{ 0 }; ///< Stage of the latest evaluation, an index into calibStageName().

    int m_calibPoint{ 0 }; ///< Evaluations completed in the current run.

    double m_lastFom{ 0 }; ///< Band power of the latest evaluation.

    double m_lastRatio{ 0 }; ///< Ratio to the baseline of the latest evaluation.

    calibParamsT m_calibBest{ 0, 0, 0 }; ///< Best parameters so far in the current or last run.

    double m_calibBestRatio{ 0 }; ///< Ratio to the baseline of m_calibBest.

    int m_evalCount{ 0 }; ///< Evaluations in the current run, used to schedule baselines.  Calibration thread only.

    std::ofstream m_csv; ///< The current run's CSV file.  Calibration thread only.

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

    /// Record the PSD stream geometry.  Aborts a running calibration, since its band no longer applies.
    /**
     * \returns 0 on success
     * \returns -1 if the stream is not float
     */
    int allocate( const psdShmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Collect the band power of a new PSD while a calibration evaluation is waiting for it.
    /**
     * \returns 0 always
     */
    int processImage( void            *curr_src, /**< [in] pointer to the start of the current PSD.*/
                      const psdShmimT &dummy     /**< [in] tag to differentiate shmimMonitor parents.*/
    );

    /// Integrate a PSD over a band and the first modes.
    /** The PSD is nFreq x nModes, frequency fastest, so element (n, m) is psd[n + m * nFreq] at frequency n * df.
     *
     * \returns sum of PSD * df over |f - centerHz| <= halfWidthHz and modes below min(modeMax, nModes)
     * \returns -1 if the band or the mode range is empty
     */
    static double bandPower( const realT *psd /**< [in] the PSD */,
                             uint32_t     nFreq /**< [in] number of frequencies */,
                             uint32_t     nModes /**< [in] number of modes */,
                             double       df /**< [in] frequency resolution, Hz */,
                             double       centerHz /**< [in] band center, Hz */,
                             double       halfWidthHz /**< [in] band half width, Hz */,
                             uint32_t     modeMax /**< [in] number of modes to sum */ );

    /// Coordinate scan minimizing an evaluator over the three calibrated parameters.
    /** Pass 0 scans each parameter in turn over its grid, holding the others at the best so far.  Pass 1 rescans each
     * over +/- one step at half the step, then evaluates the vertex of a parabola through the minimum and its
     * neighbours.  best and bestFom must hold an evaluated starting point, and are updated as better points are found.
     *
     * \returns true if the scan finished
     * \returns false if the evaluator aborted
     */
    static bool
    coordinateScan( const std::array<std::vector<double>, 3> &grids /**< [in] min, max, step per parameter */,
                    calibParamsT                             &best /**< [in,out] best parameters */,
                    double                                   &bestFom /**< [in,out] value at best */,
                    const calibEvalT                         &eval /**< [in] the evaluator */ );

    /// Get the current calibrated parameters.
    calibParamsT calibParams();

    /// Set the calibrated parameters.
    void setCalibParams( const calibParamsT &p /**< [in] rampStartUs, rampDurationUs, gain */ );

    /// Name of a calibration stage, for logs and the CSV.
    static const char *calibStageName( int stage /**< [in] the stage index */ );

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

    /** \name Calibration Thread
     * Runs calibrate mode, outside the real-time producer path.
     *
     * @{
     */

    std::thread m_calibThread; ///< The calibration thread.

    bool m_calibThreadInit{ true }; ///< Synchronizer to ensure the thread initializes before doing dangerous things.

    pid_t m_calibThreadID{ 0 }; ///< Calibration thread PID.

    pcf::IndiProperty m_calibThreadProp; ///< The property to hold the calibration thread details.

    /// Calibration thread starter function.
    static void calibThreadStart( counterOffload *c /**< [in] pointer to this */ );

    /// Calibration thread function.  Runs a calibration on each start request until m_shutdown is true.
    void calibThreadExec();

    /// Run one calibration, then apply the best parameters, or restore the originals on abort.
    void runCalibration();

    /// Set the band for a run from the frequency stream and the offloader's rate and numModes.
    /**
     * \returns true on success
     */
    bool setupBand();

    /// Open the run's CSV file in m_csvDir and write the header.
    /**
     * \returns true on success
     */
    bool openCsv();

    /// Apply parameters, wait for settled PSDs, and average their band powers.
    /**
     * \returns true on success
     * \returns false on abort, disable, timeout or a change of the offloader's rate or numModes
     */
    bool measure( const calibParamsT &p /**< [in] parameters to apply */,
                  double             &fom /**< [out] mean band power */,
                  double             &fomStd /**< [out] standard deviation of the band powers */ );

    /// Measure the gain-0 baseline at the given ramp, and record it.
    /**
     * \returns true on success
     */
    bool measureBaseline( const calibParamsT &p /**< [in] parameters; the gain is replaced by 0 */ );

    /// Evaluate parameters as a ratio to the baseline, re-measuring the baseline when due.
    /**
     * \returns true on success
     * \returns false to abort the scan
     */
    bool evaluate( const calibParamsT &p /**< [in] parameters */,
                   int                 stage /**< [in] stage index, see calibStageName() */,
                   double             &ratio /**< [out] band power over the baseline */ );

    /// Write one CSV row.
    void writeCsv( int                 stage /**< [in] stage index */,
                   const calibParamsT &p /**< [in] parameters */,
                   double              fom /**< [in] mean band power */,
                   double              fomStd /**< [in] standard deviation of the band powers */ );

    ///@}

    /** \name INDI Interface
     *
     * @{
     */

    pcf::IndiProperty m_indiP_enable; ///< Toggle switch to accept offloads.

    pcf::IndiProperty m_indiP_zero; ///< Request switch to clear and publish a zero contribution.

    pcf::IndiProperty m_indiP_gain; ///< Property to set the command gain.

    pcf::IndiProperty m_indiP_leak; ///< Property to set the leak.

    pcf::IndiProperty m_indiP_maxCommand; ///< Property to set the command clip level.

    pcf::IndiProperty m_indiP_rampStart; ///< Property to set the ramp start.

    pcf::IndiProperty m_indiP_rampDuration; ///< Property to set the ramp duration.

    pcf::IndiProperty m_indiP_timeOffset; ///< Property to set the ramp reference time offset.

    pcf::IndiProperty m_indiP_stats; ///< Read-only counters.

    pcf::IndiProperty m_indiP_calibrate; ///< Toggle switch: On starts a calibration run, Off aborts it.

    pcf::IndiProperty m_indiP_halfWidth; ///< Property to set the calibration band half width.

    pcf::IndiProperty m_indiP_calibration; ///< Read-only calibration progress and best parameters.

    pcf::IndiProperty m_indiP_offloadFps; ///< The offloader's effective offload rate (SET).

    pcf::IndiProperty m_indiP_offloadModes; ///< The offloader's numModes (SET).

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

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_maxCommand );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_rampStart );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_rampDuration );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_timeOffset );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_calibrate );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_halfWidth );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_offloadFps );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_offloadModes );

    ///@}
};

inline counterOffload::counterOffload() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_ramps.resize( m_maxRamps );
}

inline void counterOffload::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    SHMIMMONITORT_SETUP_CONFIG( psdShmimMonitorT, config );

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
                "Scale applied to each new offload when it arrives, 0 to 2.  Default is 1." );

    config.add( "counterOffload.maxCommand",
                "",
                "counterOffload.maxCommand",
                argType::Required,
                "counterOffload",
                "maxCommand",
                false,
                "real",
                "Largest magnitude of any pixel of the command and of the held contribution; larger values are clipped.  "
                "Default is 0.5." );

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

    config.add( "calibrate.freqStream",
                "",
                "calibrate.freqStream",
                argType::Required,
                "calibrate",
                "freqStream",
                false,
                "string",
                "Stream holding the PSD frequency axis.  Default is aol1_freq." );

    config.add( "calibrate.offloadDevice",
                "",
                "calibrate.offloadDevice",
                argType::Required,
                "calibrate",
                "offloadDevice",
                false,
                "string",
                "INDI device providing fps and numModes.  Default is t2wOffloader." );

    config.add( "calibrate.halfWidthHz",
                "",
                "calibrate.halfWidthHz",
                argType::Required,
                "calibrate",
                "halfWidthHz",
                false,
                "real",
                "Half width of the band integrated around the offload rate, Hz.  Default is 10." );

    config.add( "calibrate.modeMax",
                "",
                "calibrate.modeMax",
                argType::Required,
                "calibrate",
                "modeMax",
                false,
                "int",
                "Number of PSD modes summed.  0, the default, uses the offloader's numModes." );

    config.add( "calibrate.nAverage",
                "",
                "calibrate.nAverage",
                argType::Required,
                "calibrate",
                "nAverage",
                false,
                "int",
                "Settled averaged PSDs averaged for each evaluation, at least 2.  Default is 3." );

    config.add( "calibrate.baselineEvery",
                "",
                "calibrate.baselineEvery",
                argType::Required,
                "calibrate",
                "baselineEvery",
                false,
                "int",
                "Re-measure the gain-0 baseline every this many evaluations.  Default is 5." );

    config.add( "calibrate.settleS",
                "",
                "calibrate.settleS",
                argType::Required,
                "calibrate",
                "settleS",
                false,
                "real",
                "Wait after a change before PSDs are collected, s.  Must cover the modalPSDs average (psdAvgTime + "
                "psdTime, 10.5 s by default).  Default is 11." );

    config.add( "calibrate.timeoutS",
                "",
                "calibrate.timeoutS",
                argType::Required,
                "calibrate",
                "timeoutS",
                false,
                "real",
                "Longest wait for the PSDs of one evaluation after settling, s.  Default is 5." );

    config.add( "calibrate.abortRatio",
                "",
                "calibrate.abortRatio",
                argType::Required,
                "calibrate",
                "abortRatio",
                false,
                "real",
                "Abort when an evaluation exceeds this multiple of the baseline.  Default is 3." );

    config.add( "calibrate.leak",
                "",
                "calibrate.leak",
                argType::Required,
                "calibrate",
                "leak",
                false,
                "real",
                "Leak used during a calibration run and restored afterwards, above 0.  Default is 0.0025." );

    config.add( "calibrate.startGridUs",
                "",
                "calibrate.startGridUs",
                argType::Required,
                "calibrate",
                "startGridUs",
                false,
                "vector<double>",
                "rampStartUs scan as min,max,step in us.  Default is 0,400,25." );

    config.add( "calibrate.durationGridUs",
                "",
                "calibrate.durationGridUs",
                argType::Required,
                "calibrate",
                "durationGridUs",
                false,
                "vector<double>",
                "rampDurationUs scan as min,max,step in us.  Default is 100,600,50." );

    config.add( "calibrate.gainGrid",
                "",
                "calibrate.gainGrid",
                argType::Required,
                "calibrate",
                "gainGrid",
                false,
                "vector<double>",
                "gain scan as min,max,step.  Default is 0.1,1.0,0.1." );

    config.add( "calibrate.csvDir",
                "",
                "calibrate.csvDir",
                argType::Required,
                "calibrate",
                "csvDir",
                false,
                "string",
                "Directory for the per-run calibration CSV files.  Default is /home/xsup/counterOffload." );
}

inline int counterOffload::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::m_shmimName = "aol0_tweeterOffload";

    SHMIMMONITOR_LOAD_CONFIG( _config );

    psdShmimMonitorT::m_shmimName = "aol1_clpsds";

    SHMIMMONITORT_LOAD_CONFIG( psdShmimMonitorT, _config );

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
    _config( m_maxCommand, "counterOffload.maxCommand" );

    _config( m_freqStream, "calibrate.freqStream" );
    _config( m_offloadDevice, "calibrate.offloadDevice" );
    _config( m_halfWidthHz, "calibrate.halfWidthHz" );
    _config( m_modeMax, "calibrate.modeMax" );
    _config( m_nAverage, "calibrate.nAverage" );
    _config( m_baselineEvery, "calibrate.baselineEvery" );
    _config( m_settleS, "calibrate.settleS" );
    _config( m_timeoutS, "calibrate.timeoutS" );
    _config( m_abortRatio, "calibrate.abortRatio" );
    _config( m_calibLeak, "calibrate.leak" );
    _config( m_startGridUs, "calibrate.startGridUs" );
    _config( m_durationGridUs, "calibrate.durationGridUs" );
    _config( m_gainGrid, "calibrate.gainGrid" );
    _config( m_csvDir, "calibrate.csvDir" );

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

    // A scan grid is min, max, step within the parameter's limits.
    auto gridOk = [&inRange]( const std::vector<double> &g, double lo, double hi )
    {
        return g.size() == 3 && inRange( g[0], lo, hi ) && inRange( g[1], g[0], hi ) && inRange( g[2], 1e-9, hi - lo );
    };

    if( m_syncopationTiming.empty() || m_channel.size() <= suffixLength ||
        m_channel.compare( m_channel.size() - suffixLength, suffixLength, "_aux" ) != 0 ||
        m_syncopationTiming == m_channel || !inRange( m_rampStartUs, 0, 1e6 ) || !inRange( m_rampDurationUs, 1, 1e6 ) ||
        !inRange( m_leak, 0, 1 ) || !inRange( m_timeOffsetUs, -1e6, 1e6 ) || !inRange( m_gain, 0, 2 ) ||
        !inRange( m_computeLeadUs, 0, 1e6 ) || !inRange( m_maxCommand, 1e-6, 1e3 ) )
    {
        return -1;
    }

    if( !inRange( m_halfWidthHz, 0, 1e6 ) || m_modeMax < 0 || m_nAverage < 2 || m_baselineEvery < 1 ||
        !inRange( m_settleS, 0, 1e3 ) || !inRange( m_timeoutS, 1, 1e4 ) || !inRange( m_abortRatio, 1, 1e6 ) ||
        !inRange( m_calibLeak, 1e-9, 1 ) ||
        !gridOk( m_startGridUs, 0, 1e6 ) || !gridOk( m_durationGridUs, 1, 1e6 ) || !gridOk( m_gainGrid, 0, 2 ) )
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
        { m_indiP_gain, "gain", 0, 2, m_gain, INDI_NEWCALLBACK( m_indiP_gain ) },
        { m_indiP_leak, "leak", 0, 1, m_leak, INDI_NEWCALLBACK( m_indiP_leak ) },
        { m_indiP_maxCommand, "maxCommand", 1e-6, 1e3, m_maxCommand, INDI_NEWCALLBACK( m_indiP_maxCommand ) },
        { m_indiP_rampStart, "rampStartUs", 0, 1e6, m_rampStartUs, INDI_NEWCALLBACK( m_indiP_rampStart ) },
        { m_indiP_rampDuration, "rampDurationUs", 1, 1e6, m_rampDurationUs, INDI_NEWCALLBACK( m_indiP_rampDuration ) },
        { m_indiP_timeOffset, "timeOffsetUs", -1e6, 1e6, m_timeOffsetUs, INDI_NEWCALLBACK( m_indiP_timeOffset ) },
        { m_indiP_halfWidth, "calibHalfWidthHz", 0, 1e6, m_halfWidthHz, INDI_NEWCALLBACK( m_indiP_halfWidth ) } };

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
    for( const char *el : { "events", "published", "failed", "overflows", "active", "clipped" } )
    {
        m_indiP_stats.add( pcf::IndiElement( el ) );
        m_indiP_stats[el] = 0;
    }
    if( registerIndiPropertyReadOnly( m_indiP_stats ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    createStandardIndiToggleSw( m_indiP_calibrate, "calibrate" );
    if( registerIndiPropertyNew( m_indiP_calibrate, INDI_NEWCALLBACK( m_indiP_calibrate ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    createROIndiNumber( m_indiP_calibration, "calibration", "Calibration" );
    for( const char *el :
         { "stage", "point", "fom", "fomOff", "ratio", "bestStartUs", "bestDurationUs", "bestGain", "bestRatio" } )
    {
        m_indiP_calibration.add( pcf::IndiElement( el ) );
        m_indiP_calibration[el] = 0;
    }
    if( registerIndiPropertyReadOnly( m_indiP_calibration ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    REG_INDI_SETPROP( m_indiP_offloadFps, m_offloadDevice, std::string( "fps" ) );

    REG_INDI_SETPROP( m_indiP_offloadModes, m_offloadDevice, std::string( "numModes" ) );

    SHMIMMONITOR_APP_STARTUP;

    SHMIMMONITORT_APP_STARTUP( psdShmimMonitorT );

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

    if( threadStart( m_calibThread,
                     m_calibThreadInit,
                     m_calibThreadID,
                     m_calibThreadProp,
                     0,
                     "",
                     "calibrate",
                     this,
                     calibThreadStart ) < 0 )
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

    SHMIMMONITORT_APP_LOGIC( psdShmimMonitorT );

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    SHMIMMONITORT_UPDATE_INDI( psdShmimMonitorT );

    std::vector<double> stats;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );

        double active = 0;
        for( const auto &r : m_ramps )
        {
            active += r.used;
        }

        stats = { double( m_events ), double( m_published ), double( m_failed ), double( m_overflows ), active,
                  double( m_clipped ) };
    }

    updateIfChanged( m_indiP_stats,
                     std::vector<std::string>{ "events", "published", "failed", "overflows", "active", "clipped" },
                     stats );

    const uint64_t clipped = m_clipped;

    if( clipped != m_clippedLogged ) // at most one warning per appLogic cycle
    {
        log<text_log>( "command clipped at +/- " + std::to_string( m_maxCommand ) + " in " +
                           std::to_string( clipped - m_clippedLogged ) + " slots",
                       logPrio::LOG_WARNING );
        m_clippedLogged = clipped;
    }

    double leak;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        leak = m_leak;
    }

    updateIfChanged( m_indiP_leak, std::vector<std::string>{ "current", "target" }, std::vector<double>{ leak, leak } );

    // Calibration changes the parameters from its own thread, so publish them here.
    const calibParamsT p = calibParams();

    updateIfChanged(
        m_indiP_rampStart, std::vector<std::string>{ "current", "target" }, std::vector<double>{ p[0], p[0] } );
    updateIfChanged(
        m_indiP_rampDuration, std::vector<std::string>{ "current", "target" }, std::vector<double>{ p[1], p[1] } );
    updateIfChanged( m_indiP_gain, std::vector<std::string>{ "current", "target" }, std::vector<double>{ p[2], p[2] } );

    std::vector<double> calib;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_calibMutex );

        calib = { double( m_calibStage ),
                  double( m_calibPoint ),
                  m_lastFom,
                  m_fomOff,
                  m_lastRatio,
                  m_calibBest[0],
                  m_calibBest[1],
                  m_calibBest[2],
                  m_calibBestRatio };
    }

    updateIfChanged(
        m_indiP_calibration,
        std::vector<std::string>{
            "stage", "point", "fom", "fomOff", "ratio", "bestStartUs", "bestDurationUs", "bestGain", "bestRatio" },
        calib );

    updateSwitchIfChanged( m_indiP_calibrate,
                           "toggle",
                           m_calibRunning ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           m_calibRunning ? INDI_BUSY : INDI_IDLE );

    return 0;
}

inline int counterOffload::appShutdown()
{
    m_calibAbort = true;
    m_calibCond.notify_all();

    for( std::thread *t : { &m_coThread, &m_calibThread } )
    {
        if( t->joinable() )
        {
            try
            {
                t->join(); // this will throw if it was already joined
            }
            catch( ... )
            {
            }
        }
    }

    SHMIMMONITOR_APP_SHUTDOWN;

    SHMIMMONITORT_APP_SHUTDOWN( psdShmimMonitorT );

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

    slot->delta = static_cast<realT>( m_gain ) *
                  Eigen::Map<const mx::improc::eigenImage<realT>>( delta, m_accum.rows(), m_accum.cols() );
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

    const realT lim     = m_maxCommand;
    bool        clipped = false;

    if( m_accumNonzero && m_accum.abs().maxCoeff() > lim ) // anti-windup: never hold more than can be commanded
    {
        m_accum = m_accum.cwiseMax( -lim ).cwiseMin( lim );
        clipped = true;
    }

    m_command = -( m_command + m_accum );

    if( m_command.abs().maxCoeff() > lim )
    {
        m_command = m_command.cwiseMax( -lim ).cwiseMin( lim );
        clipped   = true;
    }

    if( clipped )
    {
        ++m_clipped;
    }
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

inline int counterOffload::allocate( const psdShmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    if( psdShmimMonitorT::m_dataType != _DATATYPE_FLOAT )
    {
        return log<text_log, -1>( "PSD stream is not float", logPrio::LOG_ERROR );
    }

    std::lock_guard<std::mutex> lock( m_calibMutex );

    if( m_bandValid )
    {
        m_calibAbort = true;
        log<text_log>( "PSD stream changed during calibration", logPrio::LOG_WARNING );
    }

    m_psdNFreq  = psdShmimMonitorT::m_width;
    m_psdNModes = psdShmimMonitorT::m_height;

    return 0;
}

inline int counterOffload::processImage( void *curr_src, const psdShmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    const int64_t received = synco::monotonicNs();

    std::lock_guard<std::mutex> lock( m_calibMutex );

    if( !m_bandValid || received < m_psdAfterNs ) // only PSDs whose window is entirely after the change
    {
        return 0;
    }

    const double p = bandPower( static_cast<const realT *>( curr_src ),
                                m_psdNFreq,
                                m_psdNModes,
                                m_df,
                                m_bandCenterHz,
                                m_bandHalfWidthHz,
                                m_bandModes );

    if( p >= 0 )
    {
        m_psdPowers.push_back( p );
        m_calibCond.notify_all();
    }

    return 0;
}

inline double counterOffload::bandPower( const realT *psd,
                                         uint32_t     nFreq,
                                         uint32_t     nModes,
                                         double       df,
                                         double       centerHz,
                                         double       halfWidthHz,
                                         uint32_t     modeMax )
{
    const uint32_t modes = std::min( nModes, modeMax );

    if( !( df > 0 ) || modes == 0 || nFreq == 0 )
    {
        return -1;
    }

    const double lo = std::max( 0.0, std::ceil( ( centerHz - halfWidthHz ) / df - 1e-9 ) );
    const double hi = std::min( double( nFreq - 1 ), std::floor( ( centerHz + halfWidthHz ) / df + 1e-9 ) );

    if( hi < lo )
    {
        return -1;
    }

    double sum = 0;

    const uint32_t n0 = static_cast<uint32_t>( lo );
    const uint32_t n1 = static_cast<uint32_t>( hi );

    for( uint32_t m = 0; m < modes; ++m )
    {
        for( uint32_t n = n0; n <= n1; ++n )
        {
            sum += psd[n + m * nFreq];
        }
    }

    return sum * df;
}

inline bool counterOffload::coordinateScan( const std::array<std::vector<double>, 3> &grids,
                                            calibParamsT                             &best,
                                            double                                   &bestFom,
                                            const calibEvalT                         &eval )
{
    for( int pass = 0; pass < 2; ++pass )
    {
        for( int a = 0; a < 3; ++a )
        {
            const double lo   = grids[a][0];
            const double hi   = grids[a][1];
            const double step = ( pass == 0 ) ? grids[a][2] : 0.5 * grids[a][2];

            std::vector<double> xs;

            if( pass == 0 )
            {
                const int n = static_cast<int>( std::floor( ( hi - lo ) / step + 1e-9 ) );
                for( int i = 0; i <= n; ++i )
                {
                    xs.push_back( lo + i * step );
                }
            }
            else
            {
                for( int k = -2; k <= 2; ++k )
                {
                    const double x = best[a] + k * step;
                    if( x >= lo - 1e-9 && x <= hi + 1e-9 )
                    {
                        xs.push_back( x );
                    }
                }
            }

            std::vector<double> fs( xs.size() );

            for( size_t i = 0; i < xs.size(); ++i )
            {
                calibParamsT p = best;
                p[a]           = xs[i];

                if( !eval( p, a, pass, fs[i] ) )
                {
                    return false;
                }

                if( fs[i] < bestFom )
                {
                    bestFom = fs[i];
                    best    = p;
                }
            }

            if( pass == 0 || xs.size() < 3 )
            {
                continue;
            }

            // Parabolic refinement through the scan minimum and its neighbours.
            const size_t i = std::min_element( fs.begin(), fs.end() ) - fs.begin();

            if( i == 0 || i + 1 == xs.size() )
            {
                continue;
            }

            const double denom = fs[i - 1] - 2 * fs[i] + fs[i + 1];

            if( !( denom > 0 ) )
            {
                continue;
            }

            calibParamsT p = best;
            p[a]           = std::clamp( xs[i] + 0.5 * step * ( fs[i - 1] - fs[i + 1] ) / denom, lo, hi );

            double f;

            if( !eval( p, a, pass, f ) )
            {
                return false;
            }

            if( f < bestFom )
            {
                bestFom = f;
                best    = p;
            }
        }
    }

    return true;
}

inline counterOffload::calibParamsT counterOffload::calibParams()
{
    std::lock_guard<std::mutex> guard( m_stateMutex );

    return { m_rampStartUs, m_rampDurationUs, m_gain };
}

inline void counterOffload::setCalibParams( const calibParamsT &p )
{
    std::lock_guard<std::mutex> guard( m_stateMutex );

    m_rampStartUs    = p[0];
    m_rampDurationUs = p[1];
    m_gain           = p[2];
}

inline const char *counterOffload::calibStageName( int stage )
{
    static const char *names[] = {
        "off", "initial", "start", "duration", "gain", "refine start", "refine duration", "refine gain" };

    return ( stage >= 0 && stage < 8 ) ? names[stage] : "unknown";
}

inline void counterOffload::calibThreadStart( counterOffload *c )
{
    c->calibThreadExec();
}

inline void counterOffload::calibThreadExec()
{
    m_calibThreadID = syscall( SYS_gettid );

    // Wait for the thread starter to finish initializing this thread.
    while( m_calibThreadInit == true && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    while( !m_shutdown )
    {
        if( !m_calibStart.exchange( false ) )
        {
            mx::sys::milliSleep( 100 );
            continue;
        }

        runCalibration();
    }
}

inline void counterOffload::runCalibration()
{
    const calibParamsT original = calibParams();

    double originalLeak;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        originalLeak = m_leak;
        m_leak       = m_calibLeak; // keep the held contribution small so gain steps cannot saturate the tweeter
    }

    m_calibClipStart = m_clipped;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_calibPoint     = 0;
        m_calibStage     = 0;
        m_calibBest      = original;
        m_calibBestRatio = 0;
    }

    m_evalCount = 0;

    calibParamsT best      = original;
    double       bestRatio = 0;

    bool ok = setupBand() && openCsv() && measureBaseline( original ) && evaluate( original, 1, bestRatio );

    if( ok )
    {
        ok = coordinateScan( { m_startGridUs, m_durationGridUs, m_gainGrid },
                             best,
                             bestRatio,
                             [this]( const calibParamsT &p, int axis, int pass, double &ratio )
                             {
                                 return evaluate( p, 2 + axis + 3 * pass, ratio );
                             } );
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_bandValid = false;

        if( ok )
        {
            m_calibBest      = best;
            m_calibBestRatio = bestRatio;
        }
    }

    if( ok )
    {
        setCalibParams( best );

        log<text_log>( "calibration complete, applied rampStartUs=" + std::to_string( best[0] ) +
                           " rampDurationUs=" + std::to_string( best[1] ) + " gain=" + std::to_string( best[2] ) +
                           " ratio=" + std::to_string( bestRatio ),
                       logPrio::LOG_INFO );

        if( bestRatio >= 1 )
        {
            log<text_log>( "calibration found no improvement over the gain-0 baseline", logPrio::LOG_WARNING );
        }
    }
    else
    {
        setCalibParams( original );

        log<text_log>( "calibration aborted, restored rampStartUs=" + std::to_string( original[0] ) +
                           " rampDurationUs=" + std::to_string( original[1] ) +
                           " gain=" + std::to_string( original[2] ),
                       logPrio::LOG_WARNING );
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        m_leak = originalLeak;
    }

    if( m_csv.is_open() )
    {
        m_csv.close();
    }

    m_calibRunning = false;
}

inline bool counterOffload::setupBand()
{
    double   df = 0;
    uint64_t nf = 0;

    IMAGE freq{};

    if( ImageStreamIO_openIm( &freq, m_freqStream.c_str() ) != 0 )
    {
        log<text_log>( "calibration could not open " + m_freqStream, logPrio::LOG_ERROR );
        return false;
    }

    if( freq.md->datatype == _DATATYPE_FLOAT && freq.md->nelement > 1 )
    {
        nf = freq.md->nelement;
        df = freq.array.F[1] - freq.array.F[0];
    }

    ImageStreamIO_closeIm( &freq );

    double halfWidth;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        halfWidth = m_halfWidthHz;
    }

    std::lock_guard<std::mutex> lock( m_calibMutex );

    const uint32_t modes = ( m_modeMax > 0 ) ? m_modeMax : m_offloadModes;

    if( !( df > 0 ) || nf != m_psdNFreq || m_psdNModes == 0 )
    {
        log<text_log>( "calibration: PSD stream and " + m_freqStream + " do not match, or no PSDs yet",
                       logPrio::LOG_ERROR );
        return false;
    }

    if( !( m_offloadFps > 0 ) || modes == 0 || m_offloadFps - halfWidth > ( nf - 1 ) * df )
    {
        log<text_log>( "calibration: no offload rate or numModes from " + m_offloadDevice + ", or band above the PSDs",
                       logPrio::LOG_ERROR );
        return false;
    }

    m_df              = df;
    m_bandCenterHz    = m_offloadFps;
    m_bandHalfWidthHz = halfWidth;
    m_bandModes       = modes;
    m_psdAfterNs      = std::numeric_limits<int64_t>::max(); // nothing is collected until the first measure()
    m_bandValid       = true;

    return true;
}

inline bool counterOffload::openCsv()
{
    char      ts[32];
    time_t    now = time( nullptr );
    struct tm utc;

    gmtime_r( &now, &utc );
    strftime( ts, sizeof( ts ), "%Y%m%dT%H%M%SZ", &utc );

    std::error_code ec;
    std::filesystem::create_directories( m_csvDir, ec );

    const std::string path = m_csvDir + "/" + m_configName + "_calib_" + ts + ".csv";

    m_csv.open( path );

    if( !m_csv )
    {
        log<text_log>( "calibration could not open " + path, logPrio::LOG_ERROR );
        return false;
    }

    m_csv << "time,stage,rampStartUs,rampDurationUs,gain,offloadHz,numModes,halfWidthHz,nAverage,fom,fomStd,fomOff,"
             "ratio\n";

    log<text_log>( "calibration started at " + std::to_string( m_bandCenterHz ) + " Hz over " +
                       std::to_string( m_bandModes ) + " modes, writing " + path,
                   logPrio::LOG_INFO );

    return true;
}

inline bool counterOffload::measure( const calibParamsT &p, double &fom, double &fomStd )
{
    setCalibParams( p );

    std::unique_lock<std::mutex> lock( m_calibMutex );

    m_psdPowers.clear();
    m_psdAfterNs           = synco::monotonicNs() + static_cast<int64_t>( m_settleS * 1e9 );
    const int64_t deadline = m_psdAfterNs + static_cast<int64_t>( m_timeoutS * 1e9 );

    while( m_psdPowers.size() < static_cast<size_t>( m_nAverage ) )
    {
        std::string why;

        if( m_calibAbort || m_shutdown )
        {
            why = "abort requested";
        }
        else if( !m_enabled )
        {
            why = "app disabled";
        }
        else if( m_clipped != m_calibClipStart )
        {
            why = "counter command clipped at maxCommand";
        }
        else if( std::fabs( m_offloadFps - m_bandCenterHz ) > 0.5 ||
                 ( m_modeMax == 0 && m_offloadModes != m_bandModes ) )
        {
            why = "offload rate or numModes changed";
        }
        else if( synco::monotonicNs() > deadline )
        {
            why = "timed out waiting for PSDs";
        }

        if( !why.empty() )
        {
            log<text_log>( "calibration: " + why, logPrio::LOG_WARNING );
            return false;
        }

        m_calibCond.wait_for( lock, std::chrono::milliseconds( 100 ) );
    }

    double sum = 0, sum2 = 0;

    for( double v : m_psdPowers )
    {
        sum += v;
        sum2 += v * v;
    }

    const double n = m_psdPowers.size();

    fom    = sum / n;
    fomStd = std::sqrt( std::max( 0.0, ( sum2 - sum * sum / n ) / ( n - 1 ) ) );

    return true;
}

inline bool counterOffload::measureBaseline( const calibParamsT &p )
{
    calibParamsT off = p;
    off[2]           = 0;

    double fom, fomStd;

    if( !measure( off, fom, fomStd ) )
    {
        return false;
    }

    if( !( fom > 0 ) )
    {
        log<text_log>( "calibration: the gain-0 baseline has no power in the band", logPrio::LOG_WARNING );
        return false;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_fomOff = fom;
    }

    writeCsv( 0, off, fom, fomStd );

    return true;
}

inline bool counterOffload::evaluate( const calibParamsT &p, int stage, double &ratio )
{
    if( m_evalCount > 0 && m_evalCount % m_baselineEvery == 0 && !measureBaseline( p ) )
    {
        return false;
    }

    ++m_evalCount;

    double fom, fomStd;

    if( !measure( p, fom, fomStd ) )
    {
        return false;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        ratio        = fom / m_fomOff;
        m_calibStage = stage;
        ++m_calibPoint;
        m_lastFom   = fom;
        m_lastRatio = ratio;

        if( stage == 1 || ratio < m_calibBestRatio )
        {
            m_calibBest      = p;
            m_calibBestRatio = ratio;
        }
    }

    writeCsv( stage, p, fom, fomStd );

    if( ratio > m_abortRatio )
    {
        log<text_log>( "calibration: band power " + std::to_string( ratio ) + " times the baseline, aborting",
                       logPrio::LOG_WARNING );
        return false;
    }

    return true;
}

inline void counterOffload::writeCsv( int stage, const calibParamsT &p, double fom, double fomStd )
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    std::lock_guard<std::mutex> lock( m_calibMutex );

    m_csv << std::fixed << ts.tv_sec << "." << std::setw( 9 ) << std::setfill( '0' ) << ts.tv_nsec
          << std::setfill( ' ' ) << std::defaultfloat << "," << calibStageName( stage ) << "," << p[0] << "," << p[1]
          << "," << p[2] << "," << m_bandCenterHz << "," << m_bandModes << "," << m_bandHalfWidthHz << "," << m_nAverage
          << "," << fom << "," << fomStd << "," << m_fomOff << "," << fom / m_fomOff << "\n";

    m_csv.flush();
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
    return setNumber( m_indiP_gain, ipRecv, m_gain, 0, 2 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_leak )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_leak, ipRecv );
    return setNumber( m_indiP_leak, ipRecv, m_leak, 0, 1 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_maxCommand )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_maxCommand, ipRecv );
    return setNumber( m_indiP_maxCommand, ipRecv, m_maxCommand, 1e-6, 1e3 );
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

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_calibrate )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_calibrate, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
    {
        if( m_calibRunning )
        {
            m_calibAbort = true;
            log<text_log>( "calibration abort requested", logPrio::LOG_NOTICE );
        }

        return 0;
    }

    if( m_calibRunning )
    {
        return 0;
    }

    if( !m_enabled || state() != stateCodes::OPERATING )
    {
        return log<text_log, 0>( "calibration needs the app enabled and OPERATING", logPrio::LOG_WARNING );
    }

    m_calibAbort   = false;
    m_calibRunning = true;
    m_calibStart   = true;

    log<text_log>( "calibration requested", logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_halfWidth )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_halfWidth, ipRecv );
    return setNumber( m_indiP_halfWidth, ipRecv, m_halfWidthHz, 0, 1e6 );
}

INDI_SETCALLBACK_DEFN( counterOffload, m_indiP_offloadFps )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadFps, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_offloadFps = ipRecv["current"].get<float>();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( counterOffload, m_indiP_offloadModes )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadModes, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_offloadModes = std::max( 0, ipRecv["current"].get<int>() );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // counterOffload_hpp
