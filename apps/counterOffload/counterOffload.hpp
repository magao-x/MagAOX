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
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <type_traits>
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

/// Tag for the per-frame modal residual stream monitor used by calibrate mode.
/**
 * \ingroup counterOffload
 */
struct residualShmimT
{
    /// Config section of the residual stream monitor.
    static std::string configSection()
    {
        return "residual";
    };

    /// INDI prefix of the residual stream monitor.
    static std::string indiPrefix()
    {
        return "residual";
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
 * Calibrate mode drives repeatable offloads by stepping one woofer mode (wooferModes) and reads the per-frame modal
 * residual.  For each step it matches the onset to the response to the step alone (measured with the offloader paused),
 * subtracts that response, aligns on the first offload after the step, and averages over the steps.  The figure of merit is the energy left, summed over
 * the offloaded modes, as a ratio to a gain-0 baseline interpolated in time.  It estimates the noise from replicates,
 * sweeps rampStartUs, gain and rampDurationUs, then rampStartUs and gain again, choosing each from a quadratic fit and
 * moving only for an improvement beyond the noise, and applies the result if replicates confirm it.
 *
 * The latency INDI property breaks each offload's delay into the handoff from the woofer post (whose time t2wOffloader
 * stores in the input frame's atime), the wait for a slot, and the compute and write times of that slot.
 *
 * \ingroup counterOffload
 */
class counterOffload : public MagAOXApp<true>,
                       public dev::shmimMonitor<counterOffload>,
                       public dev::shmimMonitor<counterOffload, residualShmimT>,
                       public dev::dmSyncopationProducer<counterOffload>
{
    typedef float realT; ///< Floating point type of the tweeter commands.

    typedef dev::shmimMonitor<counterOffload> shmimMonitorT; ///< The input stream monitor base.

    typedef dev::shmimMonitor<counterOffload, residualShmimT> residualShmimMonitorT; ///< The residual monitor base.

    typedef dev::dmSyncopationProducer<counterOffload> dmSyncopationProducerT; ///< The syncopation producer base.

    friend class dev::shmimMonitor<counterOffload>;

    friend class dev::shmimMonitor<counterOffload, residualShmimT>;

    friend class dev::dmSyncopationProducer<counterOffload>;

    friend class counterOffload_test;

  public:
    /// One offload change and the time its ramp is referenced to.
    struct rampT
    {
        mx::improc::eigenImage<realT> delta; ///< The offload change as it appears on the tweeter.

        int64_t t0Ns{ 0 }; ///< CLOCK_MONOTONIC reference time of the woofer command, in ns.

        bool used{ false }; ///< Whether this pool entry holds an unfinished ramp.

        int64_t wakeNs{ 0 }; ///< CLOCK_MONOTONIC time the offload was received, in ns.

        int64_t handoffNs{ -1 }; ///< Woofer post to receipt, in ns; -1 if the offload carried no woofer stamp.

        bool latencyPending{ false }; ///< Whether this offload's latency is still to be recorded.
    };

    static constexpr size_t m_maxRamps{ 8 }; ///< Size of the preallocated ramp pool.

    /// The latency terms of one offload, measured at the first slot published after it arrived.
    struct latencySampleT
    {
        double handoffUs{ -1 }; ///< Woofer post to receipt; negative if the offload carried no woofer stamp, us.

        double slotWaitUs{ 0 }; ///< Receipt to the start of computing the slot that carried it, us.

        double computeUs{ 0 }; ///< Time to build that slot's command, us.

        double writeUs{ 0 }; ///< Time from the start of the copy to the end of the publication, us.
    };

    /// Statistics of the latency terms over a window of offloads.  A value with no data in the window is -1.
    struct latencySummaryT
    {
        double pipelineUs{ -1 }; ///< Mean of handoff + compute + write, over offloads with a woofer stamp, us.

        double pipelineRmsUs{ -1 }; ///< RMS deviation of handoff + compute + write about its mean, us.

        double handoffUs{ -1 }; ///< Mean handoff, over offloads with a woofer stamp, us.

        double slotWaitUs{ -1 }; ///< Mean slot wait, us.

        double computeUs{ -1 }; ///< Mean compute time, us.

        double writeUs{ -1 }; ///< Mean write time, us.

        double computeMaxUs{ -1 }; ///< Largest compute time, us.
    };

    static constexpr size_t m_latencyWindow{ 250 }; ///< Offloads in the latency statistics window.

    static constexpr int64_t m_maxHandoffNs{ 10000000 }; ///< Longer woofer-to-receipt times are rejected as stale, ns.

    /// Woofer post to receipt, from the offload frame's atime and writetime and the CLOCK_ISIO receipt time.
    /** t2wOffloader stores the woofer post time in atime.  A frame whose atime equals its writetime was posted without
     * a woofer stamp (for example by an older t2wOffloader).
     *
     * \returns the handoff in ns
     * \returns -1 if the frame has no woofer stamp, or the handoff is negative or longer than m_maxHandoffNs
     */
    static int64_t stampedHandoffNs( const timespec &atime /**< [in] the frame's atime, the woofer post */,
                                     const timespec &writetime /**< [in] the frame's writetime */,
                                     const timespec &receivedIsio /**< [in] CLOCK_ISIO receipt time */ );

    /// Summarize a window of latency samples.
    /**
     * \returns the means, the pipeline RMS and the largest compute time; -1 for values with no data
     */
    static latencySummaryT summarizeLatency( const latencySampleT *samples /**< [in] the samples */,
                                             size_t                n /**< [in] number of samples */ );

    static constexpr realT m_zeroTol{ 1e-7 }; ///< A leaking contribution below this magnitude is set to zero.

    /// Calibrated parameters, in order: rampStartUs, rampDurationUs, gain.
    typedef std::array<double, 3> calibParamsT;

    /// Calibration stages, as used by calibStageName(), the CSV and the calibration.stage INDI element.
    enum calibStage
    {
        stageOff = 0,     ///< A gain-0 baseline.
        stageInitial,     ///< A replicate of the starting parameters, for the noise estimate.
        stageStart,       ///< A point of the rampStartUs sweep.
        stageDuration,    ///< A point of the rampDurationUs sweep.
        stageGain,        ///< A point of the gain sweep.
        stageConfirm,     ///< A replicate of the chosen parameters.
        stageClipped,     ///< No longer produced: clipped points are kept.  Kept so later stage numbers don't change.
        stageFitStart,    ///< The rampStartUs decision.
        stageFitDuration, ///< The rampDurationUs decision.
        stageFitGain,     ///< The gain decision.
        stageStepOnly     ///< The response to the woofer step alone, with the offloader paused.
    };

    /// Least-squares quadratic r(x) = c0 + c1 u + c2 u^2 in the scaled coordinate u = (x - x0) / scale.
    struct quadFitT
    {
        bool ok{ false }; ///< Whether the fit was computed.

        double x0{ 0 }; ///< Center of the fitted x values.

        double scale{ 1 }; ///< Half range of the fitted x values.

        std::array<double, 3> c{ 0, 0, 0 }; ///< Coefficients in u.

        double resSd{ 0 }; ///< Residual standard deviation, with n - 3 degrees of freedom; 0 if n <= 3.

        /// Evaluate the fit.
        double operator()( double x /**< [in] the parameter value */ ) const
        {
            const double u = ( x - x0 ) / scale;
            return c[0] + c[1] * u + c[2] * u * u;
        }
    };

    /// The decision for one parameter at the end of a sweep.
    struct choiceT
    {
        double candidate{ 0 }; ///< Value with the lowest predicted ratio within the measured range.

        double rCurrent{ 0 }; ///< Predicted ratio at the current value; 0 if outside the measured range.

        double rCandidate{ 0 }; ///< Predicted ratio at the candidate.

        double threshold{ 0 }; ///< Improvement needed to move.

        bool outside{ false }; ///< Whether the current value is outside the measured range.

        bool move{ false }; ///< Whether to move to the candidate.
    };

    /// Pooled noise estimate of the ratio, from replicate scatter and fit residuals.
    struct noiseT
    {
        double sumSq{ 0 }; ///< Sum of squared deviations.

        int dof{ 0 }; ///< Degrees of freedom.

        /// Add a standard deviation with its degrees of freedom.
        void add( double sd /**< [in] standard deviation */, int d /**< [in] degrees of freedom */ )
        {
            if( d > 0 )
            {
                sumSq += sd * sd * d;
                dof += d;
            }
        }

        /// The pooled standard deviation, but at least floor; floor if nothing was added.
        double sigma( double floor /**< [in] smallest value returned */ ) const
        {
            return ( dof > 0 ) ? std::max( floor, std::sqrt( sumSq / dof ) ) : floor;
        }
    };

    /// One measurement, held until the baselines around it are measured.
    struct calibRowT
    {
        timespec wall{}; ///< CLOCK_REALTIME at the end of the measurement.

        int64_t tNs{ 0 }; ///< CLOCK_MONOTONIC at the end of the measurement, ns.

        int stage{ 0 }; ///< Stage index.

        calibParamsT p{ 0, 0, 0 }; ///< Parameters.

        double fom{ 0 }; ///< Step figure of merit, see stepFom().

        double fomStd{ 0 }; ///< Half the difference of the figures of merit of the two halves of the steps.

        double fomOff{ 0 }; ///< Latest baseline when measured.

        double fomTiming{ 0 }; ///< Part of fom in the first two frames after each offload.

        double fomGain{ 0 }; ///< Part of fom in the other frames.

        double onsetDelay{ 0 }; ///< Mean step onset after its command, over the steps used, frames.

        double onsetDelaySd{ 0 }; ///< Standard deviation of the onset delay, frames.

        double phase{ 0 }; ///< Mean first offload after the onset, over the steps used, frames.

        double phaseSd{ 0 }; ///< Standard deviation of the phase, frames.

        uint64_t clippedSlots{ 0 }; ///< Slots that clipped from the first step to the end of the measurement.

        uint64_t publishedSlots{ 0 }; ///< Slots published from the first step to the end of the measurement.
    };

    /// Residual frames and offload times collected during one series of woofer steps.
    struct stepDataT
    {
        uint32_t nModes{ 0 }; ///< Residual values stored per frame: the first nModes of each frame.

        std::vector<float> frames; ///< Residual frames, nModes values each, in arrival order.

        std::vector<int64_t> frameNs; ///< CLOCK_ISIO atime of each frame, ns.

        std::vector<int64_t> offloadNs; ///< CLOCK_ISIO woofer stamps of the offloads received, ns, ascending.

        std::vector<size_t> cmdIndex; ///< Frames collected when each step was commanded.

        std::vector<int> signs; ///< Sign of each step, +1 or -1.
    };

    /// One woofer step located in the residual.
    struct stepT
    {
        bool found{ false }; ///< Whether the onset was detected.

        size_t cmd{ 0 }; ///< Frame index of the step command: frames collected when it was sent.

        size_t onset{ 0 }; ///< Frame index of the onset.

        size_t end{ 0 }; ///< One past the last frame before the next step command.

        long offload{ -1 }; ///< Frame index of the first offload at or after the onset; -1 if none.

        std::vector<size_t> offloads; ///< Frame indices of the offloads from the command to end, ascending.

        std::vector<double> preMean; ///< Mean of each mode over the frames before the command.

        double noiseVar{ 0 }; ///< Per-frame variance summed over the modes, from the frames before the command.
    };

    /// The step figure of merit of one measurement.
    struct stepFomT
    {
        double fom{ 0 }; ///< Energy of the averaged error, summed over lags and modes, less the noise bias.

        double fomStd{ 0 }; ///< Half the difference of the figures of merit of the two halves of the steps.

        double fomTiming{ 0 }; ///< Part of fom in the first two frames after each offload.

        double fomGain{ 0 }; ///< Part of fom in the other frames.

        size_t used{ 0 }; ///< Steps used.

        double onsetDelay{ 0 }; ///< Mean onset after the command, over the steps used, frames.

        double onsetDelaySd{ 0 }; ///< Standard deviation of the onset delay, frames.

        double phase{ 0 }; ///< Mean first offload after the onset, over the steps used, frames.

        double phaseSd{ 0 }; ///< Standard deviation of the phase, frames.
    };

    static constexpr size_t m_preFrames{ 100 }; ///< Frames before a step command used for its mean and noise.

    static constexpr size_t m_minPreFrames{ 10 }; ///< Fewest frames before a step command for it to be used.

    static constexpr size_t m_guardFrames{ 8 }; ///< Frames before the next step left out of the step-only response.

    static constexpr size_t m_maxPhaseFrames{ 16 }; ///< Largest onset-to-first-offload offset allowed, frames.

    static constexpr size_t m_matchFrames{ 8 }; ///< Most frames, beyond the shift range, matched by alignSteps().

    static constexpr size_t m_minMatchFrames{ 3 }; ///< Fewest frames matched by alignSteps().

    static constexpr size_t m_maxShiftFrames{ 4 }; ///< Largest onset change made by alignSteps(), frames.

    static constexpr double m_maxRunMin{ 20 }; ///< A run estimated to take longer than this logs a warning, min.

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_channel{ "dm01disp08_aux" }; ///< The dmSyncopation auxiliary source this app writes.

    double m_rampStartUs{ 0 }; ///< Ramp start after the woofer command, in us.

    double m_rampDurationUs{ 330 }; ///< Ramp duration, in us.

    double m_leak{ 0 }; ///< Fraction of the completed contribution removed after each published slot, 0 to 1.

    double m_timeOffsetUs{ 0 }; ///< Offset added to the input receipt time to give the ramp reference, in us.

    double m_gain{ 0.5 }; ///< Scale applied to each new offload when it arrives, 0 to 2.

    double m_maxCommand{ 1.0 }; ///< Largest magnitude of any pixel of the command and of the accumulator.

    double m_computeLeadUs{ 20 }; ///< Time reserved before the publish lead to compute the command, in us.

    int m_threadPrio{ 20 }; ///< Real-time priority of the producer thread.

    std::string m_threadCpuset; ///< The cpuset for the producer thread.

    ///@}

    /** \name Calibrate Parameters - Data
     * The devices, directory and numeric settings are optional config keys in the calibrate section, see calibKeys();
     * the residual stream is set in the residual section.  The scan grids and the step mode and amplitude are set over
     * INDI only, and are reset to these defaults on restart.  None can be changed while a calibration runs.
     *@{
     */

    std::string m_offloadDevice{ "t2wOffloader" }; ///< INDI device providing numModes and the offload toggle.

    std::string m_stepDevice{ "wooferModes" }; ///< dmMode INDI device whose mode is stepped.

    std::string m_csvDir{ "/home/xsup/counterOffload" }; ///< Directory for the per-run calibration CSV files.

    int m_modeMax{ 0 }; ///< Number of residual modes summed.  0 uses the offloader's numModes.

    int m_baselineEvery{ 4 }; ///< Measure a gain-0 baseline after every this many points of a sweep.

    double m_settleS{ 0.2 }; ///< Wait after a change for the held contribution to leak away before the steps, s.

    double m_timeoutS{ 5 }; ///< Longest wait for residual frames or a change of the offload toggle, s.

    double m_stepPeriodS{ 0.5 }; ///< Time between woofer steps, s.

    int m_stepCount{ 10 }; ///< Woofer steps per measurement, alternating in sign.

    double m_onsetSigma{ 5 }; ///< Onset threshold, in standard deviations of the pre-step residual energy.

    int m_stepMode{ 5 }; ///< Index of the stepDevice mode stepped (0005, astig00).  INDI only.

    double m_stepAmp{ 0.7 }; ///< Step amplitude about the starting amplitude; steps alternate +a and -a.  INDI only.

    double m_abortRatio{ 3 }; ///< Abort when an evaluation exceeds this multiple of the baseline.

    double m_calibLeak{ 0.0025 }; ///< Leak used during a calibration run, restored afterwards; must be > 0.

    std::vector<double> m_startGridUs{ 0, 150, 25 }; ///< rampStartUs scan: min, max, step, in us.

    std::vector<double> m_durationGridUs{ 200, 600, 100 }; ///< rampDurationUs scan: min, max, step, in us.

    std::vector<double> m_gainGrid{ 0.3, 0.7, 0.1 }; ///< gain scan: min, max, step.  Above 0.7 the loop went unstable.

    int m_replicates{ 3 }; ///< Measurements of the starting parameters for the first noise estimate.

    int m_confirmPairs{ 3 }; ///< Measurements of the chosen parameters, each after a baseline, to confirm them.

    double m_moveSigma{ 1.5 }; ///< A parameter moves only if the fit improves by this many standard errors.

    double m_sigmaFloor{ 0.002 }; ///< Smallest noise estimate of the ratio used.

    int m_rounds{ 2 }; ///< Rounds of sweeps; rounds after the first sweep only rampStartUs and gain.

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

    std::atomic<uint64_t> m_published{ 0 }; ///< Commands successfully published.  Atomic so calibration can read it.

    uint64_t m_failed{ 0 }; ///< Commands not published by syncopationPublish.

    uint64_t m_overflows{ 0 }; ///< Offloads folded immediately because the ramp pool was full.

    std::atomic<uint64_t> m_clipped{ 0 }; ///< Slots in which the command or the accumulator was clipped.

    uint64_t m_clippedLogged{ 0 }; ///< m_clipped at the last clipping warning.  appLogic only.

    uint64_t m_overflowsLogged{ 0 }; ///< m_overflows at the last ramp-pool overflow warning.  appLogic only.

    std::array<latencySampleT, m_latencyWindow> m_latency{}; ///< Ring of the latest latency samples.  m_stateMutex.

    size_t m_latencyCount{ 0 }; ///< Valid samples in m_latency, up to m_latencyWindow.  m_stateMutex.

    size_t m_latencyNext{ 0 }; ///< Index in m_latency for the next sample.  m_stateMutex.

    uint64_t m_missed{ 0 }; ///< Offloads that waited at least a whole slot before being carried.  m_stateMutex.

    std::atomic<int64_t> m_slotNs{ 0 }; ///< Slot period from successive slot targets, ns; 0 until measured.

    synco::Request m_lastRequest{}; ///< The previous slot request, for m_slotNs.  Producer thread only.

    bool m_leadLogged{ false }; ///< Whether the slot lead has been logged.  Producer thread only.

    ///@}

    /** \name Calibrate State - Data
     *@{
     */

    std::mutex m_calibMutex; ///< Protects the collected frames, the remote device values and calibration readbacks.

    std::condition_variable m_calibCond; ///< Wakes calibration waits early, on abort.

    std::atomic<bool> m_calibStart{ false }; ///< Request to start a run, consumed by the calibration thread.

    std::atomic<bool> m_calibRunning{ false }; ///< Whether a calibration run is requested or in progress.

    std::atomic<bool> m_calibAbort{ false }; ///< Request to abort the run and restore the original parameters.

    uint32_t m_offloadModes{ 0 }; ///< Number of offloaded modes reported by the offloader.

    std::string m_offloadState; ///< FSM state reported by the offloader.

    std::atomic<bool> m_offloading{ false }; ///< Whether the offloader reports its offload toggle on.

    std::string m_stepState; ///< FSM state reported by the step device.

    std::map<std::string, double> m_stepAmps; ///< Current amplitudes reported by the step device, by element name.

    uint32_t m_residualValues{ 0 }; ///< Values in each frame of the residual stream; 0 until connected.

    std::atomic<bool> m_collecting{ false }; ///< Whether residual frames and offload times are being collected.

    stepDataT m_stepData; ///< Frames collected for the current series of steps.  m_calibMutex.

    std::mutex m_stampMutex; ///< Protects m_offloadStamps, so the offload path never waits on frame collection.

    std::vector<int64_t> m_offloadStamps; ///< Woofer stamps of offloads received while collecting, ns.

    std::atomic<int64_t> m_lastFrameNs{ 0 }; ///< CLOCK_MONOTONIC arrival of the latest residual frame, ns.

    uint32_t m_stepModes{ 0 }; ///< Residual modes summed in the current run.  Calibration thread only.

    std::string m_stepElement; ///< Step device element stepped in the current run.  Calibration thread only.

    double m_stepBase{ 0 }; ///< Starting amplitude of the stepped mode, restored afterwards.  Calibration thread only.

    int m_stepPos{ -1 }; ///< Current step position, -1 or +1 about m_stepBase.  Calibration thread only.

    std::vector<double> m_stepOnly; ///< Step-only response by lag from the onset, nModes per lag.  Calibration thread.

    size_t m_stepOnlyCount{ 0 }; ///< Steps averaged into m_stepOnly.  Calibration thread only.

    size_t m_lagFrames{ 0 }; ///< Lags from the first offload in the figure of merit.  Calibration thread only.

    double m_fomOff{ 0 }; ///< Latest gain-0 baseline figure of merit.

    int m_calibStage{ 0 }; ///< Stage of the latest measurement or decision, an index into calibStageName().

    int m_calibPoint{ 0 }; ///< Points measured in the current run, not counting baselines.

    double m_lastRatio{ 0 }; ///< Ratio of the latest point to the latest baseline.

    calibParamsT m_calibBest{ 0, 0, 0 }; ///< Parameters chosen so far in the current or last run.

    double m_calibBestRatio{ 0 }; ///< Ratio predicted for m_calibBest.

    double m_confirmRatio{ 0 }; ///< Mean confirmed ratio of the chosen parameters.

    std::ofstream m_csv; ///< The current run's CSV file.  Calibration thread only.

    std::vector<int64_t> m_offTimes; ///< CLOCK_MONOTONIC times of the run's baselines, ns.  Calibration thread only.

    std::vector<double> m_offFoms; ///< The run's baselines, matching m_offTimes.  Calibration thread only.

    std::vector<calibRowT> m_rows; ///< Measurements not yet written to the CSV.  Calibration thread only.

    noiseT m_noise; ///< The run's noise estimate.  Calibration thread only.

    bool m_lastWasOff{ false }; ///< Whether the latest measurement was a baseline.  Calibration thread only.

    int m_measuredRows{ 0 }; ///< Measurements, baselines included, in this run.  Calibration thread only.

    int m_clippedRows{ 0 }; ///< Measurements in this run during which the command clipped.  Calibration thread only.

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
    /**
     * \returns the pool entry holding the new ramp
     * \returns nullptr if the buffers are not set up
     */
    rampT *addRamp( const realT *delta /**< [in] offload change, sized like m_accum */,
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

    /// Record the latency of every offload received before this slot's compute start and not yet recorded.
    /** Called after a successful publication.  An offload that waited at least one slot period is counted in
     * m_missed.  The caller must hold m_stateMutex.
     */
    void recordLatency( int64_t computeStartNs /**< [in] CLOCK_MONOTONIC start of building the command, ns */,
                        int64_t computeEndNs /**< [in] CLOCK_MONOTONIC end of building the command, ns */,
                        int64_t writeStartNs /**< [in] CLOCK_MONOTONIC start of the copy into the source, ns */,
                        int64_t writeEndNs /**< [in] CLOCK_MONOTONIC end of the publication, ns */ );

    /// Update m_slotNs from a new slot request, when it directly follows the previous one.  Producer thread only.
    void updateSlotPeriod( const synco::Request &request /**< [in] the new slot request */ );

    /// Enable or disable accepting offloads; disabling also requests a zero of the contribution.
    /** \returns true if the enabled state changed
     */
    bool setEnabled( bool on /**< [in] true to accept new offloads */ );

    /// Record the residual stream geometry.  Aborts a running calibration, since its frames no longer apply.
    /**
     * \returns 0 on success
     * \returns -1 if the stream is not float
     */
    int allocate( const residualShmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Store a residual frame and its atime while a calibration is collecting.
    /**
     * \returns 0 always
     */
    int processImage( void                 *curr_src, /**< [in] pointer to the start of the current frame.*/
                      const residualShmimT &dummy     /**< [in] tag to differentiate shmimMonitor parents.*/
    );

    /// A timespec in ns.
    static int64_t toNs( const timespec &ts /**< [in] the time */ );

    /// Locate each step of a series in the residual: its onset, its offloads, and the pre-step mean and noise.
    /** The pre-step frames are the up to m_preFrames frames before the step command.  The onset is the first frame at
     * or after the command whose squared deviation from the pre-step mean, summed over the modes, exceeds the pre-step
     * mean of that quantity by onsetSigma of its standard deviations.  Each offload is placed at the first frame whose
     * atime is at or after its woofer stamp.
     *
     * \returns the number of steps whose onset was found
     */
    static size_t findSteps( const stepDataT    &d /**< [in] the collected series */,
                             double              onsetSigma /**< [in] onset threshold, standard deviations */,
                             std::vector<stepT> &steps /**< [out] one entry per step command */ );

    /// Move each step's onset to where its response best matches the step-only response.
    /** The steps of a point have the previous step's offload spikes in their pre-step frames, which can shift the
     * threshold onset; the step-only response was measured with quiet pre-step frames.  For each step found, every
     * candidate onset o from max(command, onset - maxShift) to onset + maxShift is scored over one fixed window of
     * frames t: from the lowest candidate up to, but not including, the first offload at or after it, at least
     * m_minMatchFrames and at most matchFrames + 2 maxShift frames.  The score is the squared difference between the
     * step's sign-corrected residual, less its pre-step mean, and the step-only response at t - o (0 before the onset).
     * Ending the window at the first offload keeps the counter-offload error, which the figure of merit measures, out
     * of the alignment.  The best candidate becomes the onset (ties keep the detected one), and the first offload is
     * taken at or after it.
     *
     * \returns the number of steps whose onset moved
     */
    static size_t alignSteps( const stepDataT           &d /**< [in] the collected series */,
                              std::vector<stepT>        &steps /**< [in,out] the steps, from findSteps() */,
                              const std::vector<double> &stepOnly /**< [in] the step-only response */,
                              size_t                     matchFrames /**< [in] most frames matched, beyond the shift range */,
                              size_t                     maxShift /**< [in] largest onset change, frames */ );

    /// Mean and standard deviation of the onset delay (onset - command) of the steps found, in frames.
    static void onsetDelayStats( const std::vector<stepT> &steps /**< [in] the steps */,
                                 double                   &mean /**< [out] the mean; 0 if none found */,
                                 double                   &sd /**< [out] the standard deviation; 0 if fewer than 2 */ );

    /// Average the onset-aligned, sign-corrected response over the steps found, for lags 0 to len - 1.
    /**
     * \returns the number of steps averaged; 0 if no step has len frames after its onset
     */
    static size_t stepOnlyResponse( const stepDataT          &d /**< [in] the collected series */,
                                    const std::vector<stepT> &steps /**< [in] the steps, from findSteps() */,
                                    size_t                    len /**< [in] lags from the onset */,
                                    std::vector<double>      &response /**< [out] len x nModes, mode fastest */ );

    /// The step figure of merit of a series.
    /** For each step with an offload at most m_maxPhaseFrames after its onset, the error at lag l from that offload is
     * its sign times the residual less the pre-step mean, less the step-only response at the same lag from the onset.
     * The errors are averaged over the steps.  The figure of merit is the sum over lags and modes of the squared
     * average, less the noise bias (pre-step noise times 1/used + 1/stepOnlyCount per lag).  A lag is a timing lag if
     * an offload of more than half the steps falls on it or on the lag before.
     *
     * \returns the figure of merit; used is 0 if no step could be used
     */
    static stepFomT stepFom( const stepDataT           &d /**< [in] the collected series */,
                             const std::vector<stepT>  &steps /**< [in] the steps, from findSteps() */,
                             const std::vector<double> &stepOnly /**< [in] the step-only response, see stepOnlyResponse() */,
                             size_t                     stepOnlyCount /**< [in] steps averaged into stepOnly */,
                             size_t                     lags /**< [in] lags from the first offload */ );

    /// Baseline at a time, linearly interpolated between baselines and constant beyond the first and last.
    /**
     * \returns the interpolated baseline
     * \returns 0 if there are no baselines
     */
    static double interpBaseline( int64_t                     tNs /**< [in] the time, ns */,
                                  const std::vector<int64_t> &offTimes /**< [in] baseline times, ascending, ns */,
                                  const std::vector<double>  &offFoms /**< [in] baselines, matching offTimes */ );

    /// Order in which to measure the n points of a sweep: the indices in bit-reversed (van der Corput) order.
    /** For n = 7 the order is 0, 4, 2, 6, 1, 5, 3, so that neighbouring values are measured far apart in time and slow
     * drift does not look like a trend.
     *
     * \returns a permutation of 0 to n - 1
     */
    static std::vector<size_t> sweepOrder( size_t n /**< [in] number of points */ );

    /// Least-squares quadratic fit.
    /**
     * \returns the fit, with ok false for fewer than 3 points, a single x value, or a singular system
     */
    static quadFitT fitQuadratic( const std::vector<double> &xs /**< [in] parameter values */,
                                  const std::vector<double> &rs /**< [in] ratios */ );

    /// Choose a parameter value from a sweep.
    /** With at least 4 points, a quadratic is fitted, and the candidate is its minimum within the measured range.  With
     * fewer, or if the fit fails, the candidate is the minimum of a 3-point running mean.  The parameter moves to the
     * candidate only if the predicted improvement over the current value exceeds moveSigma * sigma * sqrt(6 / n), or if
     * the current value is outside the measured range.
     *
     * \returns the decision; with no points, the candidate is the current value and move is false
     */
    static choiceT chooseValue( const std::vector<double> &xs /**< [in] parameter values */,
                                const std::vector<double> &rs /**< [in] ratios */,
                                double                     current /**< [in] the current value */,
                                double                     sigma /**< [in] noise of one ratio */,
                                double                     moveSigma /**< [in] standard errors needed to move */ );

    /// Number of measurements a run takes with the current settings, including baselines.
    int estimatePoints();

    /// Get the current calibrated parameters.
    calibParamsT calibParams();

    /// Set the calibrated parameters.
    void setCalibParams( const calibParamsT &p /**< [in] rampStartUs, rampDurationUs, gain */ );

    /// Name of a calibration stage, for logs and the CSV.
    static const char *calibStageName( int stage /**< [in] the stage index */ );

    /// Names of the latency INDI elements, in the order appLogic publishes them.
    static const std::vector<std::string> &latencyElements();

    /// Fraction of published slots that clipped, 0 if none were published.
    static double clipFraction( uint64_t clipped /**< [in] slots that clipped */,
                                uint64_t published /**< [in] slots published */ );

    /// Whether a scan grid is min, max, step with min <= max, both within limits, and 0 < step <= hi - lo.
    static bool gridValid( const std::vector<double> &grid /**< [in] the grid */,
                           double                     lo /**< [in] smallest allowed value */,
                           double                     hi /**< [in] largest allowed value */ );

    /// Set a calibration setting, unless a calibration is running.
    /**
     * \returns 0 on success
     * \returns -1 if a calibration is running, or the value is nonfinite, out of range, or not integral for an
     *          integer setting
     */
    template <typename T>
    int applyCalibSetting( T     &setting /**< [out] the setting */,
                           double value /**< [in] the new value */,
                           double lo /**< [in] smallest allowed value */,
                           double hi /**< [in] largest allowed value */ );

    /// Set a calibration scan grid, unless a calibration is running.
    /**
     * \returns 0 on success
     * \returns -1 if a calibration is running or the grid is not valid, see gridValid()
     */
    int applyCalibGrid( std::vector<double>       &grid /**< [out] the grid setting */,
                        const std::vector<double> &value /**< [in] the new min, max, step */,
                        double                     lo /**< [in] smallest allowed value */,
                        double                     hi /**< [in] largest allowed value */ );

    /// Set the step mode and amplitude, unless a calibration is running.
    /**
     * \returns 0 on success
     * \returns -1 if a calibration is running, the mode is not an integer >= 0, or the amplitude is not finite and > 0
     */
    int applyCalibStep( double mode /**< [in] the step device mode index */,
                        double amp /**< [in] the step amplitude */ );

    /// The step device element name of a mode index, as dmMode names them (e.g. 0005).
    static std::string stepElementName( int mode /**< [in] the mode index */ );

    /// A numeric calibration setting, configured with an optional key in the calibrate section.
    struct calibKeyT
    {
        const char *key; ///< Config keyword in the calibrate section.

        double *real; ///< The setting if it is real, otherwise nullptr.

        int *integer; ///< The setting if it is an integer, otherwise nullptr.

        double lo; ///< Smallest allowed value.

        double hi; ///< Largest allowed value.

        const char *help; ///< Config help text, including the default.
    };

    /// The numeric calibration settings, with their limits and help text.
    std::vector<calibKeyT> calibKeys();

    /// Set a numeric calibration setting, see applyCalibSetting().
    /**
     * \returns 0 on success
     * \returns -1 if the value is rejected by applyCalibSetting()
     */
    int applyCalibKey( const calibKeyT &k /**< [in] the setting */, double value /**< [in] the new value */ );

    /// The current value of a numeric calibration setting.
    static double calibKeyValue( const calibKeyT &k /**< [in] the setting */ );

    /// All calibration settings and scan grids on one line, for the log.
    std::string calibSettingsString();

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

    /// Run one calibration: step-only response, noise estimate, sweeps, and confirmation.
    /** The chosen parameters are applied if their confirmed ratio is below 1, otherwise the originals are restored.  On
     * abort, the parameters chosen so far are applied if their predicted ratio is below 1.  The stepped mode's
     * amplitude and the offload toggle are always restored.
     */
    void runCalibration();

    /// Check that a run can start, and set the run's modes, step element and starting amplitude.
    /** A run needs the offloader OPERATING with its offload toggle on, the step device READY and reporting the step
     * element, and residual frames with at least as many values as modes summed.
     *
     * \returns true if the run can start; otherwise the reason is logged
     */
    bool startChecks();

    /// Open the run's CSV file in m_csvDir and write the header.
    /**
     * \returns true on success
     */
    bool openCsv();

    /// Wait, checking for an abort, a disable, a change of numModes, and residual frames stopping.
    /**
     * \returns true if the full time passed
     * \returns false otherwise, with the reason logged
     */
    bool calibWait( double seconds /**< [in] time to wait, s */,
                    bool   needOffload /**< [in] whether the offload toggle must stay on */ );

    /// Send the stepped mode's amplitude to the step device.
    /**
     * \returns true on success
     */
    bool sendStepAmp( double amp /**< [in] the amplitude */ );

    /// Turn the offloader's offload toggle on or off, and wait for it to report the change.
    /**
     * \returns true if the offloader reported the change within m_timeoutS
     */
    bool setOffloading( bool on /**< [in] the new state */ );

    /// Wait m_settleS, then make m_stepCount woofer steps m_stepPeriodS apart, collecting the residual and offloads.
    /** Clipping is counted from the first step to the end, after the previous point's held contribution leaked away.
     *
     * \returns true on success
     * \returns false on abort, disable, residual timeout, a change of numModes, or (if needOffload) offload turned off
     */
    bool collectSteps( stepDataT &d /**< [out] the collected series */,
                       bool       needOffload /**< [in] whether the offload toggle must stay on */,
                       uint64_t  &clipped /**< [out] slots that clipped from the first step */,
                       uint64_t  &published /**< [out] slots published from the first step */ );

    /// Measure the response to the woofer step alone, with the offloader's offload toggle off, and restore it.
    /** Sets m_stepOnly, m_stepOnlyCount and m_lagFrames, logs the change from the previous measurement, and writes a
     * CSV row whose fom is the response energy.
     *
     * \returns true on success
     */
    bool measureStepOnly( bool &offloadOff /**< [out] true while this function has the offload toggle off */ );

    /// Apply parameters, make the woofer steps, and compute the step figure of merit.
    /**
     * \returns true on success
     * \returns false if the steps could not be made, or fewer than max(2, m_stepCount / 2) were usable
     */
    bool measure( const calibParamsT &p /**< [in] parameters to apply */,
                  stepFomT           &f /**< [out] the figure of merit */,
                  uint64_t           &clipped /**< [out] slots that clipped from the first step */,
                  uint64_t           &published /**< [out] slots published from the first step */ );

    /// Measure the gain-0 baseline at the given ramp, and record it.
    /**
     * \returns true on success
     */
    bool measureBaseline( const calibParamsT &p /**< [in] parameters; the gain is replaced by 0 */ );

    /// Measure a baseline unless the latest measurement was one.
    /**
     * \returns true on success
     */
    bool ensureBaseline( const calibParamsT &p /**< [in] parameters; the gain is replaced by 0 */ );

    /// Measure one point and hold its row, with its clipping recorded.  A point that clipped is used like any other.
    /**
     * \returns true on success
     * \returns false on failure, or if the ratio to the latest baseline exceeds m_abortRatio
     */
    bool measurePoint( const calibParamsT &p /**< [in] parameters */,
                      int                 stage /**< [in] stage index */,
                      double             &fom /**< [out] the step figure of merit */ );

    /// Measure parameters n times, each between baselines, and return their interpolated ratios.
    /**
     * \returns true on success
     */
    bool replicate( const calibParamsT  &p /**< [in] parameters */,
                    int                  stage /**< [in] stage index */,
                    int                  n /**< [in] number of measurements */,
                    std::vector<double> &ratios /**< [out] ratios to the interpolated baseline */ );

    /// Sweep one parameter, then choose its value, see chooseValue().
    /** The sweep is bracketed by baselines, with one after every m_baselineEvery points.
     *
     * \returns true on success
     */
    bool sweep( int           axis /**< [in] 0 rampStartUs, 1 rampDurationUs, 2 gain */,
                calibParamsT &cur /**< [in,out] the chosen parameters */,
                double       &curRatio /**< [in,out] the ratio predicted for cur */ );

    /// Hold a measurement row.
    void addRow( int                 stage /**< [in] stage index */,
                 const calibParamsT &p /**< [in] parameters */,
                 const stepFomT     &f /**< [in] the figure of merit */,
                 uint64_t            clipped /**< [in] slots that clipped from the first step */,
                 uint64_t            published /**< [in] slots published from the first step */ );

    /// Write the held rows, with ratios to the latest and to the interpolated baseline.
    void flushRows();

    /// Write the time, stage, parameters and step columns of a CSV row.  The caller must hold m_calibMutex.
    void writeCsvPrefix( const timespec     &wall /**< [in] CLOCK_REALTIME of the row */,
                         int                 stage /**< [in] stage index */,
                         const calibParamsT &p /**< [in] parameters */ );

    /// Write a decision row.
    void writeFitRow( int                 stage /**< [in] stage index, one of the fit stages */,
                      const calibParamsT &p /**< [in] parameters, with the swept one at the candidate */,
                      const choiceT      &ch /**< [in] the decision */,
                      double              sigma /**< [in] the noise estimate used */ );

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

    pcf::IndiProperty m_indiP_stats; ///< Read-only counters.

    pcf::IndiProperty m_indiP_latency; ///< Read-only offload latency statistics.

    pcf::IndiProperty m_indiP_calibrate; ///< Toggle switch: On starts a calibration run, Off aborts it.

    pcf::IndiProperty m_indiP_startGrid; ///< Property to set the rampStartUs scan grid: min, max, step.

    pcf::IndiProperty m_indiP_durationGrid; ///< Property to set the rampDurationUs scan grid: min, max, step.

    pcf::IndiProperty m_indiP_gainGrid; ///< Property to set the gain scan grid: min, max, step.

    pcf::IndiProperty m_indiP_calibStep; ///< Property to set the step mode and amplitude.

    pcf::IndiProperty m_indiP_calibration; ///< Read-only calibration progress and best parameters.

    pcf::IndiProperty m_indiP_offloadModes; ///< The offloader's numModes (SET).

    pcf::IndiProperty m_indiP_offloadFsm; ///< The offloader's FSM state (SET).

    pcf::IndiProperty m_indiP_offloadToggle; ///< The offloader's offload toggle (SET).

    pcf::IndiProperty m_indiP_stepFsm; ///< The step device's FSM state (SET).

    pcf::IndiProperty m_indiP_stepAmps; ///< The step device's current amplitudes (SET).

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

    /// Create and register a calibration grid property with min, max and step elements.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int registerGrid( pcf::IndiProperty         &prop /**< [out] the property */,
                      const std::string         &name /**< [in] the property name */,
                      const std::vector<double> &grid /**< [in] the initial grid */,
                      int ( *callback )( void *, const pcf::IndiProperty & ) /**< [in] the new-property callback */ );

    /// Update a calibration grid from an INDI new-property request, see applyCalibGrid().
    /** Elements missing from the request keep their current values.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setCalibGrid( pcf::IndiProperty       &prop /**< [in,out] the property being changed */,
                      const pcf::IndiProperty &ipRecv /**< [in] the received request */,
                      std::vector<double>     &grid /**< [out] the grid setting */,
                      double                   lo /**< [in] smallest allowed value */,
                      double                   hi /**< [in] largest allowed value */ );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_enable );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_zero );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_gain );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_leak );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_rampStart );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_rampDuration );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_calibrate );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_startGrid );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_durationGrid );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_gainGrid );

    INDI_NEWCALLBACK_DECL( counterOffload, m_indiP_calibStep );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_offloadModes );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_offloadFsm );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_offloadToggle );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_stepFsm );

    INDI_SETCALLBACK_DECL( counterOffload, m_indiP_stepAmps );

    ///@}
};

inline counterOffload::counterOffload() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_ramps.resize( m_maxRamps );
}

inline void counterOffload::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    SHMIMMONITORT_SETUP_CONFIG( residualShmimMonitorT, config );

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
                "Ramp start after the woofer command, in us.  Default is 0." );

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
                "Scale applied to each new offload when it arrives, 0 to 2.  Default is 0.5." );

    config.add( "counterOffload.maxCommand",
                "",
                "counterOffload.maxCommand",
                argType::Required,
                "counterOffload",
                "maxCommand",
                false,
                "real",
                "Largest magnitude of any pixel of the command and of the held contribution; larger values are clipped.  "
                "Default is 1.0." );

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

    config.add( "calibrate.offloadDevice",
                "",
                "calibrate.offloadDevice",
                argType::Required,
                "calibrate",
                "offloadDevice",
                false,
                "string",
                "INDI device providing numModes and the offload toggle.  Default is t2wOffloader." );

    config.add( "calibrate.stepDevice",
                "",
                "calibrate.stepDevice",
                argType::Required,
                "calibrate",
                "stepDevice",
                false,
                "string",
                "dmMode INDI device whose mode is stepped.  Default is wooferModes." );

    config.add( "calibrate.csvDir",
                "",
                "calibrate.csvDir",
                argType::Required,
                "calibrate",
                "csvDir",
                false,
                "string",
                "Directory for the per-run calibration CSV files.  Default is /home/xsup/counterOffload." );

    for( const auto &k : calibKeys() )
    {
        const std::string name = std::string( "calibrate." ) + k.key;

        config.add( name, "", name, argType::Required, "calibrate", k.key, false, k.real ? "real" : "int", k.help );
    }
}

inline int counterOffload::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::m_shmimName = "aol0_tweeterOffload";

    SHMIMMONITOR_LOAD_CONFIG( _config );

    residualShmimMonitorT::m_shmimName = "aol1_modevalWFS_cbuff";

    SHMIMMONITORT_LOAD_CONFIG( residualShmimMonitorT, _config );

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

    _config( m_offloadDevice, "calibrate.offloadDevice" );
    _config( m_stepDevice, "calibrate.stepDevice" );
    _config( m_csvDir, "calibrate.csvDir" );

    for( const auto &k : calibKeys() )
    {
        double value = calibKeyValue( k );

        _config( value, std::string( "calibrate." ) + k.key );

        if( applyCalibKey( k, value ) < 0 )
        {
            return log<text_log, -1>( std::string( "invalid calibrate." ) + k.key + ": " + std::to_string( value ),
                                      logPrio::LOG_ERROR );
        }
    }

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
        !inRange( m_leak, 0, 1 ) || !inRange( m_timeOffsetUs, -1e6, 1e6 ) || !inRange( m_gain, 0, 2 ) ||
        !inRange( m_computeLeadUs, 0, 1e6 ) || !inRange( m_maxCommand, 1e-6, 1e3 ) )
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
        { m_indiP_rampStart, "rampStartUs", 0, 1e6, m_rampStartUs, INDI_NEWCALLBACK( m_indiP_rampStart ) },
        { m_indiP_rampDuration, "rampDurationUs", 1, 1e6, m_rampDurationUs, INDI_NEWCALLBACK( m_indiP_rampDuration ) } };

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

    if( registerGrid( m_indiP_startGrid, "calibStartGridUs", m_startGridUs, INDI_NEWCALLBACK( m_indiP_startGrid ) ) <
            0 ||
        registerGrid( m_indiP_durationGrid,
                      "calibDurationGridUs",
                      m_durationGridUs,
                      INDI_NEWCALLBACK( m_indiP_durationGrid ) ) < 0 ||
        registerGrid( m_indiP_gainGrid, "calibGainGrid", m_gainGrid, INDI_NEWCALLBACK( m_indiP_gainGrid ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    createROIndiNumber( m_indiP_calibStep, "calibStep", "calibStep" );
    m_indiP_calibStep.setPerm( pcf::IndiProperty::ReadWrite );
    indi::addNumberElement<double>( m_indiP_calibStep, "mode", 0, 1e6, 1, "%g" );
    indi::addNumberElement<double>( m_indiP_calibStep, "amp", 0, 1e6, 0, "%g" );
    m_indiP_calibStep["mode"] = m_stepMode;
    m_indiP_calibStep["amp"]  = m_stepAmp;

    if( registerIndiPropertyNew( m_indiP_calibStep, INDI_NEWCALLBACK( m_indiP_calibStep ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    createROIndiNumber( m_indiP_stats, "stats", "Counters" );
    for( const char *el : { "events", "published", "failed", "clipped" } )
    {
        m_indiP_stats.add( pcf::IndiElement( el ) );
        m_indiP_stats[el] = 0;
    }
    if( registerIndiPropertyReadOnly( m_indiP_stats ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    createROIndiNumber( m_indiP_latency, "latency", "Offload Latency" );
    for( const std::string &el : latencyElements() )
    {
        m_indiP_latency.add( pcf::IndiElement( el ) );
        m_indiP_latency[el] = -1;
    }
    if( registerIndiPropertyReadOnly( m_indiP_latency ) < 0 )
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
         { "stage", "point", "ratio", "bestStartUs", "bestDurationUs", "bestGain", "bestRatio", "confirmRatio" } )
    {
        m_indiP_calibration.add( pcf::IndiElement( el ) );
        m_indiP_calibration[el] = 0;
    }
    if( registerIndiPropertyReadOnly( m_indiP_calibration ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    log<text_log>( calibSettingsString(), logPrio::LOG_DEBUG );

    REG_INDI_SETPROP( m_indiP_offloadModes, m_offloadDevice, std::string( "numModes" ) );

    REG_INDI_SETPROP( m_indiP_offloadFsm, m_offloadDevice, std::string( "fsm" ) );

    REG_INDI_SETPROP( m_indiP_offloadToggle, m_offloadDevice, std::string( "offload" ) );

    REG_INDI_SETPROP( m_indiP_stepFsm, m_stepDevice, std::string( "fsm" ) );

    REG_INDI_SETPROP( m_indiP_stepAmps, m_stepDevice, std::string( "current_amps" ) );

    SHMIMMONITOR_APP_STARTUP;

    SHMIMMONITORT_APP_STARTUP( residualShmimMonitorT );

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

    SHMIMMONITORT_APP_LOGIC( residualShmimMonitorT );

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    SHMIMMONITORT_UPDATE_INDI( residualShmimMonitorT );

    std::vector<double> stats;
    uint64_t            overflows;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );

        stats     = { double( m_events ), double( m_published ), double( m_failed ), double( m_clipped ) };
        overflows = m_overflows;
    }

    updateIfChanged( m_indiP_stats, std::vector<std::string>{ "events", "published", "failed", "clipped" }, stats );

    latencySummaryT lat;
    double          missed;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );

        lat    = summarizeLatency( m_latency.data(), m_latencyCount );
        missed = double( m_missed );
    }

    const int64_t slotNs = m_slotNs;

    updateIfChanged( m_indiP_latency,
                     latencyElements(),
                     std::vector<double>{ lat.pipelineUs,
                                          lat.pipelineRmsUs,
                                          ( slotNs > 0 ) ? slotNs / 1000.0 : -1,
                                          lat.handoffUs,
                                          lat.slotWaitUs,
                                          lat.computeUs,
                                          lat.writeUs,
                                          lat.computeMaxUs,
                                          missed } );

    if( overflows != m_overflowsLogged ) // at most one warning per appLogic cycle
    {
        log<text_log>( "ramp pool full: " + std::to_string( overflows - m_overflowsLogged ) +
                           " offloads folded without a ramp",
                       logPrio::LOG_WARNING );
        m_overflowsLogged = overflows;
    }

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
                  m_lastRatio,
                  m_calibBest[0],
                  m_calibBest[1],
                  m_calibBest[2],
                  m_calibBestRatio,
                  m_confirmRatio };
    }

    updateIfChanged( m_indiP_calibration,
                     std::vector<std::string>{ "stage",
                                               "point",
                                               "ratio",
                                               "bestStartUs",
                                               "bestDurationUs",
                                               "bestGain",
                                               "bestRatio",
                                               "confirmRatio" },
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

    SHMIMMONITORT_APP_SHUTDOWN( residualShmimMonitorT );

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

    timespec receivedIsio;
    clock_gettime( CLOCK_ISIO, &receivedIsio );

    // The timestamps are read from the stream metadata, absent when the frame does not come from an attached stream.
    const int64_t handoff =
        ( shmimMonitorT::m_imageStream.md != nullptr )
            ? stampedHandoffNs( shmimMonitorT::currAtime(), shmimMonitorT::currWritetime(), receivedIsio )
            : -1;

    if( m_collecting ) // calibration: the woofer stamp locates this offload in the residual frames
    {
        const int64_t stampNs = ( handoff >= 0 ) ? toNs( receivedIsio ) - handoff : toNs( receivedIsio );

        std::lock_guard<std::mutex> stampGuard( m_stampMutex );
        m_offloadStamps.push_back( stampNs );
    }

    if( !m_enabled )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_stateMutex );

    rampT *r =
        addRamp( static_cast<const realT *>( curr_src ), received + static_cast<int64_t>( m_timeOffsetUs * 1000 ) );

    if( r )
    {
        r->wakeNs         = received;
        r->handoffNs      = handoff;
        r->latencyPending = true;
    }

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

inline counterOffload::rampT *counterOffload::addRamp( const realT *delta, int64_t t0Ns )
{
    if( m_accum.size() == 0 )
    {
        return nullptr;
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

    slot->wakeNs         = 0;
    slot->handoffNs      = -1;
    slot->latencyPending = false;

    ++m_events;

    return slot;
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
        r.used           = false;
        r.latencyPending = false; // dropped offloads are never carried
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

inline void
counterOffload::recordLatency( int64_t computeStartNs, int64_t computeEndNs, int64_t writeStartNs, int64_t writeEndNs )
{
    const int64_t slotNs = m_slotNs;

    for( auto &r : m_ramps )
    {
        if( !r.latencyPending || r.wakeNs > computeStartNs ) // arrived after this slot was computed
        {
            continue;
        }

        latencySampleT &s = m_latency[m_latencyNext];

        s.handoffUs  = ( r.handoffNs >= 0 ) ? r.handoffNs / 1000.0 : -1;
        s.slotWaitUs = ( computeStartNs - r.wakeNs ) / 1000.0;
        s.computeUs  = ( computeEndNs - computeStartNs ) / 1000.0;
        s.writeUs    = ( writeEndNs - writeStartNs ) / 1000.0;

        m_latencyNext  = ( m_latencyNext + 1 ) % m_latencyWindow;
        m_latencyCount = std::min( m_latencyCount + 1, m_latencyWindow );

        if( slotNs > 0 && computeStartNs - r.wakeNs >= slotNs )
        {
            ++m_missed;
        }

        r.latencyPending = false;
    }
}

inline void counterOffload::updateSlotPeriod( const synco::Request &request )
{
    // Consecutive steps of the same cycle are exactly one slot apart.
    if( request.epoch == m_lastRequest.epoch && request.generation == m_lastRequest.generation &&
        request.cycle == m_lastRequest.cycle && request.step == m_lastRequest.step + 1 &&
        request.targetNs > m_lastRequest.targetNs )
    {
        m_slotNs = request.targetNs - m_lastRequest.targetNs;
    }

    m_lastRequest = request;
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

inline int counterOffload::allocate( const residualShmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    if( residualShmimMonitorT::m_dataType != _DATATYPE_FLOAT )
    {
        return log<text_log, -1>( "residual stream is not float", logPrio::LOG_ERROR );
    }

    std::lock_guard<std::mutex> lock( m_calibMutex );

    if( m_calibRunning )
    {
        m_calibAbort = true;
        log<text_log>( "residual stream changed during calibration", logPrio::LOG_WARNING );
    }

    m_residualValues = residualShmimMonitorT::m_width * residualShmimMonitorT::m_height;

    return 0;
}

inline int counterOffload::processImage( void *curr_src, const residualShmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    if( !m_collecting )
    {
        return 0;
    }

    timespec atime;

    if( residualShmimMonitorT::m_imageStream.md != nullptr )
    {
        atime = residualShmimMonitorT::currAtime();
    }
    else // not from an attached stream
    {
        clock_gettime( CLOCK_ISIO, &atime );
    }

    m_lastFrameNs = synco::monotonicNs();

    std::lock_guard<std::mutex> lock( m_calibMutex );

    if( !m_collecting || m_stepData.nModes > m_residualValues )
    {
        return 0;
    }

    const float *src = static_cast<const float *>( curr_src );

    m_stepData.frames.insert( m_stepData.frames.end(), src, src + m_stepData.nModes );
    m_stepData.frameNs.push_back( toNs( atime ) );

    return 0;
}

inline int64_t counterOffload::toNs( const timespec &ts )
{
    return static_cast<int64_t>( ts.tv_sec ) * 1000000000 + ts.tv_nsec;
}

inline size_t counterOffload::findSteps( const stepDataT &d, double onsetSigma, std::vector<stepT> &steps )
{
    const size_t M       = d.nModes;
    const size_t nFrames = ( M > 0 ) ? std::min( d.frames.size() / M, d.frameNs.size() ) : 0;

    steps.assign( d.cmdIndex.size(), stepT() );

    size_t found = 0;

    for( size_t k = 0; k < d.cmdIndex.size(); ++k )
    {
        stepT &st = steps[k];

        const size_t cmd = std::min( d.cmdIndex[k], nFrames );
        st.cmd           = cmd;
        st.end           = ( k + 1 < d.cmdIndex.size() ) ? std::min( d.cmdIndex[k + 1], nFrames ) : nFrames;

        const size_t pre0 = ( cmd > m_preFrames ) ? cmd - m_preFrames : 0;
        const size_t nPre = cmd - pre0;

        if( nPre < m_minPreFrames || M == 0 )
        {
            continue;
        }

        st.preMean.assign( M, 0 );

        for( size_t i = pre0; i < cmd; ++i )
        {
            for( size_t m = 0; m < M; ++m )
            {
                st.preMean[m] += d.frames[i * M + m];
            }
        }

        for( double &v : st.preMean )
        {
            v /= nPre;
        }

        auto dev2 = [&]( size_t i )
        {
            double e = 0;
            for( size_t m = 0; m < M; ++m )
            {
                const double x = d.frames[i * M + m] - st.preMean[m];
                e += x * x;
            }
            return e;
        };

        double sum = 0, sum2 = 0;

        for( size_t i = pre0; i < cmd; ++i )
        {
            const double e = dev2( i );
            sum += e;
            sum2 += e * e;
        }

        const double mean = sum / nPre;
        const double sd   = std::sqrt( std::max( 0.0, sum2 / nPre - mean * mean ) );

        st.noiseVar = mean;

        for( size_t i = cmd; i < st.end; ++i )
        {
            if( dev2( i ) > mean + onsetSigma * sd )
            {
                st.onset = i;
                st.found = true;
                break;
            }
        }

        if( !st.found )
        {
            continue;
        }

        ++found;

        for( int64_t t : d.offloadNs )
        {
            const size_t i = std::lower_bound( d.frameNs.begin(), d.frameNs.begin() + nFrames, t ) - d.frameNs.begin();

            if( i >= cmd && i < st.end )
            {
                st.offloads.push_back( i );
            }
        }

        const auto first = std::lower_bound( st.offloads.begin(), st.offloads.end(), st.onset );

        st.offload = ( first != st.offloads.end() ) ? static_cast<long>( *first ) : -1;
    }

    return found;
}

inline size_t counterOffload::alignSteps( const stepDataT           &d,
                                          std::vector<stepT>        &steps,
                                          const std::vector<double> &stepOnly,
                                          size_t                     matchFrames,
                                          size_t                     maxShift )
{
    const size_t M = d.nModes;

    if( M == 0 || stepOnly.empty() )
    {
        return 0;
    }

    size_t moved = 0;

    for( size_t k = 0; k < steps.size(); ++k )
    {
        stepT &st = steps[k];

        if( !st.found )
        {
            continue;
        }

        const size_t len = stepOnly.size() / M;
        const size_t lo  = std::max( st.cmd, ( st.onset > maxShift ) ? st.onset - maxShift : 0 );

        const auto   next = std::lower_bound( st.offloads.begin(), st.offloads.end(), lo );
        const size_t q    = ( next != st.offloads.end() ) ? *next : st.end;
        const size_t hi   = std::min( { st.end, std::max( q, lo + m_minMatchFrames ), lo + matchFrames + 2 * maxShift } );

        if( hi <= lo )
        {
            continue;
        }

        auto cost = [&]( size_t o )
        {
            double c = 0;
            for( size_t t = lo; t < hi; ++t )
            {
                for( size_t m = 0; m < M; ++m )
                {
                    const double s = ( t >= o && t - o < len ) ? stepOnly[( t - o ) * M + m] : 0;
                    const double e = d.signs[k] * ( d.frames[t * M + m] - st.preMean[m] ) - s;
                    c += e * e;
                }
            }
            return c;
        };

        size_t best     = st.onset;
        double bestCost = cost( st.onset );

        for( size_t o = lo; o <= st.onset + maxShift; ++o )
        {
            const double c = cost( o );

            if( c < bestCost ) // ties keep the detected onset, or the earlier candidate
            {
                best     = o;
                bestCost = c;
            }
        }

        if( best != st.onset )
        {
            st.onset = best;
            ++moved;

            const auto first = std::lower_bound( st.offloads.begin(), st.offloads.end(), st.onset );

            st.offload = ( first != st.offloads.end() ) ? static_cast<long>( *first ) : -1;
        }
    }

    return moved;
}

inline void counterOffload::onsetDelayStats( const std::vector<stepT> &steps, double &mean, double &sd )
{
    double sum = 0, sum2 = 0;
    size_t n   = 0;

    for( const stepT &st : steps )
    {
        if( st.found )
        {
            const double v = double( st.onset ) - double( st.cmd );
            sum += v;
            sum2 += v * v;
            ++n;
        }
    }

    mean = ( n > 0 ) ? sum / n : 0;
    sd   = ( n > 1 ) ? std::sqrt( std::max( 0.0, ( sum2 - sum * sum / n ) / ( n - 1 ) ) ) : 0;
}

inline size_t counterOffload::stepOnlyResponse( const stepDataT          &d,
                                                const std::vector<stepT> &steps,
                                                size_t                    len,
                                                std::vector<double>      &response )
{
    const size_t M = d.nModes;

    response.assign( len * M, 0 );

    size_t n = 0;

    for( size_t k = 0; k < steps.size(); ++k )
    {
        const stepT &st = steps[k];

        if( !st.found || st.onset + len > st.end )
        {
            continue;
        }

        for( size_t j = 0; j < len; ++j )
        {
            for( size_t m = 0; m < M; ++m )
            {
                response[j * M + m] += d.signs[k] * ( d.frames[( st.onset + j ) * M + m] - st.preMean[m] );
            }
        }

        ++n;
    }

    if( n > 0 )
    {
        for( double &v : response )
        {
            v /= n;
        }
    }

    return n;
}

inline counterOffload::stepFomT counterOffload::stepFom( const stepDataT           &d,
                                                         const std::vector<stepT>  &steps,
                                                         const std::vector<double> &stepOnly,
                                                         size_t                     stepOnlyCount,
                                                         size_t                     lags )
{
    stepFomT out;

    const size_t M   = d.nModes;
    const size_t len = ( M > 0 ) ? stepOnly.size() / M : 0;

    if( M == 0 || lags == 0 || stepOnlyCount == 0 )
    {
        return out;
    }

    std::vector<double> sum( lags * M, 0 );
    std::vector<double> half[2] = { std::vector<double>( lags * M, 0 ), std::vector<double>( lags * M, 0 ) };
    std::vector<size_t> onLag( lags, 0 );

    size_t nHalf[2] = { 0, 0 };
    double noise    = 0;

    double delaySum = 0, delaySum2 = 0, phaseSum = 0, phaseSum2 = 0;

    for( size_t k = 0; k < steps.size(); ++k )
    {
        const stepT &st = steps[k];

        if( !st.found || st.offload < 0 )
        {
            continue;
        }

        const size_t off   = static_cast<size_t>( st.offload );
        const size_t phase = off - st.onset;

        if( phase > m_maxPhaseFrames || phase + lags > len || off + lags > st.end )
        {
            continue;
        }

        std::vector<double> &h = half[out.used % 2];

        for( size_t l = 0; l < lags; ++l )
        {
            for( size_t m = 0; m < M; ++m )
            {
                const double e = d.signs[k] * ( d.frames[( off + l ) * M + m] - st.preMean[m] ) -
                                 stepOnly[( phase + l ) * M + m];

                sum[l * M + m] += e;
                h[l * M + m] += e;
            }
        }

        for( size_t q : st.offloads )
        {
            if( q >= off && q < off + lags )
            {
                ++onLag[q - off];

                if( q - off + 1 < lags )
                {
                    ++onLag[q - off + 1];
                }
            }
        }

        noise += st.noiseVar;

        const double delay = double( st.onset ) - double( st.cmd );

        delaySum += delay;
        delaySum2 += delay * delay;
        phaseSum += phase;
        phaseSum2 += double( phase ) * phase;

        ++nHalf[out.used % 2];
        ++out.used;
    }

    if( out.used == 0 )
    {
        return out;
    }

    noise /= out.used;

    auto sdOf = [&]( double sum, double sum2 )
    {
        return ( out.used > 1 ) ? std::sqrt( std::max( 0.0, ( sum2 - sum * sum / out.used ) / ( out.used - 1 ) ) ) : 0.0;
    };

    out.onsetDelay   = delaySum / out.used;
    out.onsetDelaySd = sdOf( delaySum, delaySum2 );
    out.phase        = phaseSum / out.used;
    out.phaseSd      = sdOf( phaseSum, phaseSum2 );

    // Energy of an average over n steps, less its noise bias, for each lag.
    auto energy = [&]( const std::vector<double> &acc, size_t n, size_t l )
    {
        double e = 0;
        for( size_t m = 0; m < M; ++m )
        {
            const double v = acc[l * M + m] / n;
            e += v * v;
        }
        return e - noise * ( 1.0 / n + 1.0 / stepOnlyCount );
    };

    double halfFom[2] = { 0, 0 };

    for( size_t l = 0; l < lags; ++l )
    {
        const double e = energy( sum, out.used, l );

        out.fom += e;

        if( 2 * onLag[l] > out.used )
        {
            out.fomTiming += e;
        }

        for( int h = 0; h < 2; ++h )
        {
            if( nHalf[h] > 0 )
            {
                halfFom[h] += energy( half[h], nHalf[h], l );
            }
        }
    }

    out.fomGain = out.fom - out.fomTiming;
    out.fomStd  = ( nHalf[1] > 0 ) ? std::fabs( halfFom[0] - halfFom[1] ) / 2 : 0;

    return out;
}

inline double counterOffload::interpBaseline( int64_t                     tNs,
                                              const std::vector<int64_t> &offTimes,
                                              const std::vector<double>  &offFoms )
{
    if( offTimes.empty() )
    {
        return 0;
    }

    if( tNs <= offTimes.front() )
    {
        return offFoms.front();
    }

    if( tNs >= offTimes.back() )
    {
        return offFoms.back();
    }

    // offTimes[i - 1] <= tNs < offTimes[i]
    const size_t i = std::upper_bound( offTimes.begin(), offTimes.end(), tNs ) - offTimes.begin();
    const double w = double( tNs - offTimes[i - 1] ) / double( offTimes[i] - offTimes[i - 1] );

    return offFoms[i - 1] + w * ( offFoms[i] - offFoms[i - 1] );
}

inline std::vector<size_t> counterOffload::sweepOrder( size_t n )
{
    size_t bits = 0;
    while( ( size_t( 1 ) << bits ) < n )
    {
        ++bits;
    }

    std::vector<size_t> order;

    for( size_t k = 0; k < ( size_t( 1 ) << bits ); ++k )
    {
        size_t r = 0;
        for( size_t b = 0; b < bits; ++b )
        {
            if( k & ( size_t( 1 ) << b ) )
            {
                r |= size_t( 1 ) << ( bits - 1 - b );
            }
        }

        if( r < n )
        {
            order.push_back( r );
        }
    }

    return order;
}

inline counterOffload::quadFitT counterOffload::fitQuadratic( const std::vector<double> &xs,
                                                              const std::vector<double> &rs )
{
    quadFitT     fit;
    const size_t n = xs.size();

    if( n < 3 || rs.size() != n )
    {
        return fit;
    }

    const auto [xlo, xhi] = std::minmax_element( xs.begin(), xs.end() );

    fit.x0    = 0.5 * ( *xhi + *xlo );
    fit.scale = 0.5 * ( *xhi - *xlo );

    if( !( fit.scale > 0 ) )
    {
        return fit;
    }

    // Normal equations A c = y, with A_jk = sum u^(j+k) and y_j = sum u^j r.
    std::array<double, 5> su{ 0, 0, 0, 0, 0 };
    std::array<double, 3> y{ 0, 0, 0 };

    for( size_t i = 0; i < n; ++i )
    {
        const double u  = ( xs[i] - fit.x0 ) / fit.scale;
        double       uk = 1;

        for( size_t k = 0; k < 5; ++k )
        {
            su[k] += uk;
            if( k < 3 )
            {
                y[k] += uk * rs[i];
            }
            uk *= u;
        }
    }

    typedef std::array<std::array<double, 3>, 3> mat3T;

    auto det3 = []( const mat3T &m )
    {
        return m[0][0] * ( m[1][1] * m[2][2] - m[1][2] * m[2][1] ) - m[0][1] * ( m[1][0] * m[2][2] - m[1][2] * m[2][0] ) +
               m[0][2] * ( m[1][0] * m[2][1] - m[1][1] * m[2][0] );
    };

    const mat3T  A{ { { su[0], su[1], su[2] }, { su[1], su[2], su[3] }, { su[2], su[3], su[4] } } };
    const double det = det3( A );

    if( !( std::fabs( det ) > 1e-9 * su[0] * su[0] * su[0] ) )
    {
        return fit;
    }

    for( size_t j = 0; j < 3; ++j ) // Cramer's rule
    {
        mat3T Aj = A;
        for( size_t k = 0; k < 3; ++k )
        {
            Aj[k][j] = y[k];
        }
        fit.c[j] = det3( Aj ) / det;
    }

    fit.ok = true;

    if( n > 3 )
    {
        double ss = 0;
        for( size_t i = 0; i < n; ++i )
        {
            const double d = rs[i] - fit( xs[i] );
            ss += d * d;
        }
        fit.resSd = std::sqrt( ss / ( n - 3 ) );
    }

    return fit;
}

inline counterOffload::choiceT counterOffload::chooseValue( const std::vector<double> &xs,
                                                            const std::vector<double> &rs,
                                                            double                     current,
                                                            double                     sigma,
                                                            double                     moveSigma )
{
    choiceT      ch;
    const size_t n = xs.size();

    ch.candidate = current;

    if( n == 0 || rs.size() != n )
    {
        return ch;
    }

    std::vector<size_t> idx( n );
    std::iota( idx.begin(), idx.end(), 0 );
    std::sort( idx.begin(), idx.end(), [&xs]( size_t a, size_t b ) { return xs[a] < xs[b]; } );

    const double lo = xs[idx.front()];
    const double hi = xs[idx.back()];

    ch.outside = ( current < lo - 1e-9 || current > hi + 1e-9 );

    const quadFitT fit = ( n >= 4 ) ? fitQuadratic( xs, rs ) : quadFitT();

    if( fit.ok )
    {
        // The lower end, or the vertex of an upward parabola if it is inside.
        double v = ( fit( hi ) < fit( lo ) ) ? hi : lo;

        if( fit.c[2] > 0 )
        {
            const double vertex = fit.x0 - fit.scale * fit.c[1] / ( 2 * fit.c[2] );

            if( vertex >= lo && vertex <= hi )
            {
                v = vertex;
            }
        }

        ch.candidate  = v;
        ch.rCandidate = fit( v );
        ch.rCurrent   = ch.outside ? 0 : fit( current );
    }
    else
    {
        // 3-point running mean over the points in x order, interpolated linearly at the current value.
        std::vector<double> sx( n ), sm( n );

        for( size_t k = 0; k < n; ++k )
        {
            const size_t j0 = ( k > 0 ) ? k - 1 : 0;
            const size_t j1 = std::min( k + 1, n - 1 );

            double sum = 0;
            for( size_t j = j0; j <= j1; ++j )
            {
                sum += rs[idx[j]];
            }

            sx[k] = xs[idx[k]];
            sm[k] = sum / ( j1 - j0 + 1 );
        }

        const size_t kmin = std::min_element( sm.begin(), sm.end() ) - sm.begin();

        ch.candidate  = sx[kmin];
        ch.rCandidate = sm[kmin];

        if( !ch.outside )
        {
            ch.rCurrent = sm[0];

            for( size_t k = 0; k + 1 < n; ++k )
            {
                if( current >= sx[k] && current <= sx[k + 1] )
                {
                    const double w = ( sx[k + 1] > sx[k] ) ? ( current - sx[k] ) / ( sx[k + 1] - sx[k] ) : 0;
                    ch.rCurrent    = sm[k] + w * ( sm[k + 1] - sm[k] );
                    break;
                }
            }
        }
    }

    ch.threshold = moveSigma * sigma * std::sqrt( 6.0 / n );
    ch.move      = ch.outside || ( ch.rCurrent - ch.rCandidate > ch.threshold );

    return ch;
}

inline int counterOffload::estimatePoints()
{
    auto sweepPoints = [this]( const std::vector<double> &g )
    {
        const int n = static_cast<int>( std::floor( ( g[1] - g[0] ) / g[2] + 1e-9 ) ) + 1;
        return n + ( n - 1 ) / m_baselineEvery + 1; // points, the baselines within, and the closing baseline
    };

    const int startGain = sweepPoints( m_startGridUs ) + sweepPoints( m_gainGrid );

    return ( 2 * m_replicates + 1 ) + startGain + sweepPoints( m_durationGridUs ) + ( m_rounds - 1 ) * startGain +
           2 * m_confirmPairs;
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
    static const char *names[] = { "off",       "initial",      "start",    "duration", "gain",    "confirm",
                                   "clipped",   "fit start",    "fit duration", "fit gain", "step only" };

    return ( stage >= stageOff && stage <= stageStepOnly ) ? names[stage] : "unknown";
}

inline double counterOffload::clipFraction( uint64_t clipped, uint64_t published )
{
    return ( published > 0 ) ? double( clipped ) / published : 0;
}

inline const std::vector<std::string> &counterOffload::latencyElements()
{
    static const std::vector<std::string> names{ "pipelineUs", "pipelineRmsUs", "slotUs",       "handoffUs", "slotWaitUs",
                                                 "computeUs",  "writeUs",       "computeMaxUs", "missed" };

    return names;
}

inline int64_t
counterOffload::stampedHandoffNs( const timespec &atime, const timespec &writetime, const timespec &receivedIsio )
{
    if( atime.tv_sec == writetime.tv_sec && atime.tv_nsec == writetime.tv_nsec ) // posted without a woofer stamp
    {
        return -1;
    }

    const int64_t handoff = ( static_cast<int64_t>( receivedIsio.tv_sec ) - atime.tv_sec ) * 1000000000 +
                            ( static_cast<int64_t>( receivedIsio.tv_nsec ) - atime.tv_nsec );

    return ( handoff >= 0 && handoff <= m_maxHandoffNs ) ? handoff : -1;
}

inline counterOffload::latencySummaryT counterOffload::summarizeLatency( const latencySampleT *samples, size_t n )
{
    latencySummaryT out;

    double sumWait = 0, sumCompute = 0, sumWrite = 0, maxCompute = 0;
    double sumHandoff = 0, sumPipe = 0, sumPipe2 = 0;
    size_t nStamped = 0;

    for( size_t k = 0; k < n; ++k )
    {
        const latencySampleT &s = samples[k];

        sumWait += s.slotWaitUs;
        sumCompute += s.computeUs;
        sumWrite += s.writeUs;
        maxCompute = std::max( maxCompute, s.computeUs );

        if( s.handoffUs >= 0 )
        {
            const double pipe = s.handoffUs + s.computeUs + s.writeUs;

            sumHandoff += s.handoffUs;
            sumPipe += pipe;
            sumPipe2 += pipe * pipe;
            ++nStamped;
        }
    }

    if( n > 0 )
    {
        out.slotWaitUs   = sumWait / n;
        out.computeUs    = sumCompute / n;
        out.writeUs      = sumWrite / n;
        out.computeMaxUs = maxCompute;
    }

    if( nStamped > 0 )
    {
        out.handoffUs     = sumHandoff / nStamped;
        out.pipelineUs    = sumPipe / nStamped;
        out.pipelineRmsUs = std::sqrt( std::max( 0.0, sumPipe2 / nStamped - out.pipelineUs * out.pipelineUs ) );
    }

    return out;
}

inline bool counterOffload::gridValid( const std::vector<double> &grid, double lo, double hi )
{
    auto inRange = []( double v, double a, double b )
    {
        return mx::math::isFinite( v ) && v >= a && v <= b;
    };

    return grid.size() == 3 && inRange( grid[0], lo, hi ) && inRange( grid[1], grid[0], hi ) &&
           inRange( grid[2], 1e-9, hi - lo );
}

template <typename T>
inline int counterOffload::applyCalibSetting( T &setting, double value, double lo, double hi )
{
    if( m_calibRunning || !mx::math::isFinite( value ) || value < lo || value > hi ||
        ( std::is_integral_v<T> && value != std::round( value ) ) )
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_stateMutex );
    setting = static_cast<T>( value );

    return 0;
}

inline int
counterOffload::applyCalibGrid( std::vector<double> &grid, const std::vector<double> &value, double lo, double hi )
{
    if( m_calibRunning || !gridValid( value, lo, hi ) )
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_stateMutex );
    grid = value;

    return 0;
}

inline std::vector<counterOffload::calibKeyT> counterOffload::calibKeys()
{
    return {
        { "modeMax",
          nullptr,
          &m_modeMax,
          0,
          1e6,
          "Number of residual modes summed.  0 uses the offloader's numModes.  Default is 0." },
        { "baselineEvery",
          nullptr,
          &m_baselineEvery,
          1,
          1e6,
          "Measure a gain-0 baseline after every this many points of a sweep.  Default is 4." },
        { "settleS",
          &m_settleS,
          nullptr,
          0,
          1e3,
          "Wait after a change, for the held contribution to leak away, before the steps, s.  Default is 0.2." },
        { "timeoutS",
          &m_timeoutS,
          nullptr,
          1,
          1e4,
          "Longest wait for residual frames or a change of the offload toggle, s.  Default is 5." },
        { "abortRatio",
          &m_abortRatio,
          nullptr,
          1,
          1e6,
          "Abort when a measurement exceeds this multiple of the baseline.  Default is 3." },
        { "leak",
          &m_calibLeak,
          nullptr,
          1e-9,
          1,
          "Leak used during a calibration run, restored afterwards; must be > 0.  Default is 0.0025." },
        { "replicates",
          nullptr,
          &m_replicates,
          2,
          20,
          "Measurements of the starting parameters for the first noise estimate.  Default is 3." },
        { "confirmPairs",
          nullptr,
          &m_confirmPairs,
          1,
          20,
          "Measurements of the chosen parameters, each after a baseline, to confirm them.  Default is 3." },
        { "moveSigma",
          &m_moveSigma,
          nullptr,
          0,
          10,
          "A parameter moves only if the fit improves by this many standard errors.  Default is 1.5." },
        { "sigmaFloor", &m_sigmaFloor, nullptr, 0, 1, "Smallest noise estimate of the ratio used.  Default is 0.002." },
        { "rounds",
          nullptr,
          &m_rounds,
          1,
          5,
          "Rounds of sweeps; rounds after the first sweep only rampStartUs and gain.  Default is 2." },
        { "stepPeriodS",
          &m_stepPeriodS,
          nullptr,
          0.05,
          1e3,
          "Time between woofer steps; must exceed the offload train's convergence, s.  Default is 0.5." },
        { "stepCount",
          nullptr,
          &m_stepCount,
          2,
          1000,
          "Woofer steps per measurement, alternating in sign.  Default is 10." },
        { "onsetSigma",
          &m_onsetSigma,
          nullptr,
          0,
          1e3,
          "Step onset threshold, in standard deviations of the pre-step residual energy.  Default is 5." } };
}

inline int counterOffload::applyCalibStep( double mode, double amp )
{
    if( m_calibRunning || !mx::math::isFinite( mode ) || mode < 0 || mode > 9999 || mode != std::round( mode ) ||
        !mx::math::isFinite( amp ) || !( amp > 0 ) )
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_stateMutex );
    m_stepMode = static_cast<int>( mode );
    m_stepAmp  = amp;

    return 0;
}

inline std::string counterOffload::stepElementName( int mode )
{
    std::ostringstream out;
    out << std::setw( 4 ) << std::setfill( '0' ) << mode;

    return out.str();
}

inline int counterOffload::applyCalibKey( const calibKeyT &k, double value )
{
    if( k.real )
    {
        return applyCalibSetting( *k.real, value, k.lo, k.hi );
    }

    return applyCalibSetting( *k.integer, value, k.lo, k.hi );
}

inline double counterOffload::calibKeyValue( const calibKeyT &k )
{
    return k.real ? *k.real : *k.integer;
}

inline std::string counterOffload::calibSettingsString()
{
    std::ostringstream out;
    out << "calibration settings:";

    std::lock_guard<std::mutex> guard( m_stateMutex );

    for( const auto &k : calibKeys() )
    {
        out << ' ' << k.key << '=' << calibKeyValue( k );
    }

    const std::pair<const char *, const std::vector<double> *> grids[] = { { "startGridUs", &m_startGridUs },
                                                                           { "durationGridUs", &m_durationGridUs },
                                                                           { "gainGrid", &m_gainGrid } };

    for( const auto &[name, grid] : grids )
    {
        out << ' ' << name << '=' << ( *grid )[0] << ',' << ( *grid )[1] << ',' << ( *grid )[2];
    }

    out << " stepDevice=" << m_stepDevice << " stepMode=" << m_stepMode << " stepAmp=" << m_stepAmp;

    return out.str();
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
    constexpr double   none     = std::numeric_limits<double>::max(); // no ratio measured yet

    log<text_log>( calibSettingsString(), logPrio::LOG_DEBUG ); // record the settings this run uses

    if( !startChecks() )
    {
        m_calibRunning = false;
        return;
    }

    double originalLeak;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        originalLeak = m_leak;
        m_leak       = m_calibLeak; // keep the held contribution small so gain steps cannot saturate the tweeter
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_calibPoint     = 0;
        m_calibStage     = stageOff;
        m_calibBest      = original;
        m_calibBestRatio = none;
        m_confirmRatio   = 0;
    }

    m_offTimes.clear();
    m_offFoms.clear();
    m_rows.clear();
    m_noise        = noiseT();
    m_lastWasOff   = false;
    m_measuredRows = 0;
    m_clippedRows  = 0;

    m_stepOnly.clear();
    m_stepOnlyCount = 0;
    m_lagFrames     = 0;

    const double perPoint = m_settleS + m_stepCount * m_stepPeriodS;
    const int    points   = estimatePoints();
    const double minutes  = ( points * perPoint + 2 * ( perPoint + 3 * m_stepPeriodS ) ) / 60;

    log<text_log>( "calibration: about " + std::to_string( points ) + " measurements, " +
                       std::to_string( minutes ) + " min",
                   ( minutes > m_maxRunMin ) ? logPrio::LOG_WARNING : logPrio::LOG_INFO );

    calibParamsT        cur       = original;
    double              curRatio  = none;
    double              confirm   = none;
    double              confirmSe = 0;
    std::vector<double> rs;
    bool                offloadOff = false;

    // Start at the -a position so that every measured step is a full 2a step, then measure the step alone, then
    // estimate the noise from replicates of the starting point.
    m_stepPos = -1;

    bool ok = openCsv() && sendStepAmp( m_stepBase - m_stepAmp ) && calibWait( m_stepPeriodS, true ) &&
              measureStepOnly( offloadOff ) && replicate( original, stageInitial, m_replicates, rs );

    if( ok && !rs.empty() )
    {
        const double mean = std::accumulate( rs.begin(), rs.end(), 0.0 ) / rs.size();
        double       ss   = 0;

        for( double r : rs )
        {
            ss += ( r - mean ) * ( r - mean );
        }

        if( rs.size() > 1 )
        {
            m_noise.add( std::sqrt( ss / ( rs.size() - 1 ) ), static_cast<int>( rs.size() ) - 1 );
        }

        curRatio = mean;

        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_calibBestRatio = curRatio;
    }

    // Sweeps: start, gain, then duration in the first round; start and gain in later rounds.
    for( int round = 0; ok && round < m_rounds; ++round )
    {
        const std::vector<int> axes = ( round == 0 ) ? std::vector<int>{ 0, 2, 1 } : std::vector<int>{ 0, 2 };

        for( size_t k = 0; ok && k < axes.size(); ++k )
        {
            ok = sweep( axes[k], cur, curRatio );
        }
    }

    // Remeasure the step alone (its change over the run is logged), then confirm the chosen parameters.
    if( ok )
    {
        ok = measureStepOnly( offloadOff ) && replicate( cur, stageConfirm, m_confirmPairs, rs );
    }

    if( ok && !rs.empty() )
    {
        confirm = std::accumulate( rs.begin(), rs.end(), 0.0 ) / rs.size();

        double ss = 0;
        for( double r : rs )
        {
            ss += ( r - confirm ) * ( r - confirm );
        }

        confirmSe = ( rs.size() > 1 ) ? std::sqrt( ss / ( rs.size() - 1 ) / rs.size() ) : 0;

        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_confirmRatio = confirm;
    }

    flushRows(); // anything held when a run aborts

    if( m_clippedRows > 0 )
    {
        log<text_log>( "calibration: the command clipped during " + std::to_string( m_clippedRows ) + " of " +
                           std::to_string( m_measuredRows ) + " measurements; all were kept",
                       logPrio::LOG_INFO );
    }

    // Always restore the stepped mode and the offloader, whatever happened.
    if( !sendStepAmp( m_stepBase ) )
    {
        log<text_log>( "calibration could not restore " + m_stepDevice + " mode " + m_stepElement + " to " +
                           std::to_string( m_stepBase ),
                       logPrio::LOG_ERROR );
    }

    if( offloadOff && !setOffloading( true ) )
    {
        log<text_log>( "calibration could not turn " + m_offloadDevice + " offload back on", logPrio::LOG_ERROR );
    }

    const std::string params = "rampStartUs=" + std::to_string( cur[0] ) + " rampDurationUs=" +
                               std::to_string( cur[1] ) + " gain=" + std::to_string( cur[2] );

    if( ok && confirm < 1 )
    {
        setCalibParams( cur );

        log<text_log>( "calibration complete, applied " + params + " confirmed ratio=" + std::to_string( confirm ) +
                           " +/- " + std::to_string( confirmSe ),
                       logPrio::LOG_INFO );
    }
    else if( !ok && curRatio < 1 ) // an aborted run still keeps an improvement
    {
        setCalibParams( cur );

        log<text_log>( "calibration aborted, applied the parameters chosen so far: " + params +
                           " predicted ratio=" + std::to_string( curRatio ),
                       logPrio::LOG_WARNING );
    }
    else
    {
        setCalibParams( original );

        log<text_log>( std::string( ok ? "calibration found no confirmed improvement" : "calibration aborted" ) +
                           ", restored rampStartUs=" + std::to_string( original[0] ) +
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

inline bool counterOffload::startChecks()
{
    int mode;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        mode = m_stepMode;
    }

    std::string why;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );

        m_stepModes   = ( m_modeMax > 0 ) ? m_modeMax : m_offloadModes;
        m_stepElement = stepElementName( mode );

        const auto amp = m_stepAmps.find( m_stepElement );

        if( stateCodes::str2Code( m_offloadState ) != stateCodes::OPERATING )
        {
            why = m_offloadDevice + " is not OPERATING";
        }
        else if( !m_offloading )
        {
            why = m_offloadDevice + " offload is off";
        }
        else if( stateCodes::str2Code( m_stepState ) != stateCodes::READY )
        {
            why = m_stepDevice + " is not READY";
        }
        else if( amp == m_stepAmps.end() )
        {
            why = m_stepDevice + " has no mode " + m_stepElement;
        }
        else if( m_stepModes == 0 )
        {
            why = "no numModes from " + m_offloadDevice;
        }
        else if( m_residualValues < m_stepModes )
        {
            why = "the residual stream is not connected, or has fewer values than the " +
                  std::to_string( m_stepModes ) + " modes";
        }
        else
        {
            m_stepBase = amp->second;
        }
    }

    if( !why.empty() )
    {
        log<text_log>( "calibration not started: " + why, logPrio::LOG_WARNING );
        return false;
    }

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

    m_csv << "time,stage,rampStartUs,rampDurationUs,gain,stepMode,stepAmp,stepCount,numModes,fom,fomStd,fomOff,"
             "ratio,ratioInterp,fomTiming,fomGain,fitCurrent,fitCandidate,threshold,moved,sigma,clippedSlots,clipFrac,"
             "onsetDelay,onsetDelaySd,phase,phaseSd\n";

    log<text_log>( "calibration started, stepping " + m_stepDevice + " mode " + m_stepElement + " by +/- " +
                       std::to_string( m_stepAmp ) + " about " + std::to_string( m_stepBase ) + ", over " +
                       std::to_string( m_stepModes ) + " modes, writing " + path,
                   logPrio::LOG_INFO );

    return true;
}

inline bool counterOffload::calibWait( double seconds, bool needOffload )
{
    const int64_t until     = synco::monotonicNs() + static_cast<int64_t>( seconds * 1e9 );
    const int64_t timeoutNs = static_cast<int64_t>( m_timeoutS * 1e9 );

    std::unique_lock<std::mutex> lock( m_calibMutex );

    while( true )
    {
        const int64_t now = synco::monotonicNs();

        std::string why;

        if( m_calibAbort || m_shutdown )
        {
            why = "abort requested";
        }
        else if( !m_enabled )
        {
            why = "app disabled";
        }
        else if( m_modeMax == 0 && m_offloadModes != m_stepModes )
        {
            why = "numModes changed";
        }
        else if( needOffload && !m_offloading )
        {
            why = m_offloadDevice + " offload turned off";
        }
        else if( m_collecting && now - m_lastFrameNs > timeoutNs )
        {
            why = "no residual frames for " + std::to_string( m_timeoutS ) + " s";
        }

        if( !why.empty() )
        {
            log<text_log>( "calibration: " + why, logPrio::LOG_WARNING );
            return false;
        }

        if( now >= until )
        {
            return true;
        }

        m_calibCond.wait_for( lock, std::chrono::nanoseconds( std::min<int64_t>( until - now, 10000000 ) ) );
    }
}

inline bool counterOffload::sendStepAmp( double amp )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_stepDevice );
    ip.setName( "target_amps" );
    ip.add( pcf::IndiElement( m_stepElement ) );
    ip[m_stepElement] = amp;

    if( sendNewProperty( ip ) < 0 )
    {
        log<text_log>( "calibration could not send " + m_stepDevice + ".target_amps", logPrio::LOG_ERROR );
        return false;
    }

    return true;
}

inline bool counterOffload::setOffloading( bool on )
{
    if( sendNewStandardIndiToggle( m_offloadDevice, "offload", on ) < 0 )
    {
        log<text_log>( "calibration could not send " + m_offloadDevice + ".offload", logPrio::LOG_ERROR );
        return false;
    }

    const int64_t until = synco::monotonicNs() + static_cast<int64_t>( m_timeoutS * 1e9 );

    while( m_offloading != on )
    {
        if( m_shutdown || synco::monotonicNs() > until )
        {
            log<text_log>( "calibration: " + m_offloadDevice + " offload did not turn " + ( on ? "on" : "off" ),
                           logPrio::LOG_WARNING );
            return false;
        }

        mx::sys::milliSleep( 10 );
    }

    return true;
}

inline bool counterOffload::collectSteps( stepDataT &d, bool needOffload, uint64_t &clipped, uint64_t &published )
{
    const double seriesS = m_settleS + m_stepCount * m_stepPeriodS;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );

        // Room for the series at up to 4 kHz, so collecting does not reallocate while holding the lock.
        const size_t frames = static_cast<size_t>( seriesS * 4000 ) + 1;

        m_stepData        = stepDataT();
        m_stepData.nModes = m_stepModes;
        m_stepData.frames.reserve( frames * m_stepModes );
        m_stepData.frameNs.reserve( frames );
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stampMutex );
        m_offloadStamps.clear();
        m_offloadStamps.reserve( static_cast<size_t>( seriesS * 2000 ) + 1 );
    }

    m_lastFrameNs = synco::monotonicNs();
    m_collecting  = true;

    bool ok = calibWait( m_settleS, needOffload ); // the previous point's held contribution leaks away

    const uint64_t clipStart      = m_clipped;
    const uint64_t publishedStart = m_published;

    for( int k = 0; ok && k < m_stepCount; ++k )
    {
        m_stepPos = -m_stepPos;

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_calibMutex );
            m_stepData.cmdIndex.push_back( m_stepData.frameNs.size() );
            m_stepData.signs.push_back( m_stepPos );
        }

        ok = sendStepAmp( m_stepBase + m_stepPos * m_stepAmp ) && calibWait( m_stepPeriodS, needOffload );
    }

    m_collecting = false;

    clipped   = m_clipped - clipStart;
    published = m_published - publishedStart;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        d          = std::move( m_stepData );
        m_stepData = stepDataT();
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stampMutex );
        d.offloadNs.swap( m_offloadStamps );
        m_offloadStamps.clear();
    }

    std::sort( d.offloadNs.begin(), d.offloadNs.end() );

    return ok;
}

inline bool counterOffload::measureStepOnly( bool &offloadOff )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_calibStage = stageStepOnly;
    }

    if( !setOffloading( false ) )
    {
        return false;
    }

    offloadOff = true;

    stepDataT d;
    uint64_t  clipped, published;

    const bool collected = collectSteps( d, false, clipped, published );

    if( setOffloading( true ) ) // whatever happened
    {
        offloadOff = false;
    }

    if( !collected || offloadOff )
    {
        return false;
    }

    std::vector<stepT> steps;
    const size_t       found = findSteps( d, m_onsetSigma, steps );
    const size_t       need  = std::max<size_t>( 2, m_stepCount / 2 );

    size_t span = std::numeric_limits<size_t>::max();

    for( const stepT &st : steps )
    {
        if( st.found )
        {
            span = std::min( span, st.end - st.onset );
        }
    }

    if( found < need || span <= m_guardFrames + m_maxPhaseFrames + 1 )
    {
        log<text_log>( "calibration: found the onset of " + std::to_string( found ) + " of " +
                           std::to_string( steps.size() ) +
                           " woofer steps; check the step device, calibStep and onsetSigma",
                       logPrio::LOG_WARNING );
        return false;
    }

    const size_t        len = span - m_guardFrames;
    std::vector<double> response;
    const size_t        n = stepOnlyResponse( d, steps, len, response );

    double energy = 0;

    for( double v : response )
    {
        energy += v * v;
    }

    // RMS change from the previous measurement over the lags both have, relative to the previous response.
    std::string change;

    if( !m_stepOnly.empty() )
    {
        const size_t common = std::min( m_stepOnly.size(), response.size() );
        double       diff = 0, prev = 0;

        for( size_t i = 0; i < common; ++i )
        {
            diff += ( response[i] - m_stepOnly[i] ) * ( response[i] - m_stepOnly[i] );
            prev += m_stepOnly[i] * m_stepOnly[i];
        }

        if( prev > 0 )
        {
            change = ", RMS change from the previous " + std::to_string( std::sqrt( diff / prev ) );
        }
    }

    m_stepOnly      = std::move( response );
    m_stepOnlyCount = n;
    m_lagFrames     = len - m_maxPhaseFrames;

    double delay, delaySd;
    onsetDelayStats( steps, delay, delaySd );

    log<text_log>( "calibration: step-only response from " + std::to_string( n ) + " steps over " +
                       std::to_string( len ) + " frames, energy " + std::to_string( energy ) + ", onset delay " +
                       std::to_string( delay ) + " +/- " + std::to_string( delaySd ) + " frames" + change,
                   logPrio::LOG_INFO );

    timespec wall;
    clock_gettime( CLOCK_REALTIME, &wall );

    const calibParamsT p = calibParams();

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );

        writeCsvPrefix( wall, stageStepOnly, p );
        m_csv << "," << energy << ",,,,,,,,,,,,," << "," << delay << "," << delaySd << ",,\n";
        m_csv.flush();
    }

    return calibWait( 2 * m_stepPeriodS, true ); // the offload train started by turning offload back on settles
}

inline bool counterOffload::measure( const calibParamsT &p, stepFomT &f, uint64_t &clipped, uint64_t &published )
{
    setCalibParams( p );

    stepDataT d;

    if( !collectSteps( d, true, clipped, published ) )
    {
        return false;
    }

    std::vector<stepT> steps;
    const size_t       found = findSteps( d, m_onsetSigma, steps );

    alignSteps( d, steps, m_stepOnly, m_matchFrames, m_maxShiftFrames );

    f = stepFom( d, steps, m_stepOnly, m_stepOnlyCount, m_lagFrames );

    const size_t need = std::max<size_t>( 2, m_stepCount / 2 );

    if( f.used < need )
    {
        log<text_log>( "calibration: only " + std::to_string( f.used ) + " of " + std::to_string( steps.size() ) +
                           " woofer steps usable (" + std::to_string( found ) + " onsets, " +
                           std::to_string( d.offloadNs.size() ) +
                           " offloads); check the offload stamps and the residual atime",
                       logPrio::LOG_WARNING );
        return false;
    }

    return true;
}

inline bool counterOffload::measureBaseline( const calibParamsT &p )
{
    calibParamsT off = p;
    off[2]           = 0;

    stepFomT f;
    uint64_t clipped, published;

    if( !measure( off, f, clipped, published ) )
    {
        log<text_log>( "calibration: the gain-0 baseline could not be measured", logPrio::LOG_WARNING );
        return false;
    }

    if( !( f.fom > 0 ) )
    {
        log<text_log>( "calibration: the gain-0 baseline shows no offload error", logPrio::LOG_WARNING );
        return false;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_fomOff     = f.fom;
        m_calibStage = stageOff;
    }

    addRow( stageOff, off, f, clipped, published );

    m_offTimes.push_back( m_rows.back().tNs );
    m_offFoms.push_back( f.fom );
    m_lastWasOff = true;

    return true;
}

inline bool counterOffload::ensureBaseline( const calibParamsT &p )
{
    return m_lastWasOff || measureBaseline( p );
}

inline bool counterOffload::measurePoint( const calibParamsT &p, int stage, double &fom )
{
    stepFomT f;
    uint64_t clipped, published;

    if( !measure( p, f, clipped, published ) )
    {
        return false;
    }

    fom          = f.fom;
    m_lastWasOff = false;

    addRow( stage, p, f, clipped, published );

    const double ratio = fom / m_fomOff;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_calibStage = stage;
        ++m_calibPoint;
        m_lastRatio = ratio;
    }

    if( ratio > m_abortRatio )
    {
        log<text_log>( "calibration: step error " + std::to_string( ratio ) + " times the baseline, aborting",
                       logPrio::LOG_WARNING );
        return false;
    }

    return true;
}

inline bool counterOffload::replicate( const calibParamsT &p, int stage, int n, std::vector<double> &ratios )
{
    ratios.clear();

    if( !ensureBaseline( p ) )
    {
        return false;
    }

    std::vector<size_t> rows;

    for( int i = 0; i < n; ++i )
    {
        double fom;

        if( !measurePoint( p, stage, fom ) )
        {
            return false;
        }

        rows.push_back( m_rows.size() - 1 );

        if( !measureBaseline( p ) )
        {
            return false;
        }
    }

    for( size_t k : rows )
    {
        ratios.push_back( m_rows[k].fom / interpBaseline( m_rows[k].tNs, m_offTimes, m_offFoms ) );
    }

    flushRows();

    return true;
}

inline bool counterOffload::sweep( int axis, calibParamsT &cur, double &curRatio )
{
    static const char *axisName[] = { "rampStartUs", "rampDurationUs", "gain" };

    const std::vector<double> &grid  = ( axis == 0 ) ? m_startGridUs : ( axis == 1 ) ? m_durationGridUs : m_gainGrid;
    const int                  stage = stageStart + axis;
    const size_t               n     = static_cast<size_t>( std::floor( ( grid[1] - grid[0] ) / grid[2] + 1e-9 ) ) + 1;

    if( !ensureBaseline( cur ) )
    {
        return false;
    }

    std::vector<double> xs;
    std::vector<size_t> rows;

    int since = 0; // points since the latest baseline

    for( size_t i : sweepOrder( n ) )
    {
        const double x = grid[0] + i * grid[2];

        if( since == m_baselineEvery )
        {
            if( !measureBaseline( cur ) )
            {
                return false;
            }

            since = 0;
        }

        calibParamsT p = cur;
        p[axis]        = x;

        double fom;

        if( !measurePoint( p, stage, fom ) )
        {
            return false;
        }

        ++since;

        xs.push_back( x );
        rows.push_back( m_rows.size() - 1 );
    }

    if( !measureBaseline( cur ) )
    {
        return false;
    }

    // Ratios to the interpolated baseline, clipped points included.
    std::vector<double> fx, fr;

    for( size_t k = 0; k < xs.size(); ++k )
    {
        fx.push_back( xs[k] );
        fr.push_back( m_rows[rows[k]].fom / interpBaseline( m_rows[rows[k]].tNs, m_offTimes, m_offFoms ) );
    }

    const quadFitT fit = ( fx.size() >= 4 ) ? fitQuadratic( fx, fr ) : quadFitT();

    if( fit.ok )
    {
        m_noise.add( fit.resSd, static_cast<int>( fx.size() ) - 3 );
    }

    const double  sigma = m_noise.sigma( m_sigmaFloor );
    const choiceT ch    = chooseValue( fx, fr, cur[axis], sigma, m_moveSigma );

    flushRows();

    calibParamsT pc = cur;
    pc[axis]        = ch.candidate;

    writeFitRow( stageFitStart + axis, pc, ch, sigma );

    if( fx.empty() )
    {
        log<text_log>( std::string( "calibration: no usable points for " ) + axisName[axis] + ", kept " +
                           std::to_string( cur[axis] ),
                       logPrio::LOG_WARNING );
    }
    else
    {
        log<text_log>( std::string( "calibration: " ) + axisName[axis] + " candidate " +
                           std::to_string( ch.candidate ) + " (ratio " + std::to_string( ch.rCandidate ) + "), current " +
                           std::to_string( cur[axis] ) +
                           ( ch.outside ? " outside the sweep" : " (ratio " + std::to_string( ch.rCurrent ) + ")" ) +
                           ", threshold " + std::to_string( ch.threshold ) + ": " + ( ch.move ? "moved" : "kept" ),
                       logPrio::LOG_INFO );

        if( ch.move )
        {
            cur[axis] = ch.candidate;
            curRatio  = ch.rCandidate;
        }
        else
        {
            curRatio = ch.rCurrent;
        }
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_calibStage     = stageFitStart + axis;
        m_calibBest      = cur;
        m_calibBestRatio = curRatio;
    }

    return true;
}

inline void
counterOffload::addRow( int stage, const calibParamsT &p, const stepFomT &f, uint64_t clipped, uint64_t published )
{
    calibRowT row;

    clock_gettime( CLOCK_REALTIME, &row.wall );
    row.tNs       = synco::monotonicNs();
    row.stage     = stage;
    row.p         = p;
    row.fom       = f.fom;
    row.fomStd    = f.fomStd;
    row.fomOff    = m_fomOff;
    row.fomTiming = f.fomTiming;
    row.fomGain   = f.fomGain;

    row.onsetDelay   = f.onsetDelay;
    row.onsetDelaySd = f.onsetDelaySd;
    row.phase        = f.phase;
    row.phaseSd      = f.phaseSd;

    row.clippedSlots   = clipped;
    row.publishedSlots = published;

    m_rows.push_back( row );

    ++m_measuredRows;

    if( clipped > 0 )
    {
        ++m_clippedRows;

        log<text_log>( std::string( "calibration: " ) + calibStageName( stage ) + " rampStartUs=" +
                           std::to_string( p[0] ) + " rampDurationUs=" + std::to_string( p[1] ) +
                           " gain=" + std::to_string( p[2] ) + " clipped in " + std::to_string( clipped ) + " of " +
                           std::to_string( published ) + " slots (" +
                           std::to_string( clipFraction( clipped, published ) ) + "), kept",
                       logPrio::LOG_INFO );
    }
}

inline void counterOffload::flushRows()
{
    std::lock_guard<std::mutex> lock( m_calibMutex );

    for( const calibRowT &row : m_rows )
    {
        writeCsvPrefix( row.wall, row.stage, row.p );

        m_csv << "," << row.fom << "," << row.fomStd << "," << row.fomOff << "," << row.fom / row.fomOff << ","
              << row.fom / interpBaseline( row.tNs, m_offTimes, m_offFoms ) << "," << row.fomTiming << ","
              << row.fomGain << ",,,,,," << row.clippedSlots << ","
              << clipFraction( row.clippedSlots, row.publishedSlots ) << "," << row.onsetDelay << ","
              << row.onsetDelaySd << "," << row.phase << "," << row.phaseSd << "\n";
    }

    m_rows.clear();
    m_csv.flush();
}

inline void counterOffload::writeCsvPrefix( const timespec &wall, int stage, const calibParamsT &p )
{
    m_csv << std::fixed << wall.tv_sec << "." << std::setw( 9 ) << std::setfill( '0' ) << wall.tv_nsec
          << std::setfill( ' ' ) << std::defaultfloat << "," << calibStageName( stage ) << "," << p[0] << "," << p[1]
          << "," << p[2] << "," << m_stepElement << "," << m_stepAmp << "," << m_stepCount << "," << m_stepModes;
}

inline void counterOffload::writeFitRow( int stage, const calibParamsT &p, const choiceT &ch, double sigma )
{
    timespec wall;
    clock_gettime( CLOCK_REALTIME, &wall );

    std::lock_guard<std::mutex> lock( m_calibMutex );

    writeCsvPrefix( wall, stage, p );

    m_csv << ",,,,,,,,";

    if( !ch.outside )
    {
        m_csv << ch.rCurrent;
    }

    m_csv << "," << ch.rCandidate << "," << ch.threshold << "," << int( ch.move ) << "," << sigma << ",,,,,,\n";
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

        updateSlotPeriod( request );

        if( !m_leadLogged )
        {
            log<text_log>( "slot lead: computeLeadUs " + std::to_string( m_computeLeadUs ) + " + writeLeadUs " +
                           std::to_string( m_writeLeadUs ) + " + synco readyLeadUs " +
                           std::to_string( ( request.targetNs - request.cutoffNs ) / 1000.0 ) + " = " +
                           std::to_string( ( request.targetNs - request.cutoffNs ) / 1000.0 + m_writeLeadUs +
                                           m_computeLeadUs ) +
                           " us" );
            m_leadLogged = true;
        }

        // Compute as late as possible so the most recent offload is included.
        const int64_t computeAt = request.cutoffNs - static_cast<int64_t>( ( m_writeLeadUs + m_computeLeadUs ) * 1000 );
        const timespec wake     = synco::toTimespec( computeAt );

        while( synco::monotonicNs() < computeAt && !m_shutdown )
        {
            clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr );
        }

        bool    zero           = false;
        int64_t computeStartNs = 0;
        int64_t computeEndNs   = 0;

        { //mutex scope
            std::lock_guard<std::mutex> guard( m_stateMutex );

            if( m_command.size() == 0 || !needPublish() )
            {
                continue;
            }

            computeStartNs = synco::monotonicNs();

            zero = m_zeroRequested;

            if( zero )
            {
                clearState();
            }
            else
            {
                buildCommand( request.targetNs );
            }

            computeEndNs = synco::monotonicNs();
        }

        // syncopationPublish waits until its publish lead before copying, so the write starts at the later of the two.
        const int64_t publishAt =
            request.cutoffNs - static_cast<int64_t>( std::ceil( m_writeLeadUs * 1000 ) );
        const int64_t writeStartNs = std::max( synco::monotonicNs(), publishAt );

        const bool published = dmSyncopationProducerT::syncopationPublish( m_command.data(), request, m_auxStream );

        const int64_t writeEndNs = synco::monotonicNs();

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
        else
        {
            recordLatency( computeStartNs, computeEndNs, writeStartNs, writeEndNs );
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

inline int counterOffload::registerGrid( pcf::IndiProperty         &prop,
                                         const std::string         &name,
                                         const std::vector<double> &grid,
                                         int ( *callback )( void *, const pcf::IndiProperty & ) )
{
    createROIndiNumber( prop, name, name );
    prop.setPerm( pcf::IndiProperty::ReadWrite );

    const char *els[] = { "min", "max", "step" };

    for( size_t i = 0; i < 3; ++i )
    {
        indi::addNumberElement<double>( prop, els[i], 0, 1e6, 0, "%g" );
        prop[els[i]] = grid[i];
    }

    return registerIndiPropertyNew( prop, callback );
}

inline int counterOffload::setCalibGrid(
    pcf::IndiProperty &prop, const pcf::IndiProperty &ipRecv, std::vector<double> &grid, double lo, double hi )
{
    const std::vector<std::string> els{ "min", "max", "step" };

    std::vector<double> value = grid;

    for( size_t i = 0; i < els.size(); ++i )
    {
        if( ipRecv.find( els[i] ) )
        {
            value[i] = ipRecv[els[i]].get<double>();
        }
    }

    if( applyCalibGrid( grid, value, lo, hi ) < 0 )
    {
        return log<text_log, -1>( "rejected " + prop.getName() + ": need min <= max within " + std::to_string( lo ) +
                                      " to " + std::to_string( hi ) + " and step > 0, and no calibration running",
                                  logPrio::LOG_ERROR );
    }

    updateIfChanged( prop, els, grid );

    log<text_log>( "set " + prop.getName() + " to " + std::to_string( grid[0] ) + "," + std::to_string( grid[1] ) +
                       "," + std::to_string( grid[2] ),
                   logPrio::LOG_NOTICE );

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

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_startGrid )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_startGrid, ipRecv );
    return setCalibGrid( m_indiP_startGrid, ipRecv, m_startGridUs, 0, 1e6 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_durationGrid )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_durationGrid, ipRecv );
    return setCalibGrid( m_indiP_durationGrid, ipRecv, m_durationGridUs, 1, 1e6 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_gainGrid )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gainGrid, ipRecv );
    return setCalibGrid( m_indiP_gainGrid, ipRecv, m_gainGrid, 0, 2 );
}

INDI_NEWCALLBACK_DEFN( counterOffload, m_indiP_calibStep )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_calibStep, ipRecv );

    double mode, amp;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_stateMutex );
        mode = m_stepMode;
        amp  = m_stepAmp;
    }

    if( ipRecv.find( "mode" ) )
    {
        mode = ipRecv["mode"].get<double>();
    }

    if( ipRecv.find( "amp" ) )
    {
        amp = ipRecv["amp"].get<double>();
    }

    if( applyCalibStep( mode, amp ) < 0 )
    {
        return log<text_log, -1>( "rejected calibStep: need an integer mode from 0 to 9999 and amp > 0, and no "
                                  "calibration running",
                                  logPrio::LOG_ERROR );
    }

    updateIfChanged( m_indiP_calibStep, std::vector<std::string>{ "mode", "amp" }, std::vector<double>{ mode, amp } );

    log<text_log>( "set calibStep to mode " + std::to_string( int( mode ) ) + " amp " + std::to_string( amp ),
                   logPrio::LOG_NOTICE );

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

INDI_SETCALLBACK_DEFN( counterOffload, m_indiP_offloadFsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadFsm, ipRecv );

    if( ipRecv.find( "state" ) )
    {
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_offloadState = ipRecv["state"].get<std::string>();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( counterOffload, m_indiP_offloadToggle )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadToggle, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        m_offloading = ( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( counterOffload, m_indiP_stepFsm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stepFsm, ipRecv );

    if( ipRecv.find( "state" ) )
    {
        std::lock_guard<std::mutex> lock( m_calibMutex );
        m_stepState = ipRecv["state"].get<std::string>();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( counterOffload, m_indiP_stepAmps )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stepAmps, ipRecv );

    std::lock_guard<std::mutex> lock( m_calibMutex );

    for( const auto &[name, el] : ipRecv.getElements() )
    {
        m_stepAmps[name] = el.get<double>();
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // counterOffload_hpp
